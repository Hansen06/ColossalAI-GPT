import contextlib
import os

import colossalai
import colossalai.utils as utils
import torch
import torch.nn as nn
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai import nn as col_nn
from colossalai.engine.schedule import (InterleavedPipelineSchedule, PipelineSchedule)
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn import LinearWarmupLR
from colossalai.trainer import Trainer, hooks
from colossalai.utils import is_using_pp, colo_set_process_memory_fraction
from colossalai.utils.timer import MultiTimer
from colossalai.zero.init_ctx import ZeroInitContext
from colossalai.pipeline.pipelinable import PipelinableContext
import colossalai
import psutil
import torch
import torch.nn as nn
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import HybridAdam
from functools import partial
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.utils import get_current_device
from colossalai.nn.parallel import ZeroDDP
from colossalai.zero import ZeroOptimizer
from colossalai.tensor import ProcessGroup
from functools import partial
from gpt_model import GPTLMLoss, gpt2_small
from dataset_wb import WBdistDataset
import json
from transformers import GPT2LMHeadModel, GPT2Config, BertTokenizer
from packaging import version

PLACEMENT_POLICY = 'auto'


def calc_local_model_size(model: torch.nn.Module):
    numel_per_device = 0
    for p in model.parameters():
        numel_per_device += p.numel()
    return numel_per_device

def get_cpu_mem():
    return psutil.Process().memory_info().rss / 1024**2

def get_gpu_mem():
    return torch.cuda.memory_allocated() / 1024**2

def get_mem_info(prefix=''):
    return f'{prefix}GPU memory usage: {get_gpu_mem():.2f} MB, CPU memory usage: {get_cpu_mem():.2f} MB'

def get_tflops(model_numel, batch_size, seq_len, step_time):
    return model_numel * batch_size * seq_len * 8 / 1e12 / (step_time + 1e-12)

def main():
    parser = colossalai.get_default_parser()
    parser.add_argument('--from_torch', default=False, action='store_true')
    parser.add_argument("--model_checkpoint", type=str, default="config/cgpt/", help="Path or URL of the model")
    parser.add_argument("--train_path", type=str, default="data/toy_train.txt",
                        help="Path of the train dataset for dist dataset. ")
    parser.add_argument("--valid_path", type=str, default="data/toy_valid.txt",
                        help="Path of the valid dataset for dist dataset. ")
    parser.add_argument("--max_history", type=int, default=15, help="Number of previous exchanges to keep in history")

    args = parser.parse_args()
    disable_existing_loggers()
    if args.from_torch:
        colossalai.launch_from_torch(config=args.config)
    else:
        colossalai.launch_from_slurm(config=args.config, host=args.host, port=29500, seed=42)

    logger = get_dist_logger()

    tokenizer = BertTokenizer(os.path.join(args.model_checkpoint, "vocab.txt"), do_lower_case=True)

    # 读取模型配置文件
    with open(os.path.join(args.model_checkpoint, 'config.json'), "r", encoding='utf-8') as reader:
        text = reader.read()
    config = json.loads(text)
    logger.info("model config : {}".format(config))

    ###构建 Webtext 加载器
    logger.info('Build data loader', ranks=[0])
    train_ds = WBdistDataset(tokenizer, data_path=args.train_path, max_history=args.max_history,
                             n_ctx=config['n_ctx'])

    train_dataloader = utils.get_dataloader(train_ds,
                                            seed=42,
                                            batch_size=gpc.config.BATCH_SIZE,
                                            pin_memory=True,
                                            shuffle=True,
                                            drop_last=True,
                                            num_workers=8,
                                            collate_fn=train_ds.collate)

    logger.info('Build model', ranks=[0])
    # build GPT model
    pg = ProcessGroup()
    with ColoInitContext(device=get_current_device()):
        model = gpt2_small(checkpoint=True)
    numel = sum([p.numel() for p in model.parameters()])
    logger.info(f'Model numel: {numel}', ranks=[0])
    get_tflops_func = partial(get_tflops, numel, BATCH_SIZE, SEQ_LEN)

    cai_version = colossalai.__version__
    logger.info(f'using Colossal-AI version {cai_version}')
    if version.parse(cai_version) > version.parse("0.1.10"):
        from colossalai.gemini import GeminiManager
        from colossalai.gemini.chunk import init_chunk_manager
        chunk_manager = init_chunk_manager(
            model=model,
            init_device=get_current_device(),
            search_range_mb=32
        )
        gemini_manager = GeminiManager(PLACEMENT_POLICY, chunk_manager)
        model = ZeroDDP(model, gemini_manager, pin_memory=True)
    elif version.parse(cai_version) <= version.parse("0.1.10") and version.parse(cai_version) >= version.parse("0.1.9"):
        from colossalai.gemini import ChunkManager, GeminiManager
        chunk_size = ChunkManager.search_chunk_size(model, 64 * 1024 ** 2, 32)
        chunk_manager = ChunkManager(chunk_size, pg, enable_distributed_storage=True,
                                     init_device=GeminiManager.get_default_device(PLACEMENT_POLICY))

    if version.parse(torch.__version__) > version.parse("0.1.11"):
        logger.error(f'{torch.__version__} may not supported, please use torch version 0.1.11')

    logger.info(get_mem_info(prefix='After init model, '), ranks=[0])

    logger.info(chunk_manager, ranks=[0])

    ###定义优化器，损失函数和学习率调度器
    criterion = GPTLMLoss()

    logger.info('Build optimizer', ranks=[0])
    optimizer = gpc.config.optimizer.pop('type')(model.parameters(), **gpc.config.optimizer)

    lr_scheduler = LinearWarmupLR(optimizer, total_steps=gpc.config.NUM_EPOCHS, warmup_steps=5)

    ###启动用于训练的 Colossal-AI engine
    engine, train_dataloader, _, lr_scheduler = colossalai.initialize(model,
                                                                      optimizer,
                                                                      criterion,
                                                                      train_dataloader=train_dataloader,
                                                                      lr_scheduler=lr_scheduler)
    global_batch_size = gpc.config.BATCH_SIZE * \
        gpc.get_world_size(ParallelMode.DATA) * getattr(gpc.config, "gradient_accumulation", 128)

    print('==========================================================')
    logger.info(f'Init done, global  sibatchze = {global_batch_size}', ranks=[0])

    timier = MultiTimer()

    trainer = Trainer(engine=engine, logger=logger, timer=timier)

    hook_list = [
        hooks.LossHook(),
        hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=True),
        hooks.LogMetricByEpochHook(logger),
        hooks.ThroughputHook(ignored_steps=10, tflop_per_step=tflop),
        hooks.LogMetricByStepHook(),
    # hooks.TensorboardHook(log_dir='./tb_logs', ranks=[0]),
        hooks.LogMemoryByEpochHook(logger),
    # hooks.LogTimingByEpochHook(timer, logger),
        hooks.SaveCheckpointHook(checkpoint_dir='./ckpt/pytorch_model.bin')
    ]

    trainer.fit(train_dataloader=train_dataloader,
                epochs=gpc.config.NUM_EPOCHS,
                test_interval=1,
                hooks=hook_list,
                display_progress=True,
                return_output_label=False)


if __name__ == '__main__':
    main()
