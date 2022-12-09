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
from titans.loss.lm_loss import GPTLMLoss
from dataset.dataset_wb import WBdistDataset
import json
from transformers import GPT2LMHeadModel, CONFIG_NAME, GPT2Config, BertTokenizer



def calc_local_model_size(model: torch.nn.Module):
    numel_per_device = 0
    for p in model.parameters():
        numel_per_device += p.numel()
    return numel_per_device


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
    use_pipeline = is_using_pp()
    use_interleaved = hasattr(gpc.config.model, 'num_chunks')
    num_chunks = getattr(gpc.config.model, 'num_chunks', 1)
    use_zero3 = hasattr(gpc.config, 'zero')

    print('use_pipeline: {}'.format(use_pipeline))
    print('num_chunks: {}'.format(num_chunks))
    print('use_zero3: {}'.format(use_zero3))

    if not use_pipeline:
        ctx = contextlib.nullcontext()
        if use_zero3:
            ctx = ZeroInitContext(target_device=torch.cuda.current_device(),
                                  shard_strategy=gpc.config.zero.model_config.shard_strategy,
                                  shard_param=True)
        with ctx:
            model = gpc.config.model.pop('type')(**gpc.config.model)
    else:
        pipelinable = PipelinableContext()
        with pipelinable:
            model = gpc.config.model.pop('type')(**gpc.config.model)

        def mask_function(attention_mask=None):
            if attention_mask is not None:
                batch_size = gpc.config.BATCH_SIZE // gpc.config.NUM_MICRO_BATCHES
                attention_mask = attention_mask.view(batch_size, -1)
                attention_mask = col_nn.partition_batch(attention_mask)
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attention_mask = (1.0 - attention_mask) * -10000.0
            return attention_mask

        # GPT2_small exec_seq
        # (lyl)TODO: The exec_seq for gpt3 will be added here and to_layer_list should be more friendly to use.
        exec_seq = ['embed', mask_function, 'blocks.0', 'blocks.1', 'blocks.2', 'blocks.3', 'blocks.4', 'blocks.5', (mask_function, "front"), \
                    'blocks.6', 'blocks.7', 'blocks.8', 'blocks.9', 'blocks.10', 'blocks.11', 'norm', 'head']
        pipelinable.to_layer_list(exec_seq)
        ctx = contextlib.nullcontext()
        # (lyl)TODO: Zero context and pipelinable context should be integrated into one context.
        if use_zero3:
            print('===============================use_zero3===============================')
            ctx = ZeroInitContext(target_device=torch.cuda.current_device(),
                                  shard_strategy=gpc.config.zero.model_config.shard_strategy,
                                  shard_param=True)
        with ctx:
            print('===============================ctx===============================')
            model = pipelinable.partition(num_chunks, gpc.pipeline_parallel_size,
                                          gpc.get_local_rank(ParallelMode.PIPELINE))

    if use_zero3:
        numel = ctx.model_numel_tensor.item()
    else:
        numel = calc_local_model_size(model)

    tflop = numel * gpc.config.BATCH_SIZE * gpc.config.SEQ_LEN \
        * gpc.get_world_size(ParallelMode.MODEL) * gpc.get_world_size(ParallelMode.DATA) * 8 / (1024 ** 4)

    print('tflop: {}'.format(tflop))

    ###定义优化器，损失函数和学习率调度器
    criterion = getattr(gpc.config, 'loss_fn', None)
    if criterion is not None:
        criterion = criterion.type()
    else:
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
