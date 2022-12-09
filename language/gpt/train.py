from colossalai.zero.init_ctx import ZeroInitContext
import contextlib
import os
import colossalai
import colossalai.utils as utils
import torch
import torch.nn as nn
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.engine.schedule import (InterleavedPipelineSchedule,
                                        PipelineSchedule)
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn import LinearWarmupLR
from colossalai.trainer import Trainer, hooks
from colossalai.utils import is_using_pp
from colossalai.utils.timer import MultiTimer
from titans.loss.lm_loss import GPTLMLoss
from dataset.dataset_wb import WBdistDataset
import json
from transformers import GPT2LMHeadModel, CONFIG_NAME, GPT2Config, BertTokenizer

ATTR_TO_SPECIAL_TOKEN = {'additional_special_tokens':['[next]', '[user]', '[assistant]', '[pediatrics]', '[gynecology]']}

def add_special_tokens(model, tokenizer):
    orig_num_tokens = len(tokenizer.vocab)
    num_add_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
    if num_add_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_add_tokens)



def main():
    ###启动 Colossal-AI
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
        colossalai.launch_from_slurm(config=args.config,
                                     host=args.host,
                                     port=29500,
                                     seed=42)
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

    print('gpc :{}'.format(gpc.config))

    train_dataloader = utils.get_dataloader(train_ds,
                                            seed=42,
                                            batch_size=gpc.config.BATCH_SIZE,
                                            pin_memory=True,
                                            shuffle=True,
                                            drop_last=True,
                                            num_workers=8,
                                            collate_fn=train_ds.collate)

    ###构建 ZeRO GPT-2 模型
    logger.info('Build model', ranks=[0])
    use_pipeline = is_using_pp()
    use_interleaved = hasattr(gpc.config.model, 'num_chunks')
    use_zero3 = hasattr(gpc.config, 'zero')
    ctx = contextlib.nullcontext()
    if use_zero3:
        ctx = ZeroInitContext(target_device=torch.cuda.current_device(),
                              shard_strategy=gpc.config.zero.model_config.shard_strategy,
                              shard_param=True
                              )
    with ctx:
        model = gpc.config.model.pop('type')(**gpc.config.model)
    if use_pipeline and use_interleaved and not isinstance(model, nn.ModuleList):
        model = nn.ModuleList([model])

    print(model)
    ###定义优化器，损失函数和学习率调度器
    criterion = getattr(gpc.config, 'loss_fn', None)
    if criterion is not None:
        criterion = criterion.type()
    else:
        criterion = GPTLMLoss()
    logger.info('Build optimizer', ranks=[0])
    optimizer = gpc.config.optimizer.pop('type')(
        model.parameters(), **gpc.config.optimizer)
    lr_scheduler = LinearWarmupLR(
        optimizer, total_steps=gpc.config.NUM_EPOCHS, warmup_steps=5)

    ###启动用于训练的 Colossal-AI engine
    engine, train_dataloader, _, lr_scheduler = colossalai.initialize(model,
                                                                      optimizer,
                                                                      criterion,
                                                                      train_dataloader=train_dataloader,
                                                                      lr_scheduler=lr_scheduler)
    global_batch_size = gpc.config.BATCH_SIZE * \
        gpc.get_world_size(ParallelMode.DATA) * getattr(gpc.config, "gradient_accumulation", 1)
    logger.info(f'Init done, global batch size = {global_batch_size}', ranks=[0])
    timier = MultiTimer()

    trainer = Trainer(
        engine=engine,
        logger=logger,
        timer=timier
    )
    hook_list = [
        hooks.LossHook(),
        hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=True),
        hooks.LogMetricByEpochHook(logger),
        hooks.ThroughputHook(),
        hooks.LogMetricByStepHook(),
        # hooks.TensorboardHook(log_dir='./tb_logs', ranks=[0]),
        # hooks.LogMemoryByEpochHook(logger),
        # hooks.LogTimingByEpochHook(timer, logger),
        # hooks.SaveCheckpointHook(checkpoint_dir='./ckpt')
    ]

    trainer.fit(train_dataloader=train_dataloader,
                epochs=gpc.config.NUM_EPOCHS,
                test_interval=1,
                hooks=hook_list,
                display_progress=True,
                return_output_label=False)


if __name__ == '__main__':
    main()
