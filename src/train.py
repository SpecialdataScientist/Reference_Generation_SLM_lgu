import os
import sys

sys.path.append(os.path.abspath('../'))

import json
import math

from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)

from src import config

from src.dataset import ReferenceGeneratorDataset, ReferenceGeneratorEvaluationDataset
from src.Models import ReferenceGenerationModel
from src.preprocess import preprocess_passage

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from modules.utils.gpu_utils import set_seed, init_gpu_params
from modules.utils.checkpoint_utils import *
from modules.utils.eval_utils import gather_metric, model_validation

from transformers import PreTrainedTokenizerFast
from transformers.optimization import get_linear_schedule_with_warmup

# Train and Validation with gold passage
def train(args):
    # set logger
    report_formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    report_handler = logging.FileHandler(os.path.join(f'{args.save_file_path}/{args.task_name}/Report_reference_generate.log'))
    report_handler.setFormatter(report_formatter)
    report_handler.setLevel(logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.addHandler(report_handler)

    # load tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.transformers_generator_model_name)

    # load raw passage data
    with open(os.path.join(args.dataset_path, f'passages.json'), 'r') as f:
        raw_passage_data = json.load(f)

    # preprocess raw passage data
    id2passage = preprocess_passage(raw_passage_data)

    # set dataset
    train = ReferenceGeneratorDataset(data_type=f'train', id2passage=id2passage,
                                      tokenizer=tokenizer, args=args)
    dev = ReferenceGeneratorDataset(data_type=f'dev', id2passage=id2passage, tokenizer=tokenizer,
                                    args=args)

    if args.validation_mode == 'T':
        batch_ = args.validation_batch_size
    else:
        batch_ = args.per_gpu_batch_size

    train_sampler = DistributedSampler(train)
    dev_sampler = DistributedSampler(dev)

    train_dataloader = DataLoader(train,
                                  batch_size=batch_,
                                  sampler=train_sampler,
                                  pin_memory=True,
                                  collate_fn=train.collate_fn)
    dev_dataloader = DataLoader(dev,
                                batch_size=args.validation_batch_size,
                                sampler=dev_sampler,
                                pin_memory=True,
                                collate_fn=dev.collate_fn)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs

    # load model
    model = ReferenceGenerationModel(args)
    model.cuda()

    if args.validation_mode == 'T':
        model = load_best_model(args.save_file_path, f'{args.task_name}/best_model_reference_generate.pt', model)

    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.device_ids[args.local_rank]],
                                        output_device=args.device_ids[args.local_rank],
                                        find_unused_parameters=True)

    # validation step
    if args.validation_mode == 'T':
        if args.is_master:
            logger.info('Train Set Validation')

        validation_res = model_validation(model=model.module, dataloader=train_dataloader, tokenizer=tokenizer, args=args)

        meters = gather_metric(validation_res, data_path=os.path.join(args.check_point_path, f'{args.task_name}/train_result/train_result.json'))

        if args.is_master:
            logger.info(
                f'F1-Score: {meters["f1"]}, Meteor-Score: {meters["meteor"]}, Rouge-Score: {meters["rouge"]}, Bleu-Score: {meters["bleu"]}')
            logger.info(f'Total-Score: {meters["f1"] + meters["meteor"] + meters["rouge"] + meters["bleu"]}')

            logger.info(f'saved train validation!')

            logger.info('Dev Set Validation')

        validation_res = model_validation(model=model.module, dataloader=dev_dataloader, tokenizer=tokenizer, args=args)

        meters = gather_metric(validation_res, data_path=os.path.join(args.check_point_path, f'{args.task_name}/dev_result/dev_result.json'))

        if args.is_master:
            logger.info(
                f'F1-Score: {meters["f1"]}, Meteor-Score: {meters["meteor"]}, Rouge-Score: {meters["rouge"]}, Bleu-Score: {meters["bleu"]}')
            logger.info(f'Total-Score: {meters["f1"] + meters["meteor"] + meters["rouge"] + meters["bleu"]}')

            logger.info(f'saved dev validation!')

        return

    # set optimizer and scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    model_parameters = [p for p in model.named_parameters()]

    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model_parameters if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in model_parameters if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.lr,
                      betas=(args.adam_beta1, args.adam_beta2),
                      eps=args.adam_epsilon)

    warmup_steps = math.ceil(t_total * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_training_steps=t_total,
                                                num_warmup_steps=warmup_steps)

    if args.is_master:
        print(args)
        configuration = vars(args)
        save_configuration_path = os.path.join(args.save_file_path, f'{args.task_name}/configuration_reference_generate.json')
        with open(save_configuration_path, 'w') as fp:
            json.dump(configuration, fp, indent=2, ensure_ascii=False)

        logger.info('---------Running Training---------')
        logger.info(f' Num examples = {len(train)}')
        logger.info(f' Num Epochs = {args.epochs}')
        logger.info(f' Instantaneous batch size per GPU = {args.per_gpu_batch_size}')
        logger.info(f' Total train batch size (w. parallel, distributed & accumulation) = '
                    f'{args.per_gpu_batch_size * args.gradient_accumulation_steps * torch.distributed.get_world_size() if args.multi_gpu else 1}')
        logger.info(f' Gradient_ Accumulation steps = {args.gradient_accumulation_steps}')
        logger.info(f' Total optimization steps = {t_total}')

    # training step
    best_epoch = 0
    global_steps = 0
    best_total_score = -1
    loss = 0
    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch * 1000)

        model.train()

        losses = []
        processing_bar = tqdm(train_dataloader, desc='Iter', disable=not args.is_master)
        for ii, batch in enumerate(processing_bar):
            batch = {k: v.cuda() for k, v in batch.items()}

            output = model(encoder_input_ids=batch['encoder_input_ids'],
                           encoder_attention_mask=batch['encoder_attention_mask'],
                           decoder_input_ids=batch['decoder_input_ids'],
                           decoder_attention_mask=batch['decoder_attention_mask'],
                           labels=batch['labels'], target=batch['target'], num_beams=args.num_beams,
                           num_return_sequences=args.num_return_sequences,
                           no_repeat_ngram_size=args.no_repeat_ngram_size)

            loss = output['loss']

            if args.gradient_accumulation_steps > 1:
                loss /= args.gradient_accumulation_steps

            loss.backward()

            if (ii + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clipping)

                optimizer.step()
                scheduler.step()

                optimizer.zero_grad()

                global_steps += 1

                if global_steps % args.log_step_count_steps == 0 and args.is_master:
                    write_log(args.check_point_path, f'{args.task_name}/log_step_reference_generate.txt', processing_bar)

            losses.append(loss.item())

            processing_bar.set_postfix({
                'epoch': f'{epoch + 1}',
                'global_steps': f'{global_steps + 1}',
                'rolling_loss': f'{sum(losses) / (len(losses)) * args.gradient_accumulation_steps:.5f}',
                'last_loss': f'{loss.item() * args.gradient_accumulation_steps:.5f}',
            })

        # validation step per epoch
        if args.is_master:
            write_log(args.check_point_path, f'{args.task_name}/log_epoch_reference_generate.txt', processing_bar)
            logger.info(' ')
            logger.info('Dev Set Validation')
            logger.info(f'{epoch + 1} epoch [dev]')

        validation_res = model_validation(model=model.module, dataloader=dev_dataloader, tokenizer=tokenizer, args=args)

        meters = gather_metric(validation_res, data_path=os.path.join(args.check_point_path, f'{args.task_name}/dev_result/dev_result_epoch_{epoch + 1}.json'))

        if args.is_master:
            total_score = meters["f1"] + meters["meteor"] + meters["rouge"] + meters["bleu"]
            logger.info(
                f'F1-Score: {meters["f1"]}, Meteor-Score: {meters["meteor"]}, Rouge-Score: {meters["rouge"]}, Bleu-Score: {meters["bleu"]}')
            logger.info(f'Total-Score: {meters["f1"] + meters["meteor"] + meters["rouge"] + meters["bleu"]}')

            if best_total_score < total_score:
                best_epoch = epoch + 1
                best_total_score = total_score

                save_model_state_dict(args.save_file_path, f'{args.task_name}/best_model_reference_generate.pt',
                                      model)

                params = {
                    'state_dict': model.module if hasattr(model, 'module') else model,
                    'model_optimizer': optimizer,
                    'loss': loss,
                    'epochs': epoch + 1,
                    'total_score': total_score
                }

                check_point_path = os.path.join(args.check_point_path, f'{args.task_name}/check_model/')
                save_checkpoints(check_point_path, params)

                logger.info('saved best model...')

    if args.is_master:
        logger.info(f'Best Epoch: {best_epoch}')

        logger.info('-----Train set end!!-----')

    return

# main function
if __name__ == '__main__':
    args = config.arg_parse()

    set_seed(args)
    init_gpu_params(args)

    train(args)