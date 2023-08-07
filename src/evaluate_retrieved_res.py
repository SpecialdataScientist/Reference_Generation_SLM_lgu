import os
import sys

sys.path.append(os.path.abspath('../'))

import json

import logging
logger = logging.getLogger(__name__)

from src import config

from src.dataset import ReferenceGeneratorDataset, ReferenceGeneratorEvaluationDataset
from src.Models import ReferenceGenerationModel
from src.preprocess import preprocess_passage

from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from modules.utils.gpu_utils import set_seed, init_gpu_params
from modules.utils.checkpoint_utils import *
from modules.utils.eval_utils import gather_metric, model_validation

from transformers import PreTrainedTokenizerFast

# Evaluation with retrieved passage
def train(args):
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
    train = ReferenceGeneratorEvaluationDataset(data_type=f'train', id2passage=id2passage, tokenizer=tokenizer, args=args)
    dev = ReferenceGeneratorEvaluationDataset(data_type=f'dev', id2passage=id2passage, tokenizer=tokenizer, args=args)

    train_sampler = DistributedSampler(train)
    dev_sampler = DistributedSampler(dev)

    train_dataloader = DataLoader(train,
                                  batch_size=args.validation_batch_size,
                                  sampler=train_sampler,
                                  pin_memory=True,
                                  collate_fn=train.collate_fn)

    dev_dataloader = DataLoader(dev,
                                batch_size=args.validation_batch_size,
                                sampler=dev_sampler,
                                pin_memory=True,
                                collate_fn=dev.collate_fn)

    # load model
    model = ReferenceGenerationModel(args)
    model.cuda()

    model = load_best_model(args.save_file_path, f'{args.task_name}/best_model_reference_generate.pt', model)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.device_ids[args.local_rank]],
                                        output_device=args.device_ids[args.local_rank],
                                        find_unused_parameters=True)

    # evaluation step (retrieved passage)
    if args.is_master:
        logger.info(f'Train Set with retrieved passage Evaluation')

    evaluation_res = model_validation(model=model.module, dataloader=train_dataloader, tokenizer=tokenizer, args=args)

    meters = gather_metric(evaluation_res,
                           data_path=os.path.join(args.gen_reference_file_path, f'evaluation/train_generated_reference_dict.json'))

    if args.is_master:
        logger.info(
            f'F1-Score: {meters["f1"]}, Meteor-Score: {meters["meteor"]}, Rouge-Score: {meters["rouge"]}, Bleu-Score: {meters["bleu"]}')
        logger.info(f'Total-Score: {meters["f1"] + meters["meteor"] + meters["rouge"] + meters["bleu"]}')

        logger.info(f'saved train set Result...')

        logger.info(f'Dev Set with retrieved passage Evaluation')

    evaluation_res = model_validation(model=model.module, dataloader=dev_dataloader,
                                                         tokenizer=tokenizer, args=args)

    meters = gather_metric(evaluation_res, data_path=os.path.join(args.gen_reference_file_path, f'evaluation/dev_generated_reference_dict.json'))

    if args.is_master:
        logger.info(
            f'F1-Score: {meters["f1"]}, Meteor-Score: {meters["meteor"]}, Rouge-Score: {meters["rouge"]}, Bleu-Score: {meters["bleu"]}')
        logger.info(f'Total-Score: {meters["f1"] + meters["meteor"] + meters["rouge"] + meters["bleu"]}')

        logger.info(f'saved dev set Result...')

    return

if __name__ == '__main__':
    args = config.arg_parse()

    set_seed(args)
    init_gpu_params(args)

    train(args)