import os
import sys
sys.path.append(os.path.abspath('../'))

from tqdm import tqdm

import json

import torch.distributed as dist

from modules.utils.evaluation_utils import matching_evaluate, normalize_answer, evaluate_score

import evaluate

# model evaluation method
def model_validation(model, dataloader, tokenizer, args):
    model.eval()
    
    res_dict = {'query_id': [], 'pred_list': [], 'label_list': []}

    processing_bar = tqdm(dataloader, desc='Iter', disable=not args.is_master)
    for ii, batch in enumerate(processing_bar):
        batch = {k: e.cuda() for k, e in batch.items()}

        query_id = batch['query_id'].view(-1).tolist()

        output = model(encoder_input_ids=batch['encoder_input_ids'],
                       encoder_attention_mask=batch['encoder_attention_mask'],
                       decoder_input_ids=None, decoder_attention_mask=None, labels=batch['labels'],
                       target=None, num_beams=args.num_beams, num_return_sequences=args.num_return_sequences,
                       no_repeat_ngram_size=args.no_repeat_ngram_size, gen_max_label_len=args.gen_max_label_len)

        res_dict['query_id'].extend(query_id)
        res_dict['pred_list'].extend(output['pred'])
        res_dict['label_list'].extend(output['label'])

    # detokenize result
    detokenized_pred, detokenized_label = detokenizing_output(tokenizer=tokenizer, res_dict=res_dict)

    # construct result format
    res_set = get_evaluation_set(query_list=res_dict['query_id'],
                                 detokenized_pred=detokenized_pred, detokenized_label=detokenized_label,
                                 key_='query', value_1='pred', value_2='label')

    return res_set

# detokenizing result method
def detokenizing_output(tokenizer, res_dict):
    if 'label_list' in res_dict.keys():
        detokenized_pred = [tokenizer.decode(d, skip_special_tokens=True) for d in res_dict['pred_list']]
        detokenized_label = [tokenizer.decode(d, skip_special_tokens=True) for d in res_dict['label_list']]
    else:
        detokenized_pred = [tokenizer.decode(d, skip_special_tokens=True, clean_up_tokenization_spaces=False) for d in res_dict['pred_list']]
        detokenized_label = None

    return detokenized_pred, detokenized_label

# constructing result format
def get_evaluation_set(query_list=None, detokenized_pred=None, detokenized_label=None,
                       key_='query', value_1='response', value_2='label'):
    evaluation_set = []
    if detokenized_label is not None:
        for query, pred, label in zip(query_list, detokenized_pred, detokenized_label):
            normalized_answer_ = normalize_answer(label)
            normalized_answer_ = normalized_answer_.replace('<extra_id_0>', '').replace('extraid0', '').strip()
            normalized_answer_ = normalized_answer_ if len(normalized_answer_) else 'placeholder'

            normalized_pred_ = normalize_answer(pred)
            normalized_pred_ = normalized_pred_.replace('<extra_id_0>', '').replace('extraid0', '').strip()
            normalized_pred_ = normalized_pred_ if len(normalized_pred_) else 'placeholder'

            evaluation_set.append({key_: query, value_1: normalized_pred_.strip(), value_2: normalized_answer_.strip()})
    else:
        for query, pred in zip(query_list, detokenized_pred):
            evaluation_set.append({key_: query, value_1: pred.strip()})

    return evaluation_set

# scoring result
def measure_result(detokenized_pred, detokenized_label):
    meters = dict()

    meteor = evaluate.load('meteor')
    sacrebleu = evaluate.load("sacrebleu")
    rouge = evaluate.load('rouge')

    hypothesis_list = [
        x.replace('<extra_id_0>', '').replace('extraid0', '').strip() for x in detokenized_pred
    ]

    hypothesis_list = [x if len(x) > 10 else 'placeholder' for x in hypothesis_list]

    reference_list = [
        x.replace('<response>', '').strip() for x in detokenized_label
    ]

    instance_num = len(reference_list)

    # f1-score
    f1, em = matching_evaluate(reference_list, hypothesis_list)
    meters['f1'] = f1
    meters['em'] = em

    # bleu-score
    bleu_score = [
        sacrebleu.compute(predictions=[hypothesis], references=[[reference]], tokenize='none')['score']
        for hypothesis, reference in zip(hypothesis_list, reference_list)
    ]
    bleu_score = sum(bleu_score) / instance_num
    meters['bleu'] = bleu_score

    # Rouge-score
    rouge_score = rouge.compute(predictions=hypothesis_list, references=reference_list)
    meters['rouge'] = rouge_score['rougeL'] * 100

    meteor_score = meteor.compute(predictions=hypothesis_list, references=reference_list)
    meters['meteor'] = meteor_score['meteor'] * 100

    return meters

# gathering distributed result method
def gather_metric(res_write_set, data_path=None):
    word_size_ = dist.get_world_size()

    local_vectors_to_gather = [None for _ in range(word_size_)]
    dist.barrier()
    dist.all_gather_object(object_list=local_vectors_to_gather, obj=res_write_set)

    global_visited = set()
    global_res = []
    detokenized_pred, detokenized_label = [], []
    dist.barrier()
    meters = None
    if dist.get_rank() == 0:
        for local_vector_ in local_vectors_to_gather:
            for local_ in local_vector_:
                if local_['query'] not in global_visited:
                    global_res.append(local_)
                    global_visited.add(local_['query'])
                    detokenized_pred.append(local_['pred'])
                    detokenized_label.append(local_['label'])
        meters = measure_result(detokenized_pred, detokenized_label)

        with open(data_path, 'w') as f:
            for i in global_res: f.write(json.dumps(i, ensure_ascii=False) + '\n')

    dist.barrier()

    return meters