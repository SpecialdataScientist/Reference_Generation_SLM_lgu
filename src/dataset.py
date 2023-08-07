import os
import sys

sys.path.append(os.path.abspath('../'))

import json

import torch
from torch.utils.data import Dataset

from src.preprocess import preprocess_dialogue, parsing_span

# Dataset for train and validation with gold passage
class ReferenceGeneratorDataset(Dataset):
    def __init__(self, data_type='train', id2passage=None, tokenizer=None, args=None):
        self.args = args
        self.tokenizer = tokenizer
        self.id2passage = id2passage

        # load dialogue data
        self.raw_dialogue_data = [json.loads(line) for line in
                                  open(os.path.join(f'{args.dataset_path}', f'{data_type}.jsonl')).readlines()]
        # preprocess dialogue data
        self.dialogue_data = preprocess_dialogue(self.raw_dialogue_data)

    def __len__(self):
        return len(self.dialogue_data)

    # construct prompt
    def get_encoder_input_set(self, dialogue_history, context_text, max_encoder_token):
        input_format = f'Context: {context_text} Dialogue: {dialogue_history}'
        tokenized_input_format = self.tokenizer.tokenize(input_format)

        split_point = 1
        while len(tokenized_input_format) > max_encoder_token:
            if split_point == len(tokenized_input_format):
                over_size = len(tokenized_input_format) - max_encoder_token

                input_format = f'Context: {context_text} Dialogue: {split_dialogue_history}'
                tokenized_input_format = self.tokenizer.tokenize(input_format)
                tokenized_input_format = tokenized_input_format[:-over_size]
                break

            split_dialogue_history = [d.strip() for d in dialogue_history.split('||')[split_point:]]
            split_dialogue_history = ' || '.join(split_dialogue_history).strip()
            input_format = f'Context: {context_text} Dialogue: {split_dialogue_history}'

            tokenized_input_format = self.tokenizer.tokenize(input_format)

            split_point += 1

        input_ids = self.tokenizer.convert_tokens_to_ids(tokenized_input_format)
        input_ids = input_ids + [self.tokenizer.eos_token_id]

        return input_ids

    # construct decoder input
    def get_decoder_input_set(self, response, max_token):
        if len(response) > max_token:
            response = response[:max_token]

        decoder_input_ids = [self.tokenizer.eos_token_id] + response
        target = response + [self.tokenizer.eos_token_id]

        return decoder_input_ids, target

    # preprocess example
    def __getitem__(self, item):
        dialogue_data_ = self.dialogue_data[item]

        query_id = dialogue_data_['query_id']
        gold_passage_id = dialogue_data_['gold_passage_id']

        gold_passage_data = self.id2passage[str(gold_passage_id)]
        gold_passage_text = gold_passage_data['passage']
        gold_span_text = parsing_span(span_range=dialogue_data_['reference'],
                                      ids_sp_set=gold_passage_data['reference_id_set'])

        dialogue_history = ' || '.join(dialogue_data_['dialogue_history']).strip()
        dialogue_history = dialogue_history.replace('<last_turn>', 'last_turn:').replace('<user>', 'user:').replace(
            '<agent>', 'agent:').strip()

        encoder_input_ids = self.get_encoder_input_set(dialogue_history=dialogue_history,
                                                       context_text=gold_passage_text,
                                                       max_encoder_token=(self.args.query_max_token +
                                                                          self.args.passage_max_token) - 1)

        label = self.tokenizer.tokenize(gold_span_text)
        label = self.tokenizer.convert_tokens_to_ids(label)

        decoder_input_ids, target = self.get_decoder_input_set(label, self.args.decoder_max_token - 1)

        if len(label) > self.args.max_label_len:
            label = label[:self.args.max_label_len]

        return {'query_id': [query_id],
                'encoder_input_ids': encoder_input_ids, 'decoder_input_ids': decoder_input_ids,
                'label': label, 'target': target}

    # get padded batch data
    def get_batch_setting(self, batch_data, pad_idx=0, max_token=512, top_type='top_1'):
        unified_batch_format = []

        if top_type == 'top_1':
            for example_ in batch_data:
                if len(example_) < max_token:
                    padding_ids = [pad_idx] * (max_token - len(example_))
                    unified_batch_format.append(example_ + padding_ids)
                else:
                    unified_batch_format.append(example_)
        else:
            for example_ in batch_data:
                candidate_unified_input_ids = []
                for ids_ in example_:
                    if len(ids_) < max_token:
                        padding_ids = [pad_idx] * (max_token - len(ids_))
                        candidate_unified_input_ids.append(ids_ + padding_ids)
                    else:
                        candidate_unified_input_ids.append(ids_)
                unified_batch_format.append(candidate_unified_input_ids)

        return unified_batch_format

    def collate_fn(self, batch):
        query_id = torch.tensor([d['query_id'] for d in batch], dtype=torch.long)
       
        encoder_input_ids = torch.tensor(self.get_batch_setting([d['encoder_input_ids'] for d in batch],
                                                                pad_idx=self.tokenizer.pad_token_id,
                                                                max_token=self.args.query_max_token + self.args.passage_max_token,
                                                                top_type='top_1'), dtype=torch.long)
        encoder_attention_mask = encoder_input_ids.ne(self.tokenizer.pad_token_id).long()

        decoder_input_ids = torch.tensor(self.get_batch_setting([d['decoder_input_ids'] for d in batch],
                                                                pad_idx=self.tokenizer.pad_token_id,
                                                                max_token=self.args.decoder_max_token,
                                                                top_type='top_1'), dtype=torch.long)

        target = torch.tensor(self.get_batch_setting([d['target'] for d in batch],
                                                     pad_idx=-100, max_token=self.args.decoder_max_token,
                                                     top_type='top_1'), dtype=torch.long)
        decoder_attention_mask = target.ne(-100).long()

        labels = torch.tensor(self.get_batch_setting([d['label'] for d in batch], pad_idx=self.tokenizer.pad_token_id,
                                                     max_token=self.args.max_label_len, top_type='top_1'),
                              dtype=torch.long)

        return {'query_id': query_id,
                'encoder_input_ids': encoder_input_ids,
                'encoder_attention_mask': encoder_attention_mask,
                'decoder_input_ids': decoder_input_ids, 'decoder_attention_mask': decoder_attention_mask,
                'labels': labels, 'target': target}

# Dataset for Evaluation with retrieved passage
class ReferenceGeneratorEvaluationDataset(Dataset):
    def __init__(self, data_type='dev', id2passage=None, tokenizer=None, args=None):
        self.args = args
        self.tokenizer = tokenizer
        self.id2passage = id2passage

        # load dialogue data
        self.raw_dialogue_data = [json.loads(line) for line in
                                  open(os.path.join(f'{args.dataset_path}', f'{data_type}.jsonl')).readlines()]
        # preprocess dialogue data
        self.dialogue_data = preprocess_dialogue(self.raw_dialogue_data)

        # load retrieved passage
        self.re_ranking_list = [json.loads(line) for line in open(os.path.join(f'{args.dataset_path}', f'{data_type}_reranking_res.jsonl')).readlines()]
        self.top_k_dict = {str(d['origin_idx']): d['cand_passages'] for d in self.re_ranking_list}

    def __len__(self):
        return len(self.dialogue_data)

    # construct prompt
    def get_encoder_input_set(self, dialogue_history, context_text, max_encoder_token):
        input_format = f'Context: {context_text} Dialogue: {dialogue_history}'
        tokenized_input_format = self.tokenizer.tokenize(input_format)

        split_point = 1
        while len(tokenized_input_format) > max_encoder_token:
            if split_point == len(tokenized_input_format):
                over_size = len(tokenized_input_format) - max_encoder_token

                input_format = f'Context: {context_text} Dialogue: {split_dialogue_history}'
                tokenized_input_format = self.tokenizer.tokenize(input_format)
                tokenized_input_format = tokenized_input_format[:-over_size]
                break

            split_dialogue_history = [d.strip() for d in dialogue_history.split('||')[split_point:]]
            split_dialogue_history = ' || '.join(split_dialogue_history).strip()
            input_format = f'Context: {context_text} Dialogue: {split_dialogue_history}'

            tokenized_input_format = self.tokenizer.tokenize(input_format)

            split_point += 1

        input_ids = self.tokenizer.convert_tokens_to_ids(tokenized_input_format)
        input_ids = input_ids + [self.tokenizer.eos_token_id]

        return input_ids

    # preprocess example
    def __getitem__(self, item):
        dialogue_data_ = self.dialogue_data[item]

        query_id = dialogue_data_['query_id']
        gold_passage_id = dialogue_data_['gold_passage_id']

        gold_passage_data = self.id2passage[str(gold_passage_id)]
        gold_span_text = parsing_span(span_range=dialogue_data_['reference'],
                                      ids_sp_set=gold_passage_data['reference_id_set'])

        dialogue_history = ' || '.join(dialogue_data_['dialogue_history']).strip()
        dialogue_history = dialogue_history.replace('<last_turn>', 'last_turn:').replace('<user>', 'user:').replace(
            '<agent>', 'agent:').strip()

        retrieved_passage_id_top_1 = self.top_k_dict[str(query_id)][0]
        retrieved_passage_data_top_1 = self.id2passage[str(retrieved_passage_id_top_1)]
        retrieved_passage_text_top_1 = retrieved_passage_data_top_1['passage']

        retrieved_encoder_input_ids_top_1 = self.get_encoder_input_set(dialogue_history=dialogue_history,
                                                       context_text=retrieved_passage_text_top_1,
                                                       max_encoder_token=(self.args.query_max_token +
                                                                          self.args.passage_max_token) - 1)

        label = self.tokenizer.tokenize(gold_span_text)
        label = self.tokenizer.convert_tokens_to_ids(label)

        if len(label) > self.args.max_label_len:
            label = label[:self.args.max_label_len]

        return {'query_id': [query_id], 'retrieved_encoder_input_ids_top_1': retrieved_encoder_input_ids_top_1,
                'label': label}

    # get padded batch data
    def get_batch_setting(self, batch_data, pad_idx=0, max_token=512, top_type='top_1'):
        unified_batch_format = []

        if top_type == 'top_1':
            for example_ in batch_data:
                if len(example_) < max_token:
                    padding_ids = [pad_idx] * (max_token - len(example_))
                    unified_batch_format.append(example_ + padding_ids)
                else:
                    unified_batch_format.append(example_)
        else:
            for example_ in batch_data:
                candidate_unified_input_ids = []
                for ids_ in example_:
                    if len(ids_) < max_token:
                        padding_ids = [pad_idx] * (max_token - len(ids_))
                        candidate_unified_input_ids.append(ids_ + padding_ids)
                    else:
                        candidate_unified_input_ids.append(ids_)
                unified_batch_format.append(candidate_unified_input_ids)

        return unified_batch_format

    def collate_fn(self, batch):
        query_id = torch.tensor([d['query_id'] for d in batch], dtype=torch.long)

        encoder_input_ids = torch.tensor(self.get_batch_setting([d['retrieved_encoder_input_ids_top_1'] for d in batch],
                                                                pad_idx=self.tokenizer.pad_token_id,
                                                                max_token=self.args.query_max_token + self.args.passage_max_token,
                                                                top_type='top_1'), dtype=torch.long)
        encoder_attention_mask = encoder_input_ids.ne(self.tokenizer.pad_token_id).long()

        labels = torch.tensor(self.get_batch_setting([d['label'] for d in batch], pad_idx=self.tokenizer.pad_token_id,
                                                     max_token=self.args.max_label_len, top_type='top_1'),
                              dtype=torch.long)

        return {'query_id': query_id, 'encoder_input_ids': encoder_input_ids, 'encoder_attention_mask': encoder_attention_mask,
                'labels': labels}








