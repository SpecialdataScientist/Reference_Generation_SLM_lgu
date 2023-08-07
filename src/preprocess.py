import os
import sys
sys.path.append(os.path.abspath('../'))

import re

# preprocess dialogue format
def preprocess_dialogue(raw_data):
    preprocessing_data = []
    for example_ in raw_data:
        query_id = example_['query_id']
        dialogue_history = convert_query_string_to_list(example_['query'])
        gold_passage_id = example_['gold_passage_id']
        reference = example_['span_ids']
        response = example_['response']

        preprocessing_data.append({'query_id': query_id, 'dialogue_history': dialogue_history,
                                   'gold_passage_id': gold_passage_id, 'reference': reference,
                                   'response': response})
    return preprocessing_data

# preprocess passage format
def preprocess_passage(raw_data):
    preprocessing_data = dict()
    sorted_raw_data = sorted(raw_data.items())

    for example_ in sorted_raw_data:
        passage_id = example_[0]
        content = example_[-1]

        all_title, sub_title, original_passage_text = parsing_passage_content(content)

        preprocessing_data[str(passage_id)] = {'title': all_title, 'sub_title': sub_title,
                                               'passage': original_passage_text, 'reference_id_set': content['span_id_set']}
    return preprocessing_data

# string query -> list query
def convert_query_string_to_list(query_):
    split_query_ = re.split(r'(<(?:last_turn|agent|user)>)', query_)[1:]

    split_query = []
    for i in range(0, len(split_query_), 2):
        split_query.append(split_query_[i] + split_query_[i+1])

    dialogue_history = [' '.join(d.strip().split()) for d in split_query]

    return dialogue_history

# span to passage
def convert_span_to_passage(span_id_set):
    text_list = [v for k, v in span_id_set.items()]
    passage = ' '.join(text_list)

    return ' '.join(passage.strip().split()).strip()

# parsing passage information (example)
def parsing_passage_content(example_):
    # parsing title
    all_title = example_['doc_title']
    sub_title = all_title.split('|')

    # parsing passage
    original_passage_text = convert_span_to_passage(example_['span_id_set'])

    return all_title, sub_title, original_passage_text

# all span to passage
def parsing_span(span_range, ids_sp_set):
    span_list = []
    for i in range(span_range[0], span_range[-1] + 1):
        span = ids_sp_set[str(i)]
        span_list.append(span)

    return ' '.join(span_list)