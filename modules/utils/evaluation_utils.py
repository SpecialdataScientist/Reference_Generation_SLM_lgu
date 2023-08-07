import os
import sys

sys.path.append(os.path.abspath('../'))

from collections import Counter

def evaluate_score(model_result, top_k_list):
    res = {}

    gold_passage_id = model_result['gold_passage_id']
    top_k_passage_id_dict = model_result['top_k_passage_id']
    for k_ in top_k_list:
        point_ = 0
        recall_ = 0
        total_top_k_passage = top_k_passage_id_dict[f'top_{k_}']
        for example_id, gold_passage_id_ in enumerate(gold_passage_id):
            top_k_passage = total_top_k_passage[example_id]

            if gold_passage_id_ not in top_k_passage:
                point_ += 0
            else:
                rank = (top_k_passage == gold_passage_id_).nonzero(as_tuple=True)[0][0] + 1
                point_ += 1 / rank

            if gold_passage_id_ in top_k_passage:
                recall_ += 1

        top_k_point = point_ / len(gold_passage_id)
        top_k_recall = recall_ / len(gold_passage_id)

        res[f'top_{k_}_mrr_score'] = top_k_point
        res[f'top_{k_}_recall'] = top_k_recall

    return res


def normalize_answer(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def matching_evaluate(references, predictions):
    f1 = em = total = 0
    for ref_text, prediction in zip(references, predictions):
        total += 1
        ground_truths = [ref_text]
        f1 += metric_max_over_ground_truths(f1_score, prediction,
                                            ground_truths)
        em += metric_max_over_ground_truths(exact_match_score, prediction,
                                            ground_truths)
    f1 = 100.0 * f1 / total
    em = 100.0 * em / total

    return f1, em
