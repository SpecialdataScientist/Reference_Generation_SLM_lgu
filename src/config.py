import argparse

def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--parallel_gpu', default=True, type=bool)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--device_ids', default='10', type=str)
    parser.add_argument('--n_gpu', default=4, type=int)
    parser.add_argument('--multi_gpu', default=True, type=bool)
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--seed', default=1341, type=int)
    parser.add_argument('--use_gpu', default=4, type=int)
    parser.add_argument('--per_gpu_batch_size', default=8, type=int)
    parser.add_argument('--passage_batch_size', default=1024, type=int)
    parser.add_argument('--validation_batch_size', default=8, type=int)

    # path
    parser.add_argument('--dataset_path', default='../dataset/')
    parser.add_argument('--save_file_path', default='../save_file/', type=str)
    parser.add_argument('--check_point_path', default='../check_point/', type=str)
    parser.add_argument('--gen_reference_file_path', default='../gen_reference_file/', type=str)
    parser.add_argument('--transformers_generator_model_name', default='gogamza/kobart-base-v2', type=str)

    # train
    parser.add_argument('--query_max_token', default=200, type=int)
    parser.add_argument('--passage_max_token', default=312, type=int)
    parser.add_argument('--max_label_len', default=1024, type=int)
    parser.add_argument('--decoder_max_token', default=512, type=int)
    parser.add_argument('--gen_max_label_len', default=385, type=int)
    parser.add_argument('--gen_min_label_len', default=2, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--adam_beta1', default=0.9, type=float)
    parser.add_argument('--adam_beta2', default=0.99, type=float)
    parser.add_argument('--adam_epsilon', default=1e-06, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--warmup_proportion', default=0.06, type=float)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--grad_clipping', default=1.0, type=float)
    parser.add_argument('--task_name', default='reference_generator', type=str)
    parser.add_argument('--log_step_count_steps', default=1, type=int)
    parser.add_argument('--add_tokens', default='True', type=str)
    parser.add_argument('--best_epoch', default=-1, type=int)
    parser.add_argument('--validation_mode', default='F', type=str)
    parser.add_argument('--num_beams', default=1, type=int)
    parser.add_argument('--num_return_sequences', default=3, type=int)
    parser.add_argument('--no_repeat_ngram_size', default=3, type=int)
    parser.add_argument('--evaluation_data_type', default='test', type=str)
    parser.add_argument('--using_top_k', default=1, type=int)

    args = parser.parse_args()

    return args


