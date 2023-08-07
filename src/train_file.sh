#!/bin/bash
export NODE_RANK=0
export N_NODES=1
export N_GPU_NODE=4
export WORLD_SIZE=4
export MASTER_PORT=1341
export MASTER_ADDR='localhost'

GLOBAL_DEVICES_IDS=0,1,2,3
DEVICE_IDS=0,1,2,3

SEED=1341

NUM_TRAIN_EPOCHS=20
PER_GPU_BATCH_SIZE=2
VALIDATION_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=2
TASK_NAME='reference_generator'
LEARNING_RATE=1e-5

PASSAGE_MAX_TOKEN=384
QUERY_MAX_TOKEN=128
DECODER_MAX_TOKEN=512
MAX_LABEL_LEN=1024
GEN_MAX_LABEL_LEN=384
GEN_MIN_LABEL_LEN=2

NUM_RETURN_SEQUENCE=1
NUM_BEAMS=1

# train (gold passage)
VALIDATION_MODE='F'

BATCH_SIZE=`expr ${PER_GPU_BATCH_SIZE} \* ${GRADIENT_ACCUMULATION_STEPS} \* ${N_GPU_NODE}`

CUDA_VISIBLE_DEVICES=${GLOBAL_DEVICES_IDS} \
python -m torch.distributed.launch \
     --nproc_per_node ${N_GPU_NODE} \
     --nnodes ${N_NODES} \
     --node_rank ${NODE_RANK} \
     --master_addr ${MASTER_ADDR} \
     --master_port ${MASTER_PORT} \
     train.py --n_gpu ${WORLD_SIZE} \
              --device_ids ${DEVICE_IDS} \
              --epochs ${NUM_TRAIN_EPOCHS} \
              --per_gpu_batch_size ${PER_GPU_BATCH_SIZE} \
              --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
              --task_name ${TASK_NAME} \
              --lr ${LEARNING_RATE} \
              --passage_max_token ${PASSAGE_MAX_TOKEN} \
              --query_max_token ${QUERY_MAX_TOKEN} \
              --max_label_len ${MAX_LABEL_LEN} \
              --decoder_max_token ${DECODER_MAX_TOKEN} \
              --gen_max_label_len ${GEN_MAX_LABEL_LEN} \
              --gen_min_label_len ${GEN_MIN_LABEL_LEN} \
              --seed ${SEED} \
              --validation_mode ${VALIDATION_MODE} \
              --num_return_sequences ${NUM_RETURN_SEQUENCE} \
              --num_beams ${NUM_BEAMS};

# validation (gold passage)
VALIDATION_MODE='T'

BATCH_SIZE=`expr ${PER_GPU_BATCH_SIZE} \* ${GRADIENT_ACCUMULATION_STEPS} \* ${N_GPU_NODE}`

CUDA_VISIBLE_DEVICES=${GLOBAL_DEVICES_IDS} \
python -m torch.distributed.launch \
     --nproc_per_node ${N_GPU_NODE} \
     --nnodes ${N_NODES} \
     --node_rank ${NODE_RANK} \
     --master_addr ${MASTER_ADDR} \
     --master_port ${MASTER_PORT} \
     train.py --n_gpu ${WORLD_SIZE} \
              --device_ids ${DEVICE_IDS} \
              --epochs ${NUM_TRAIN_EPOCHS} \
              --per_gpu_batch_size ${PER_GPU_BATCH_SIZE} \
              --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
              --task_name ${TASK_NAME} \
              --lr ${LEARNING_RATE} \
              --passage_max_token ${PASSAGE_MAX_TOKEN} \
              --query_max_token ${QUERY_MAX_TOKEN} \
              --max_label_len ${MAX_LABEL_LEN} \
              --decoder_max_token ${DECODER_MAX_TOKEN} \
              --gen_max_label_len ${GEN_MAX_LABEL_LEN} \
              --gen_min_label_len ${GEN_MIN_LABEL_LEN} \
              --seed ${SEED} \
              --validation_mode ${VALIDATION_MODE} \
              --num_return_sequences ${NUM_RETURN_SEQUENCE} \
              --num_beams ${NUM_BEAMS};