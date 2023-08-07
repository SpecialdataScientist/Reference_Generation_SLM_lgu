# Reference_Generation_SLM_lgu
: 생성된 reference와 검색된 passage (Wiki + LGU Data 검색결과)를 이용해 SLM (BART-base) 기반 CoT (Chain-of-thought)를 통해 응답 생성 Task 수행 

## 1. Training 
    train_file.sh 실행 
    - seed: 1341 seed 값 고정 
    - NUM_TRAIN_EPOCHS: 20 epochs로 학습 수행
    - LEARNING_RATE: 낮은 leraning rate 설정에서 best 성능 달성
    - PER_GPU_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS: GPU당 2 batch를 사용하였고, gradient_accumulation_step을 2로 설정하여 총 4 (batch * accumulation step) * 4 (사용 GPU 수) = 16 batch로 학습 수행
    - VALIDATION_MODEL: gold passage를 이용해 dev set에 대한 validation을 수행할지 여부의 setting (String 'T' or 'F')
***

## 2. Evaluation
    evaluate_file.sh 실행
    - NUM_RETURN_SEQUENCE, NUM_BEAMS: 1로 설정하여 greedy search를 사용해 생성

***
