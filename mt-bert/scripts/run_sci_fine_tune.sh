#!/bin/sh

if [ $# -ne 3 ]; then
  echo "fine_tune.sh <batch_size> <gpu> <task>"
  exit 1
fi

BATCH_SIZE=$1
gpu=$2
task=$3

echo "export CUDA_VISIBLE_DEVICES=${gpu}"
export CUDA_VISIBLE_DEVICES=${gpu}
tstr=$(date +"%FT%H%M")

MODEL_NAME="model_4"
ROOT="/data1/home/liuxiaofeng/code/bluebert-master"
BERT_PATH="$ROOT/checkpoints_mt/sci-mt-cgi+egi+local_chemprot,ddi,cdr,ppim_adam_5_2e-5_2020-10-21T1837/$MODEL_NAME.pt"
DATA_DIR="$ROOT/bio_data/canonical_data/bert_uncased_512_lower"
#DATA_DIR="$ROOT/bio_data/canonical_data/biobert_cased"


if [ "$task" = "cdr" ]; then
  answer_opt=0
  optim="adam"
  grad_clipping=0
  global_grad_clipping=1
  epochs=5
  lr="2e-5"
elif [ "$task" = "chemprot" ]; then
  answer_opt=0
  optim="adam"
  grad_clipping=0
  global_grad_clipping=1
  epochs=5
  lr="2e-5"
elif [ "$task" = "ddi" ]; then
  answer_opt=0
  optim="adam"
  grad_clipping=0
  global_grad_clipping=1
  epochs=5
  lr="2e-5"
elif [ "$task" = "ppim" ]; then
  answer_opt=0
  optim="adam"
  grad_clipping=0
  global_grad_clipping=1
  epochs=5
  lr="2e-5"
else
  echo "Cannot recognize $task"
  exit
fi

train_datasets=$task
test_datasets=$task

model_dir="$ROOT/checkpoints_mt/finetune_reuse/sci-mt-cgi+egi+local_chemprot,ddi,cdr,ppim_adam_5_2e-5_2020-10-21T1837/${task}_${epochs}_${lr}_${tstr}"
log_file="${model_dir}/log.log"
python mt_bluebert/finetune_train.py \
  --data_dir ${DATA_DIR} \
  --init_checkpoint ${BERT_PATH} \
  --task_def mt_bluebert/blue_task_def.yml \
  --batch_size "${BATCH_SIZE}" \
  --epochs ${epochs} \
  --output_dir "${model_dir}" \
  --log_file "${log_file}" \
  --answer_opt ${answer_opt} \
  --optimizer ${optim} \
  --train_datasets "${train_datasets}" \
  --test_datasets "${test_datasets}" \
  --grad_clipping ${grad_clipping} \
  --global_grad_clipping ${global_grad_clipping} \
  --learning_rate ${lr} \
  --max_seq_len 512 \
  --not_save \
  --reuse