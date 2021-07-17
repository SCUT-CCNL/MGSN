#!/bin/sh

if [ $# -ne 2 ]; then
  echo "train.sh <batch_size> <gpu>"
  exit 1
fi

prefix="sci-cgi+egi+eli"
#prefix="sci-test"
BATCH_SIZE=$1
gpu=$2
echo "export CUDA_VISIBLE_DEVICES=${gpu}"
export CUDA_VISIBLE_DEVICES=${gpu}
tstr=$(date +"%FT%H%M")

#train_datasets="chemprot,ddi,cdr-document"
#test_datasets="chemprot,ddi,cdr-document"
#
train_datasets="ade"
test_datasets="ade"



ROOT="/data1/home/liuxiaofeng/code/bluebert-master/"

#BERT_PATH="$ROOT/model/biobert_cased/biobert.pt"
#DATA_DIR="$ROOT/bio_data/canonical_data/biobert_cased"

BERT_PATH="$ROOT/model/scibert_uncased/scibert.pt"
DATA_DIR="$ROOT/bio_data/canonical_data/bert_uncased_150_lower"
#
#BERT_PATH="$ROOT/model/pubmedBERT_uncased_fulltext/pubmedbert.pt"
#DATA_DIR="$ROOT/bio_data/canonical_data/pubmedbert_uncased_512_lower"


answer_opt=0
optim="adam"
grad_clipping=1
global_grad_clipping=1
lr="2e-5"
epochs=5


#model_dir="$ROOT/checkpoints_bio/${prefix}_${train_datasets}_${optim}_${epochs}_${lr}_${tstr}"
#model_dir="$ROOT/checkpoints_pubmed/${prefix}_${train_datasets}_${optim}_${epochs}_${lr}_${tstr}"
model_dir="$ROOT/checkpoints/${prefix}_${train_datasets}_${optim}_${epochs}_${lr}_${tstr}"
#model_dir="$ROOT/checkpoints_mt/${prefix}_${train_datasets}_${optim}_${epochs}_${lr}_${tstr}"
#model_dir="$ROOT/checkpoints_engineering/${prefix}_${train_datasets}_${optim}_${epochs}_${lr}_${tstr}"
log_file="${model_dir}/log.log"
python mt_bluebert/blue_train.py \
  --data_dir ${DATA_DIR} \
  --init_checkpoint ${BERT_PATH} \
  --task_def mt_bluebert/blue_task_def.yml \
  --batch_size "${BATCH_SIZE}" \
  --output_dir "${model_dir}" \
  --log_file "${log_file}" \
  --answer_opt ${answer_opt} \
  --optimizer ${optim} \
  --train_datasets ${train_datasets} \
  --test_datasets ${test_datasets} \
  --grad_clipping ${grad_clipping} \
  --global_grad_clipping ${global_grad_clipping} \
  --learning_rate ${lr} \
  --multi_gpu_on \
  --epochs ${epochs} \
  --max_seq_len 150