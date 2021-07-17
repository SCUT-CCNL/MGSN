#! /bin/sh

ROOT="/data1/home/liuxiaofeng/code/bluebert-master/bio_data/"
#BERT_PATH="/data1/home/liuxiaofeng/code/bluebert-master/model/biobert_cased/"
BERT_PATH="/data1/home/liuxiaofeng/code/bluebert-master/model/scibert_uncased/"
#BERT_PATH="/data1/home/liuxiaofeng/code/bluebert-master/model/pubmedBERT_uncased_fulltext/"

#datasets="ddi,cdr,ppim,chemprot"
datasets="ade"


python mt_bluebert/blue_prepro.py \
  --root_dir $ROOT \
  --task_def mt_bluebert/blue_task_def.yml \
  --datasets $datasets \
  --overwrite

python mt_bluebert/blue_prepro_std.py \
  --vocab ${BERT_PATH}/vocab.txt \
  --root_dir ${ROOT}/canonical_data \
  --task_def mt_bluebert/blue_task_def.yml \
  --datasets ${datasets} \
  --max_seq_len 150 \
  --overwrite \
  --do_lower_case \
