#!/usr/bin/env bash
export BERT_BASE_DIR=multi_cased_L-12_H-768_A-12
export QE_DIR=WMT17/sentence_level/en_de
export OUTPUT_DIR=QE_RUN/xslr_xent
export CUDA_VISIBLE_DEVICES=7
python run_regression.py \
  --task_name=qe-sent \
  --do_train=True \
  --do_eval=True \
  --do_predict=True \
  --data_dir=$QE_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$OUTPUT_DIR/last/model.ckpt-120000 \
  --train_batch_size=12 \
  --learning_rate=1e-6 \
  --eval_steps=1000 \
  --save_checkpoints_steps 1000 \
  --num_train_epochs=500 \
  --loss_type=xent \
  --max_seq_length=256 \
  --do_lower_case=False \
  --output_dir=$OUTPUT_DIR
