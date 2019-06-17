#!/usr/bin/env bash
export BERT_BASE_DIR=model/multi_cased_L-12_H-768_A-12
export QE_DIR=bert_tmp_data
export OUTPUT_DIR=bert_tmp_out
export CUDA_VISIBLE_DEVICES=
python run_regression.py \
  --task_name=qe-sent \
  --do_train=False \
  --do_eval=False \
  --do_predict=True \
  --data_dir=$QE_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=12 \
  --learning_rate=1e-6 \
  --eval_steps=1000 \
  --save_checkpoints_steps 1000 \
  --num_train_epochs=500 \
  --loss_type=xent \
  --max_seq_length=256 \
  --do_lower_case=False \
  --output_dir=$OUTPUT_DIR
