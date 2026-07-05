#!/usr/bin/env bash
set -e

cd /media/4tb/jiarui/dreamNav/step2_baseline

CUDA_VISIBLE_DEVICES=0 python train_rgb_condition_predictor.py \
  --root_dir ../pairUAV/ \
  --dataset_type train \
  --output_dir ./outputs_rgb_condition_predictor_R1_full \
  --model resnet18 \
  --image_size 128 \
  --range_scale 100.0 \
  --epochs 5 \
  --batch_size 256 \
  --num_workers 16 \
  --lr 1e-4 \
  --val_ratio 0.02 \
  --max_train_samples -1 \
  --max_val_samples 2 \
  --log_every 100 \
  --device cuda \
  --amp