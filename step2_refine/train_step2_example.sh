#!/usr/bin/env bash
set -e

cd /media/4tb/jiarui/dreamNav/step2_baseline

OUT=./outputs_baseline_30k_R6_scratch_lr5e-4_warmup120k_rgbLoss001_ddim50_256
mkdir -p "$OUT"

DISABLE_EPOCH_CKPT=0 SAVE_EVERY_N_EPOCHS=1 CUDA_VISIBLE_DEVICES=0 python train_rgb_loss.py \
  --config_path ./cldm_v15_pose_hybrid.yaml \
  --checkpoint_path ../models/controlnet/control_sd15_ini.ckpt \
  --root_dir ../pairUAV/ \
  --dataset_type train \
  --step1_json_path ../step1/step1_fixed64_truepose.json \
  --test_save_dir "$OUT/test_results/" \
  --output_dir "$OUT/" \
  --probe_save_dir "$OUT/probes/" \
  --train_mode lora_control_decoder_hint \
  --lora_rank 8 \
  --lora_alpha 8 \
  --lora_dropout 0.0 \
  --batch_size 128 \
  --test_batch_size 4 \
  --num_workers 16 \
  --gpus 1 \
  --precision 32 \
  --logger_freq 500 \
  --learning_rate 5e-4 \
  --warmup_steps 8000 \
  --warmup_min_lr_scale 0.02 \
  --rgb_loss_checkpoint ./outputs_rgb_condition_predictor_R1_full/best.pt \
  --rgb_loss_weight 0.001 \
  --rgb_loss_start_step 8000 \
  --rgb_loss_every_n_steps 4 \
  --rgb_loss_batch_size 8 \
  --rgb_loss_model resnet18 \
  --rgb_loss_image_size 128 \
  --rgb_loss_range_scale 100.0 \
  --rgb_loss_angle_weight 1.0 \
  --rgb_loss_range_weight 1.0 \
  --rgb_loss_norm_weight 0.01 \
  --rgb_loss_t_min 0 \
  --rgb_loss_t_max 500 \
  --rgb_loss_print_every 500 \
  --rgb_loss_plot_every 1000 \
  --rgb_loss_plot_dir "$OUT/rgb_loss_curves" \
  --heading_offset 10.0 \
  --range_offset 1.5 \
  --test_ddim_steps 50 \
  --max_test_samples 64 \
  --num_probe_samples 16 \
  --probe_ddim_steps 50 \
  --cfg_scale 1.0 \
  --image_size 256 \
  --max_epochs 999 \
  --max_steps 150000 \
  --fixed_true_json_path ../step1/step1_fixed64_truepose.json \
  --fixed_pred_json_path ../step1/step1_fixed64_predictpose.json \
  --fixed_visual_save_dir "$OUT/fixed_visuals" \
  --fixed_visual_every_n_steps 1000 \
  --fixed_visual_max_samples 64 \
  --fixed_visual_batch_size 4 \
  --fixed_visual_ddim_steps 50 \
  --fixed_visual_num_workers 2
