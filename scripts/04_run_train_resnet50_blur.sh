#!/bin/bash

set -e

PROJECT_DIR="/home/coletta/DJMA/classificationTomatoBlur"
DATA_DIR="/home/coletta/DJMA/tomato_final"
OUTPUT_DIR="/home/coletta/DJMA/classificationTomatoBlur/results_resnet50_blur_light"
MLRUNS_DIR="/home/coletta/DJMA/mlruns"
LOGS_DIR="/home/coletta/DJMA/classificationTomatoBlur/logs"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$MLRUNS_DIR"
mkdir -p "$LOGS_DIR"

LOG_FILE="$LOGS_DIR/train_resnet50_blur_light_$(date +%Y%m%d_%H%M%S).log"

cd "$PROJECT_DIR"

echo "Lancement entraînement blur léger..."
echo "Log file : $LOG_FILE"

CUDA_VISIBLE_DEVICES=0 python train_resnet50_tomato_blur.py \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size 32 \
  --num_workers 4 \
  --epochs_head 5 \
  --epochs_ft 15 \
  --lr_head 1e-3 \
  --lr_ft 1e-4 \
  --unfreeze layer4 \
  --gaussian_blur \
  --blur_p 0.15 \
  --blur_kernel 5 \
  --blur_sigma_min 0.1 \
  --blur_sigma_max 0.8 \
  --mlflow_experiment Tomato_ResNet50_Blur \
  --mlflow_run_name resnet50_blur_light_seed42 \
  --mlflow_tracking_uri file:///home/coletta/DJMA/mlruns \
  2>&1 | tee "$LOG_FILE"
