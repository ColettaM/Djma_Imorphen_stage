#!/bin/bash

set -e

PROJECT_DIR="/home/coletta/DJMA/classificationTomato"
DATA_DIR="/home/coletta/DJMA/tomato_final"
OUTPUT_DIR="/home/coletta/DJMA/classificationTomato/results_resnet50_exp1"
MLRUNS_DIR="/home/coletta/DJMA/classificationTomato/mlruns"
LOGS_DIR="/home/coletta/DJMA/classificationTomato/logs"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$MLRUNS_DIR"
mkdir -p "$LOGS_DIR"

LOG_FILE="$LOGS_DIR/train_resnet50_exp1_$(date +%Y%m%d_%H%M%S).log"

cd "$PROJECT_DIR"

echo "Lancement entraînement..."
echo "Log file : $LOG_FILE"

CUDA_VISIBLE_DEVICES=0 python train_resnet50_tomato.py \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size 32 \
  --num_workers 4 \
  --epochs_head 5 \
  --epochs_ft 15 \
  --lr_head 1e-3 \
  --lr_ft 1e-4 \
  --unfreeze layer4 \
  --mlflow_experiment Tomato_ResNet50_Baseline \
  --mlflow_run_name resnet50_layer4_seed42 \
  --mlflow_tracking_uri file:///home/coletta/DJMA/mlruns \
  2>&1 | tee "$LOG_FILE"
