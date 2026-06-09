#!/bin/bash

set -e

PROJECT_DIR="/home/coletta/DJMA/classificationSAM_CNN"
DATA_DIR="/home/coletta/DJMA/tomato_segment_dataset"
OUTPUT_DIR="/home/coletta/DJMA/classificationSAM_CNN/results_resnet50_sam3"
MLRUNS_DIR="/home/coletta/DJMA/mlruns"
LOGS_DIR="/home/coletta/DJMA/classificationSAM_CNN/logs"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$MLRUNS_DIR"
mkdir -p "$LOGS_DIR"

LOG_FILE="$LOGS_DIR/train_resnet50_sam3_$(date +%Y%m%d_%H%M%S).log"

cd "$PROJECT_DIR"

echo "Lancement entraînement SAM3 + CNN..."
echo "Log file : $LOG_FILE"

CUDA_VISIBLE_DEVICES=0 python train_resnet50_sam3.py \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size 32 \
  --num_workers 4 \
  --epochs_head 5 \
  --epochs_ft 15 \
  --lr_head 1e-3 \
  --lr_ft 1e-4 \
  --unfreeze layer4 \
  --mlflow_experiment Tomato_ResNet50_SAM3 \
  --mlflow_run_name resnet50_sam3_layer4_seed42 \
  --mlflow_tracking_uri file:///home/coletta/DJMA/mlruns \
  2>&1 | tee "$LOG_FILE"
