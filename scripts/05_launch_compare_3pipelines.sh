#!/bin/bash

set -e

ROOT_DIR="/home/coletta/DJMA"
MLRUNS_DIR="$ROOT_DIR/mlruns"

DATA_RAW="$ROOT_DIR/tomato_final"
DATA_SAM="$ROOT_DIR/tomato_segment_dataset"

BASELINE_DIR="$ROOT_DIR/classificationTomato"
BLUR_DIR="$ROOT_DIR/classificationTomatoBlur"
SAM_DIR="$ROOT_DIR/classificationSAM_CNN"

LOGS_DIR="$ROOT_DIR/logs_compare_3pipelines"
mkdir -p "$LOGS_DIR" "$MLRUNS_DIR"

SEEDS=(42 123 2026 1373 1979)

COMMON_ARGS="\
  --batch_size 32 \
  --num_workers 4 \
  --epochs_head 5 \
  --epochs_ft 15 \
  --lr_head 1e-3 \
  --lr_ft 1e-4 \
  --unfreeze layer4 \
  --mlflow_tracking_uri file://$MLRUNS_DIR"

echo "======================================================"
echo "Comparaison 5 pipelines sur seeds : ${SEEDS[*]}"
echo "======================================================"

for SEED in "${SEEDS[@]}"; do

  echo "======================================================"
  echo "SEED = $SEED"
  echo "======================================================"

  # ---------------------------------------------------------------------------
  # 1. CNN baseline
  # ---------------------------------------------------------------------------
  cd "$BASELINE_DIR"

  OUT_BASE="$BASELINE_DIR/results_seed_${SEED}"
  mkdir -p "$OUT_BASE"

  LOG_BASE="$LOGS_DIR/baseline_seed_${SEED}_$(date +%Y%m%d_%H%M%S).log"

  echo "[RUN] CNN baseline — seed $SEED"

  CUDA_VISIBLE_DEVICES=0 python train_resnet50_tomato.py \
    --data_dir "$DATA_RAW" \
    --output_dir "$OUT_BASE" \
    --seed "$SEED" \
    --mlflow_experiment Tomato_Compare_3Pipelines \
    --mlflow_run_name "baseline_seed_${SEED}" \
    $COMMON_ARGS \
    2>&1 | tee "$LOG_BASE"


  # ---------------------------------------------------------------------------
  # 2. CNN + blur léger
  # ---------------------------------------------------------------------------
  cd "$BLUR_DIR"

  OUT_BLUR="$BLUR_DIR/results_blur_light_seed_${SEED}"
  mkdir -p "$OUT_BLUR"

  LOG_BLUR="$LOGS_DIR/blur_light_seed_${SEED}_$(date +%Y%m%d_%H%M%S).log"

  echo "[RUN] CNN + blur léger — seed $SEED"

  CUDA_VISIBLE_DEVICES=0 python train_resnet50_tomato_blur.py \
    --data_dir "$DATA_RAW" \
    --output_dir "$OUT_BLUR" \
    --seed "$SEED" \
    --gaussian_blur \
    --blur_p 0.15 \
    --blur_kernel 5 \
    --blur_sigma_min 0.1 \
    --blur_sigma_max 0.8 \
    --mlflow_experiment Tomato_Compare_3Pipelines \
    --mlflow_run_name "blur_light_seed_${SEED}" \
    $COMMON_ARGS \
    2>&1 | tee "$LOG_BLUR"


  # ---------------------------------------------------------------------------
  # 3. SAM3 + CNN
  # ---------------------------------------------------------------------------
  cd "$SAM_DIR"

  OUT_SAM="$SAM_DIR/results_sam3_cnn_seed_${SEED}"
  mkdir -p "$OUT_SAM"

  LOG_SAM="$LOGS_DIR/sam3_cnn_seed_${SEED}_$(date +%Y%m%d_%H%M%S).log"

  echo "[RUN] SAM3 + CNN — seed $SEED"

  CUDA_VISIBLE_DEVICES=0 python train_resnet50_sam3.py \
    --data_dir "$DATA_SAM" \
    --output_dir "$OUT_SAM" \
    --seed "$SEED" \
    --mlflow_experiment Tomato_Compare_3Pipelines \
    --mlflow_run_name "sam3_cnn_seed_${SEED}" \
    $COMMON_ARGS \
    2>&1 | tee "$LOG_SAM"

done

echo "======================================================"
echo "Toutes les expériences sont terminées."
echo "Logs : $LOGS_DIR"
echo "MLflow : $MLRUNS_DIR"
echo "======================================================"
