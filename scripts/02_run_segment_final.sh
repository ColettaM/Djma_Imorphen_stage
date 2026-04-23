#!/bin/bash

set -e

# ==============================================================================
# Environnement
# ==============================================================================
source ~/miniconda3/etc/profile.d/conda.sh
conda activate seg

WORKDIR=/home/coletta/DJMA/segmentation_sam3
INPUT_DIR=/home/coletta/DJMA/tomato_final
RESULTS_DIR=$WORKDIR/results
MLRUNS_DIR=/home/coletta/DJMA/mlruns
LOGS_DIR=$WORKDIR/logs

cd "$WORKDIR"
mkdir -p "$LOGS_DIR" "$RESULTS_DIR" "$MLRUNS_DIR"

# ==============================================================================
# Paramètres du run
# ==============================================================================
RUN_NAME=${RUN_NAME:-final_total}
LIMIT=${LIMIT:-}
DEVICE=${DEVICE:-cuda}

OUTPUT_DIR="$RESULTS_DIR/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"

LOG_FILE="$LOGS_DIR/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

# ==============================================================================
# Arguments optionnels
# ==============================================================================
EXTRA_ARGS=()

if [ -n "${LIMIT}" ]; then
    EXTRA_ARGS+=(--limit "$LIMIT")
fi

# ==============================================================================
# Diagnostics
# ==============================================================================
{
echo "========================================"
echo "RUN        : $RUN_NAME"
echo "HOSTNAME   : $(hostname)"
echo "PWD        : $(pwd)"
echo "Python     : $(which python)"
python --version
echo "========================================"
nvidia-smi
echo "========================================"
python -c "
import torch, mlflow
print('CUDA available :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU name       :', torch.cuda.get_device_name(0))
    print('VRAM total     :', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
print('Torch version  :', torch.__version__)
print('MLflow version :', mlflow.__version__)
"
echo "========================================"

echo "Input      : $INPUT_DIR"
echo "Output     : $OUTPUT_DIR"
echo "MLruns     : $MLRUNS_DIR"
echo "Run name   : $RUN_NAME"
echo "Config     : prompt=leaves | sliding_window=ON | fallback=diseased leaves -> plant"
echo "Limit      : ${LIMIT:-none}"
echo "Images     : $(find "$INPUT_DIR" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.tif' -o -iname '*.tiff' -o -iname '*.webp' \) | wc -l)"
echo "========================================"
} | tee "$LOG_FILE"

# ==============================================================================
# MLflow local
# ==============================================================================
export MLFLOW_TRACKING_URI="file://$MLRUNS_DIR"

# ==============================================================================
# Lancement
# ==============================================================================
CUDA_VISIBLE_DEVICES=0 python segment_final.py \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    --device "$DEVICE" \
    "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"

echo "========================================" | tee -a "$LOG_FILE"
echo "Run $RUN_NAME terminé." | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# ==============================================================================
# Résumé post-run
# ==============================================================================
python -c "
import json
from pathlib import Path

report = Path('$OUTPUT_DIR/pipeline_report.json')
if not report.exists():
    print('Rapport introuvable.')
    raise SystemExit(1)

with report.open() as f:
    r = json.load(f)

g = r['metrics']['global']
print(f'ok             : {g[\"n_ok\"]} ({g[\"pct_ok\"]}%)')
print(f'empty_mask     : {g[\"n_empty_mask\"]} ({g[\"pct_empty_mask\"]}%)')
print(f'fallback       : {g[\"n_fallback\"]} ({g[\"pct_fallback\"]}%)')
print(f'error          : {g[\"n_error\"]}')
print(f'area_mean      : {g[\"area_ratio_mean\"]} ± {g[\"area_ratio_std\"]}')
print(f'mask_small     : {g.get(\"pct_mask_small\", 0)}%')
print(f'mask_large     : {g.get(\"pct_mask_large\", 0)}%')
print(f'low_score      : {g.get(\"pct_low_score\", 0)}%')
print(f'score_mean     : {g[\"score_mean\"]} ± {g[\"score_std\"]}')
print(f't/image        : {g[\"time_per_image_s\"]}s')
print(f'total          : {g[\"time_total_s\"]}s')
" | tee -a "$LOG_FILE"

# ==============================================================================
# Vérification des sorties
# ==============================================================================
echo "Sorties générées :" | tee -a "$LOG_FILE"
echo "  masks    : $(find "$OUTPUT_DIR/masks"    -type f 2>/dev/null | wc -l)" | tee -a "$LOG_FILE"
echo "  overlays : $(find "$OUTPUT_DIR/overlays" -type f 2>/dev/null | wc -l)" | tee -a "$LOG_FILE"
echo "  crops    : $(find "$OUTPUT_DIR/crops"    -type f 2>/dev/null | wc -l)" | tee -a "$LOG_FILE"
echo "  json     : $(find "$OUTPUT_DIR/json"     -type f 2>/dev/null | wc -l)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
