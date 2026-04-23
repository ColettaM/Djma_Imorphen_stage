#!/bin/bash
#SBATCH --job-name=sam3_segment
#SBATCH -p gpu
#SBATCH --cluster=nautilus
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 0-04:00:00
#SBATCH --mail-user=cmoutsihatsala@univ-angers.fr
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/scratch/nautilus/users/cmoutsihatsala@univ-angers.fr/TOMATO/segmentation_sam3/logs/%x_%j.out
#SBATCH --error=/scratch/nautilus/users/cmoutsihatsala@univ-angers.fr/TOMATO/segmentation_sam3/logs/%x_%j.err

set -e

# ==============================================================================
# Environnement
# ==============================================================================
source ~/.bashrc
micromamba activate ClassificationTomatoGPU

WORKDIR=/scratch/nautilus/users/cmoutsihatsala@univ-angers.fr/TOMATO/segmentation_sam3
INPUT_DIR=/scratch/nautilus/users/cmoutsihatsala@univ-angers.fr/TOMATO/tomato_3616
RESULTS_DIR=$WORKDIR/results
MLRUNS_DIR=$WORKDIR/mlruns

cd "$WORKDIR"
mkdir -p logs "$RESULTS_DIR" "$MLRUNS_DIR"

# ==============================================================================
# Paramètres du run (passés via sbatch --export=...)
# ==============================================================================
RUN_NAME=${RUN_NAME:?RUN_NAME non défini}
PROMPT=${PROMPT:-leaf}
USE_CLAHE=${USE_CLAHE:-0}
USE_SW=${USE_SW:-0}
USE_FB=${USE_FB:-0}
USE_DP=${USE_DP:-0}
LIMIT=${LIMIT:-}
DEVICE=${DEVICE:-cuda}

OUTPUT_DIR="$RESULTS_DIR/$RUN_NAME"

# ==============================================================================
# Construction des flags optionnels
# ==============================================================================
EXTRA_ARGS=()

if [ "$USE_CLAHE" = "1" ]; then
    EXTRA_ARGS+=(--use_clahe)
fi

if [ "$USE_SW" = "1" ]; then
    EXTRA_ARGS+=(--use_sliding_window)
fi

if [ "$USE_FB" = "1" ]; then
    EXTRA_ARGS+=(--use_fallback)
fi

if [ "$USE_DP" = "1" ]; then
    EXTRA_ARGS+=(--use_dual_prompt)
fi

if [ -n "${LIMIT}" ]; then
    EXTRA_ARGS+=(--limit "$LIMIT")
fi

# ==============================================================================
# Diagnostics
# ==============================================================================
echo "========================================"
echo "JOB        : $RUN_NAME"
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
echo "Prompt     : $PROMPT"
echo "CLAHE      : $USE_CLAHE"
echo "Sliding    : $USE_SW"
echo "Fallback   : $USE_FB"
echo "DualPrompt : $USE_DP"
echo "Limit      : ${LIMIT:-none}"
echo "Images     : $(find "$INPUT_DIR" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.tif' -o -iname '*.tiff' -o -iname '*.webp' \) | wc -l)"
echo "========================================"

# ==============================================================================
# MLflow local
# ==============================================================================
export MLFLOW_TRACKING_URI="file://$MLRUNS_DIR"

# ==============================================================================
# Lancement
# ==============================================================================
python segment.py \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    --prompt "$PROMPT" \
    --device "$DEVICE" \
    "${EXTRA_ARGS[@]}"

echo "========================================"
echo "Run $RUN_NAME terminé."
echo "========================================"

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
"

# ==============================================================================
# Vérification des sorties
# ==============================================================================
echo "Sorties générées :"
echo "  masks    : $(find "$OUTPUT_DIR/masks"    -type f 2>/dev/null | wc -l)"
echo "  overlays : $(find "$OUTPUT_DIR/overlays" -type f 2>/dev/null | wc -l)"
echo "  crops    : $(find "$OUTPUT_DIR/crops"    -type f 2>/dev/null | wc -l)"
echo "  json     : $(find "$OUTPUT_DIR/json"     -type f 2>/dev/null | wc -l)"
echo "========================================"
