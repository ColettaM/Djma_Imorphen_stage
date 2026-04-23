"""
segment_experience.py — Pipeline de segmentation SAM3 pour étude d'ablation contrôlée
============================================================================

Objectif
--------
Comparer proprement l'apport de plusieurs modifications du pipeline de
segmentation avant passage à la classification.

Principe méthodologique
-----------------------
Toutes les expériences doivent être lancées :
- sur le même dataset
- avec les mêmes constantes
- sur les mêmes images
- avec une seule modification à la fois

Expériences recommandées
------------------------
Run 0 : baseline
    prompt="leaf"
    sans CLAHE, sans sliding window, sans fallback, sans dual prompt

Run 1 : prompt only
    prompt="leaves"

Run 2 : sliding only
    prompt="leaves" + sliding window

Run 3 : CLAHE only
    prompt="leaves" + CLAHE

Run 4 : sliding + CLAHE
    prompt="leaves" + sliding window + CLAHE

Run 5 : + fallback
    prompt="leaves" + sliding window + CLAHE + fallback

Run 6 : + dual prompt
    prompt="leaves" + sliding window + CLAHE + fallback + dual prompt

Exemples
--------
python segment.py --input /path/to/tomato_3616 --output results/run_0_baseline      --run_name run_0_baseline      --prompt leaf
python segment.py --input /path/to/tomato_3616 --output results/run_1_prompt_leaves --run_name run_1_prompt_leaves --prompt leaves
python segment.py --input /path/to/tomato_3616 --output results/run_2_sliding_only  --run_name run_2_sliding_only  --prompt leaves --use_sliding_window
python segment.py --input /path/to/tomato_3616 --output results/run_3_clahe_only    --run_name run_3_clahe_only    --prompt leaves --use_clahe
python segment.py --input /path/to/tomato_3616 --output results/run_4_sliding_clahe --run_name run_4_sliding_clahe --prompt leaves --use_sliding_window --use_clahe
python segment.py --input /path/to/tomato_3616 --output results/run_5_fallback      --run_name run_5_fallback      --prompt leaves --use_sliding_window --use_clahe --use_fallback
python segment.py --input /path/to/tomato_3616 --output results/run_6_dual_prompt   --run_name run_6_dual_prompt   --prompt leaves --use_sliding_window --use_clahe --use_fallback --use_dual_prompt
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import mlflow
import numpy as np
import torch
from PIL import Image, ImageFilter

warnings.filterwarnings("ignore", message=".*flash attention.*")
warnings.filterwarnings("ignore", message=".*scaled_dot_product_attention.*")

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ── Constantes fixes de l'étude ────────────────────────────────────────────────
# Ces valeurs ne changent PAS d'un run à l'autre.
# Seuls les flags CLI (use_clahe, use_sliding_window…) varient.

FALLBACK_PROMPTS = ["diseased leaves", "plant", "vegetation"]
FALLBACK_CONFIDENCE = 0.2
CONFIDENCE_DEFAULT = 0.5

MASK_MIN_PX = 250
AREA_RATIO_MAX = 0.85
UNION_MIN_GAIN = 1.05  # gain > 5% pour accepter la fusion dual prompt

WINDOW_SIZE = 1024
STRIDE = 512

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

SAMPLE_CLASSES = [
    "tomato_late_blight",
    "tomato_mosaic_virus",
    "tomato_verticulium_wilt",
    "tomato_early_blight",
    "tomato_leaf_blight",
]
SAMPLES_PER_CLASS = 2

PALETTE = [(255, 100, 100), (100, 200, 100), (100, 150, 255), (255, 200, 50)]


# ── Utilitaires ────────────────────────────────────────────────────────────────


class nullcontext:
    """Contexte vide pour uniformiser le code CPU / GPU."""

    def __enter__(self):
        return None

    def __exit__(self, *_):
        return False


def get_device(requested: str = "auto") -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            log.info(f"GPU détecté : {torch.cuda.get_device_name(0)}")
            return "cuda"
        log.warning("Aucun GPU CUDA — bascule sur CPU.")
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA demandé mais indisponible — bascule sur CPU.")
        return "cpu"
    return requested


def load_sam3(device: str):
    try:
        import sam3 as sam3_pkg
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        bpe_path = (
            Path(sam3_pkg.__file__).parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        )
        if not bpe_path.exists():
            raise FileNotFoundError(f"BPE introuvable : {bpe_path}")

        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        log.info("Chargement SAM3...")
        model = build_sam3_image_model(
            bpe_path=str(bpe_path), device=device, eval_mode=True, load_from_HF=True
        )
        log.info("SAM3 chargé.")
        return model, Sam3Processor

    except ImportError as e:
        raise ImportError(f"Impossible d'importer sam3 : {e}")


# ── Prétraitement CLAHE ────────────────────────────────────────────────────────


def apply_clahe(image_pil: Image.Image) -> Image.Image:
    """
    CLAHE sur le canal L (espace LAB).
    Homogénéise l'éclairage sans modifier les teintes.
    Aide SAM3 sur les images terrain peu contrastées.
    """
    bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    lightness, a_channel, b_channel = cv2.split(lab)
    lightness_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
    lab_eq = cv2.merge((lightness_eq, a_channel, b_channel))

    result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


# ── Nettoyage morphologique ────────────────────────────────────────────────────


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Supprime les composantes connexes < MASK_MIN_PX pixels."""
    if not mask.any():
        return mask
    uint8 = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(uint8, connectivity=8)
    cleaned = np.zeros_like(uint8)
    for lbl in range(1, n):
        if int(stats[lbl, cv2.CC_STAT_AREA]) >= MASK_MIN_PX:
            cleaned[labels == lbl] = 1
    return cleaned.astype(bool)


# ── Sélection du meilleur masque ───────────────────────────────────────────────


def select_best_mask(masks: np.ndarray, scores: np.ndarray, H: int, W: int) -> tuple:
    """
    Fusionne les masques valides (MASK_MIN_PX ≤ aire ≤ AREA_RATIO_MAX).
    Retour : (mask_bool, score). Retourne (zeros, 0.0) si tout est vide.
    """
    total_px = H * W
    masks_bool = [(masks[i] > 0) for i in range(len(scores))]
    areas = np.array([m.sum() for m in masks_bool], dtype=np.float32)
    ratios = areas / total_px
    valid_idx = np.where((areas >= MASK_MIN_PX) & (ratios <= AREA_RATIO_MAX))[0]

    if len(valid_idx) == 0:
        idx = int(np.argmax(areas))
        best = masks_bool[idx]
        if not best.any():
            return np.zeros((H, W), dtype=bool), 0.0
        return best, float(scores[idx])

    merged = np.zeros((H, W), dtype=bool)
    for idx in valid_idx:
        merged |= masks_bool[idx]

    if float(merged.sum() / total_px) > AREA_RATIO_MAX:
        fb = valid_idx[int(np.argmax(areas[valid_idx]))]
        return masks_bool[fb], float(scores[fb])

    merged = clean_mask(merged)
    best_score = float(scores[valid_idx[int(np.argmax(scores[valid_idx]))]])
    return merged, best_score


# ── Segmentation d'un patch ────────────────────────────────────────────────────


def segment_patch(
    model, ProcessorClass, image_pil, prompt: str, confidence: float
) -> tuple:
    """Segmente une image entière ou un patch avec un prompt textuel."""
    H, W = np.array(image_pil).shape[:2]
    processor = ProcessorClass(model, confidence_threshold=confidence)
    state = processor.set_image(image_pil)
    processor.reset_all_prompts(state)
    state = processor.set_text_prompt(state=state, prompt=prompt)

    masks = state["masks"]
    scores = state["scores"]

    if isinstance(scores, torch.Tensor):
        scores = scores.detach().to(torch.float32).cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().to(torch.float32).cpu().numpy()
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    if masks.size == 0 or scores.size == 0:
        return np.zeros((H, W), dtype=bool), 0.0

    return select_best_mask(masks, scores, H, W)


# ── Sliding window ─────────────────────────────────────────────────────────────


def segment_sliding_window(
    model, ProcessorClass, image_pil, prompt: str, confidence: float
) -> tuple:
    """
    Découpe en patches WINDOW_SIZE×WINDOW_SIZE (overlap via STRIDE).
    Fusionne par vote pondéré (seuil 10%).
    Délègue à segment_patch si image ≤ WINDOW_SIZE dans les deux dimensions.
    """
    W, H = image_pil.size
    if W <= WINDOW_SIZE and H <= WINDOW_SIZE:
        return segment_patch(model, ProcessorClass, image_pil, prompt, confidence)

    def compute_starts(length: int):
        if length <= WINDOW_SIZE:
            return [0]
        starts = list(range(0, length - WINDOW_SIZE, STRIDE))
        last = length - WINDOW_SIZE
        if not starts or starts[-1] < last:
            starts.append(last)
        return starts

    xs = compute_starts(W)
    ys = compute_starts(H)
    log.info(
        f"[sliding] {len(xs)}x{len(ys)} = {len(xs) * len(ys)} patches ({WINDOW_SIZE}px, stride={STRIDE})"
    )

    accum = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    for y0 in ys:
        for x0 in xs:
            x1, y1 = min(x0 + WINDOW_SIZE, W), min(y0 + WINDOW_SIZE, H)
            patch = image_pil.crop((x0, y0, x1, y1))

            m, _ = segment_patch(model, ProcessorClass, patch, prompt, confidence)

            ph, pw = y1 - y0, x1 - x0
            raw = m.astype(np.float32)

            if raw.shape[0] != ph or raw.shape[1] != pw:
                raw = cv2.resize(raw, (pw, ph), interpolation=cv2.INTER_NEAREST)

            raw = raw[:ph, :pw]

            if raw.shape != (ph, pw):
                log.warning(
                    f"[sliding] patch ignoré : shape invalide "
                    f"(got={raw.shape}, expected={(ph, pw)}) "
                    f"pour patch ({x0},{y0})-({x1},{y1})"
                )
                continue

            accum[y0:y1, x0:x1] += raw
            count[y0:y1, x0:x1] += 1.0

    count = np.maximum(count, 1.0)
    merged = (accum / count) >= 0.1

    if float(merged.sum() / (H * W)) > AREA_RATIO_MAX:
        log.warning(
            f"Masque fusionné trop grand ({float(merged.sum() / (H * W)):.2f}) — seuil relevé à 0.5"
        )
        merged = (accum / count) >= 0.5

    merged = clean_mask(merged)
    return merged, float(accum.max())


# ── Sauvegarde des sorties ─────────────────────────────────────────────────────


def _save_mask(mask, stem, class_name, out_dir):
    p = out_dir / "masks" / class_name / f"{stem}_mask.png"
    Image.fromarray((mask.astype(np.uint8)) * 255, mode="L").save(p)
    return str(p)


def _save_crop(img_rgb, mask, stem, class_name, out_dir, padding=10):
    H, W = img_rgb.shape[:2]
    ys, xs = np.where(mask)
    x1 = max(0, int(xs.min()) - padding)
    y1 = max(0, int(ys.min()) - padding)
    x2 = min(W, int(xs.max()) + padding)
    y2 = min(H, int(ys.max()) + padding)
    masked = img_rgb.copy()
    masked[~mask] = 0
    p = out_dir / "crops" / class_name / f"{stem}_crop.png"
    Image.fromarray(masked[y1:y2, x1:x2]).save(p)
    return str(p)


def _save_overlay(img_rgb, mask, stem, class_name, out_dir, alpha=0.45):
    overlay = img_rgb.copy().astype(np.float32)
    color = PALETTE[0]
    for c in range(3):
        overlay[:, :, c] = np.where(
            mask,
            overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
            overlay[:, :, c],
        )
    seg_pil = Image.fromarray(mask.astype(np.uint8) * 255)
    dilated = np.array(seg_pil.filter(ImageFilter.MaxFilter(3))) > 0
    eroded = np.array(seg_pil.filter(ImageFilter.MinFilter(3))) > 0
    contour = dilated & ~eroded
    for c in range(3):
        overlay[:, :, c] = np.where(contour, color[c], overlay[:, :, c])
    p = out_dir / "overlays" / class_name / f"{stem}_overlay.png"
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(p)
    return p


def _save_json(
    stem,
    image_path,
    class_name,
    shape,
    mask,
    score,
    area_ratio,
    elapsed,
    fallback_used,
    prompt,
    use_clahe,
    use_sw,
    use_fb,
    use_dp,
    out_dir,
):
    ys, xs = np.where(mask)
    data = {
        "image": stem,
        "image_shape": list(shape),
        "class": class_name,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "prompt": prompt,
            "use_clahe": use_clahe,
            "use_sliding_window": use_sw,
            "use_fallback": use_fb,
            "use_dual_prompt": use_dp,
        },
        "score": round(float(score), 4),
        "area_px": int(mask.sum()),
        "area_ratio": round(float(area_ratio), 4),
        "bbox_xywh": [
            int(xs.min()),
            int(ys.min()),
            int(xs.max() - xs.min()),
            int(ys.max() - ys.min()),
        ],
        "centroid": {"x": int(xs.mean()), "y": int(ys.mean())},
        "fallback_used": fallback_used,
        "time_s": round(float(elapsed), 4),
    }
    p = out_dir / "json" / class_name / f"{stem}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _make_result(
    status,
    image_path,
    class_name,
    score,
    area,
    elapsed,
    fallback_used,
    error=None,
    area_ratio=0.0,
    overlay_path=None,
):
    return {
        "status": status,
        "image": str(image_path),
        "class": class_name,
        "score": round(float(score), 4),
        "area": int(area),
        "area_ratio": round(float(area_ratio), 4),
        "time_s": round(float(elapsed), 4),
        "fallback_used": fallback_used,
        "error": error,
        "overlay_path": overlay_path,
    }


# ── Traitement d'une image ─────────────────────────────────────────────────────


def process_image(
    model,
    ProcessorClass,
    image_path: Path,
    out_dir: Path,
    use_clahe: bool,
    use_sliding_window: bool,
    use_fallback: bool,
    use_dual_prompt: bool,
    primary_prompt: str,
    confidence: float = CONFIDENCE_DEFAULT,
) -> dict:
    """
    Traite une image selon la configuration du run.

    Ordre :
    1. Lecture image
    2. CLAHE (si use_clahe)
    3. Segmentation avec prompt principal
    4. Dual prompt "diseased leaves" + fusion (si use_dual_prompt)
    5. Fallback sur prompts génériques (si use_fallback et masque vide)
    6. Sauvegarde des sorties
    """
    class_name = image_path.parent.name
    stem = image_path.stem
    t0 = time.perf_counter()

    for sub in ["masks", "crops", "overlays", "json"]:
        (out_dir / sub / class_name).mkdir(parents=True, exist_ok=True)

    try:
        pil_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        log.warning(f"Image corrompue : {image_path} — {e}")
        return _make_result(
            "error", image_path, class_name, 0.0, 0, 0.0, None, error=str(e)
        )

    # 1. CLAHE
    if use_clahe:
        pil_img = apply_clahe(pil_img)

    img_rgb = np.array(pil_img)
    H, W = img_rgb.shape[:2]
    seg_fn = segment_sliding_window if use_sliding_window else segment_patch

    # 2. Prompt principal
    mask_final, score_final = seg_fn(
        model, ProcessorClass, pil_img, primary_prompt, confidence
    )

    # 3. Dual prompt — "diseased leaves" en parallèle
    # "leaves" capte la partie verte, "diseased leaves" la zone nécrosée.
    # On fusionne uniquement si le gain de surface est > UNION_MIN_GAIN.
    if use_dual_prompt:
        mask_dual, score_dual = seg_fn(
            model, ProcessorClass, pil_img, "diseased leaves", confidence
        )

        if mask_dual.any():
            union = clean_mask(np.logical_or(mask_final, mask_dual))

            if mask_final.any():
                old_area = int(mask_final.sum())
                union_area = int(union.sum())

                if union_area > old_area * UNION_MIN_GAIN:
                    gain_px = union_area - old_area
                    mask_final = union
                    score_final = max(score_final, score_dual)
                    log.debug(f"[dual] gain +{gain_px} px")

            else:
                # prompt principal totalement vide → on prend le dual tel quel
                mask_final = mask_dual
                score_final = score_dual

    fallback_used = None

    # 4. Fallback — uniquement si le masque est vide ou trop petit
    if use_fallback and (not mask_final.any() or mask_final.sum() < MASK_MIN_PX):
        fallback_prompts = FALLBACK_PROMPTS.copy()

        # Si le dual prompt est déjà activé, inutile de retester "diseased leaves"
        if use_dual_prompt:
            fallback_prompts = [p for p in fallback_prompts if p != "diseased leaves"]

        for fb_prompt in fallback_prompts:
            fb_mask, fb_score = seg_fn(
                model, ProcessorClass, pil_img, fb_prompt, FALLBACK_CONFIDENCE
            )
            if fb_mask.any() and fb_mask.sum() >= MASK_MIN_PX:
                mask_final = fb_mask
                score_final = fb_score
                fallback_used = fb_prompt
                log.info(f"[fallback '{fb_prompt}'] score={fb_score:.3f}")
                break

    elapsed = time.perf_counter() - t0

    # 5. empty_mask si tout a échoué
    if not mask_final.any() or mask_final.sum() < MASK_MIN_PX:
        log.info(f"empty_mask : {class_name}/{image_path.name}")
        return _make_result("empty_mask", image_path, class_name, 0.0, 0, elapsed, None)

    # 6. Sauvegarde
    area = int(mask_final.sum())
    area_ratio = area / (H * W)

    _save_mask(mask_final, stem, class_name, out_dir)
    _save_crop(img_rgb, mask_final, stem, class_name, out_dir)
    overlay_path = _save_overlay(img_rgb, mask_final, stem, class_name, out_dir)
    _save_json(
        stem,
        image_path,
        class_name,
        img_rgb.shape,
        mask_final,
        score_final,
        area_ratio,
        elapsed,
        fallback_used,
        primary_prompt,
        use_clahe,
        use_sliding_window,
        use_fallback,
        use_dual_prompt,
        out_dir,
    )

    return _make_result(
        "ok",
        image_path,
        class_name,
        score_final,
        area,
        elapsed,
        fallback_used=fallback_used,
        area_ratio=area_ratio,
        overlay_path=str(overlay_path),
    )


# ── Métriques globales et par classe ───────────────────────────────────────────


def compute_metrics(results: list) -> dict:
    n_total = len(results)
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_empty = sum(1 for r in results if r["status"] == "empty_mask")
    n_error = sum(1 for r in results if r["status"] == "error")
    n_fallback = sum(1 for r in results if r.get("fallback_used"))

    ok = [r for r in results if r["status"] == "ok"]
    areas = [r["area_ratio"] for r in ok]
    scores = [r["score"] for r in ok]
    times = [r["time_s"] for r in results]

    def safe(fn, lst):
        return round(float(fn(lst)), 4) if lst else 0.0

    global_metrics = {
        "n_total": n_total,
        "n_ok": n_ok,
        "n_empty_mask": n_empty,
        "n_error": n_error,
        "n_fallback": n_fallback,
        "pct_ok": round(100 * n_ok / max(n_total, 1), 2),
        "pct_empty_mask": round(100 * n_empty / max(n_total, 1), 2),
        "pct_fallback": round(100 * n_fallback / max(n_total, 1), 2),
        "area_ratio_mean": safe(np.mean, areas),
        "area_ratio_std": safe(np.std, areas),
        "pct_mask_tiny": round(
            100 * sum(1 for a in areas if a < 0.01) / max(len(ok), 1), 2
        ),
        "pct_mask_small": round(
            100 * sum(1 for a in areas if a < 0.05) / max(len(ok), 1), 2
        ),
        "pct_mask_large": round(
            100 * sum(1 for a in areas if a > 0.85) / max(len(ok), 1), 2
        ),
        "score_mean": safe(np.mean, scores),
        "score_std": safe(np.std, scores),
        "pct_low_score": round(
            100 * sum(1 for s in scores if s < 0.3) / max(len(ok), 1), 2
        ),
        "time_per_image_s": safe(np.mean, times),
        "time_total_s": round(sum(times), 2),
    }

    by_class = defaultdict(list)
    for r in results:
        by_class[r["class"]].append(r)

    per_class = {}
    for cls, cls_res in by_class.items():
        cls_ok = [r for r in cls_res if r["status"] == "ok"]
        cls_empty = sum(1 for r in cls_res if r["status"] == "empty_mask")
        cls_areas = [r["area_ratio"] for r in cls_ok]
        cls_scores = [r["score"] for r in cls_ok]
        per_class[cls] = {
            "n_total": len(cls_res),
            "n_ok": len(cls_ok),
            "n_empty_mask": cls_empty,
            "pct_empty_mask": round(100 * cls_empty / max(len(cls_res), 1), 2),
            "area_ratio_mean": safe(np.mean, cls_areas),
            "score_mean": safe(np.mean, cls_scores),
        }

    return {"global": global_metrics, "per_class": per_class}


# ── Logging MLflow ─────────────────────────────────────────────────────────────


def _log_histograms(results, out_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ok = [r for r in results if r["status"] == "ok"]
        areas = [r["area_ratio"] for r in ok]
        scores = [r["score"] for r in ok]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"Distributions — {out_dir.name}", fontsize=12)

        if areas:
            axes[0].hist(areas, bins=40, color="#3498db", alpha=0.85, edgecolor="white")
            axes[0].axvline(0.05, color="red", linestyle="--", label="< 5%")
            axes[0].axvline(0.85, color="orange", linestyle="--", label="> 85%")
            axes[0].set_title("Distribution area_ratio")
            axes[0].set_xlabel("area_ratio")
            axes[0].set_ylabel("count")
            axes[0].grid(alpha=0.3)
            axes[0].legend(fontsize=8)

        if scores:
            axes[1].hist(
                scores, bins=40, color="#2ecc71", alpha=0.85, edgecolor="white"
            )
            axes[1].axvline(0.5, color="red", linestyle="--", label="confidence=0.5")
            axes[1].axvline(
                0.3, color="orange", linestyle="--", label="low score (0.3)"
            )
            axes[1].set_title("Distribution score")
            axes[1].set_xlabel("score")
            axes[1].set_ylabel("count")
            axes[1].grid(alpha=0.3)
            axes[1].legend(fontsize=8)

        plt.tight_layout()
        hist_path = out_dir / "histograms.png"
        plt.savefig(hist_path, dpi=150, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(str(hist_path))
        log.info(f"Histogrammes loggués : {hist_path}")

    except Exception as e:
        log.warning(f"Histogrammes non générés : {e}")


def log_to_mlflow(metrics, config, results, out_dir, report_path):
    # Paramètres
    for k, v in config.items():
        mlflow.log_param(k, v)

    # Métriques globales
    for k, v in metrics["global"].items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(k, v)

    # Métriques par classe (préfixées)
    for cls, cls_m in metrics["per_class"].items():
        short = cls.replace("tomato_", "")
        for k, v in cls_m.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"{short}__{k}", v)

    # Rapport JSON
    mlflow.log_artifact(str(report_path))

    # Overlays d'exemple — classes difficiles uniquement
    sample_counts = defaultdict(int)
    for r in results:
        cls = r.get("class", "")
        if (
            cls in SAMPLE_CLASSES
            and r["status"] == "ok"
            and r.get("overlay_path")
            and sample_counts[cls] < SAMPLES_PER_CLASS
        ):
            mlflow.log_artifact(
                r["overlay_path"], artifact_path=f"overlays_sample/{cls}"
            )
            sample_counts[cls] += 1

    _log_histograms(results, out_dir)


# ── Pipeline principal ─────────────────────────────────────────────────────────


def run_pipeline(args):
    input_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "prompt": args.prompt,
        "use_clahe": args.use_clahe,
        "use_sliding_window": args.use_sliding_window,
        "use_fallback": args.use_fallback,
        "use_dual_prompt": args.use_dual_prompt,
        "confidence": args.confidence,
    }

    log.info("=" * 70)
    log.info(f"Run : {args.run_name}")
    log.info(f"Config : {config}")
    log.info("=" * 70)

    images = sorted(
        [p for p in input_path.rglob("*") if p.suffix.lower() in SUPPORTED_EXT]
    )
    if args.limit:
        images = images[: args.limit]
    log.info(f"{len(images)} images trouvées.")

    device = get_device(args.device)

    # ── MLflow : URI depuis variable d'environnement (définie par le SLURM)
    # Fallback : mlruns/ à côté du dossier results/
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not mlflow_uri:
        mlruns_dir = out_dir.parent.parent / "mlruns"
        mlruns_dir.mkdir(parents=True, exist_ok=True)
        mlflow_uri = f"file://{mlruns_dir.resolve()}"
        os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri

    mlflow.set_tracking_uri(mlflow_uri)
    log.info(f"MLflow URI : {mlflow_uri}")

    mlflow.set_experiment("sam3_segmentation")

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_param("device", device)
        mlflow.log_param("n_images", len(images))

        model, ProcessorClass = load_sam3(device)

        ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device == "cuda"
            else nullcontext()
        )

        results = []
        t_global = time.perf_counter()

        with torch.inference_mode():
            with ctx:
                for i, img_path in enumerate(images):
                    r = process_image(
                        model,
                        ProcessorClass,
                        img_path,
                        out_dir,
                        use_clahe=args.use_clahe,
                        use_sliding_window=args.use_sliding_window,
                        use_fallback=args.use_fallback,
                        use_dual_prompt=args.use_dual_prompt,
                        primary_prompt=args.prompt,
                        confidence=args.confidence,
                    )
                    results.append(r)

                    if (i + 1) % 200 == 0 or (i + 1) == len(images):
                        elapsed = time.perf_counter() - t_global
                        n_ok = sum(1 for x in results if x["status"] == "ok")
                        n_empty = sum(1 for x in results if x["status"] == "empty_mask")
                        n_err = sum(1 for x in results if x["status"] == "error")
                        log.info(
                            f"[{i + 1}/{len(images)}] {elapsed:.0f}s — "
                            f"ok={n_ok}  empty={n_empty}  error={n_err}"
                        )

        metrics = compute_metrics(results)
        metrics["global"]["time_total_s"] = round(time.perf_counter() - t_global, 2)

        report = {
            "run_name": args.run_name,
            "config": config,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "results": results,
        }
        report_path = out_dir / "pipeline_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        log_to_mlflow(metrics, config, results, out_dir, report_path)

        g = metrics["global"]
        log.info("=" * 70)
        log.info(f"RÉSULTATS — {args.run_name}")
        log.info(f"  ok           : {g['n_ok']} ({g['pct_ok']}%)")
        log.info(f"  empty_mask   : {g['n_empty_mask']} ({g['pct_empty_mask']}%)")
        log.info(f"  fallback     : {g['n_fallback']} ({g['pct_fallback']}%)")
        log.info(f"  error        : {g['n_error']}")
        log.info(f"  area_mean    : {g['area_ratio_mean']} ± {g['area_ratio_std']}")
        log.info(f"  mask_tiny    : {g['pct_mask_tiny']}%")
        log.info(f"  mask_small   : {g['pct_mask_small']}%")
        log.info(f"  mask_large   : {g['pct_mask_large']}%")
        log.info(f"  score_mean   : {g['score_mean']} ± {g['score_std']}")
        log.info(f"  low_score    : {g['pct_low_score']}%")
        log.info(f"  t/image      : {g['time_per_image_s']}s")
        log.info(f"  total        : {g['time_total_s']}s")
        log.info("=" * 70)

    return results


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline SAM3 — étude d'ablation contrôlée",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument(
        "--run_name", "-n", required=True, help="Nom du run MLflow (ex: run_0_baseline)"
    )
    parser.add_argument("--prompt", default="leaf")
    parser.add_argument("--use_clahe", action="store_true")
    parser.add_argument("--use_sliding_window", action="store_true")
    parser.add_argument("--use_fallback", action="store_true")
    parser.add_argument("--use_dual_prompt", action="store_true")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_DEFAULT)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--limit", type=int, default=None, help="Limiter aux N premières images (test)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
