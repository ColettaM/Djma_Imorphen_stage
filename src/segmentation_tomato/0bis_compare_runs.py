"""
compare_runs.py — Comparaison quantitative et qualitative des runs SAM3
========================================================================

But
---
Comparer les runs d'ablation produits par segment.py et générer :
- un tableau de synthèse console
- des graphiques comparatifs
- une grille qualitative sur une même image
- un rapport Markdown
- un run MLflow dédié à la comparaison

Runs attendus
-------------
- run_0_baseline
- run_1_prompt_leaves
- run_2_sliding_only
- run_3_clahe_only
- run_4_sliding_clahe
- run_5_fallback
- run_6_dual_prompt

Usage
-----
python compare_runs.py --results_dir results
python compare_runs.py --results_dir results --sample_stem IMG_1234
"""

import os
import json
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mlflow


# ── Configuration ──────────────────────────────────────────────────────────────

RUNS = [
    ("run_0_baseline", "baseline\n(leaf)"),
    ("run_1_prompt_leaves", "prompt\n(+leaves)"),
    ("run_2_sliding_only", "sliding\nonly"),
    ("run_3_clahe_only", "CLAHE\nonly"),
    ("run_4_sliding_clahe", "SW+\nCLAHE"),
    ("run_5_fallback", "+fallback"),
    ("run_6_dual_prompt", "+dual\nprompt"),
]

COLORS = ["#e74c3c", "#e67e22", "#3498db", "#1abc9c", "#2ecc71", "#9b59b6", "#34495e"]

CLASSES_DIFFICILES = [
    "tomato_late_blight",
    "tomato_mosaic_virus",
    "tomato_verticulium_wilt",
]


# ── Chargement ─────────────────────────────────────────────────────────────────


def load_report(results_dir: Path, run_name: str):
    p = results_dir / run_name / "pipeline_report.json"
    if not p.exists():
        print(f"[WARNING] Rapport introuvable : {p}")
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_reports(results_dir: Path) -> dict:
    reports = {}
    for run_key, _ in RUNS:
        r = load_report(results_dir, run_key)
        if r is not None:
            reports[run_key] = r
    return reports


# ── Helpers métriques ──────────────────────────────────────────────────────────


def get_metric(report: dict, key: str, default=0.0):
    return report.get("metrics", {}).get("global", {}).get(key, default)


def get_class_metric(report: dict, class_name: str, key: str, default=0.0):
    return (
        report.get("metrics", {})
        .get("per_class", {})
        .get(class_name, {})
        .get(key, default)
    )


def get_pct_error(report: dict) -> float:
    n_total = get_metric(report, "n_total", 0)
    n_error = get_metric(report, "n_error", 0)
    return round(100.0 * n_error / max(n_total, 1), 2)


# ── Tableau console ────────────────────────────────────────────────────────────


def print_summary_table(reports: dict):
    header = (
        f"{'Run':<22} {'%ok':>6} {'%empty':>8} {'%err':>7} "
        f"{'%small':>8} {'%large':>8} {'%low':>8} "
        f"{'area_mean':>10} {'score_mean':>11} {'t/img(s)':>10}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for run_key, _ in RUNS:
        if run_key not in reports:
            continue
        r = reports[run_key]
        print(
            f"{run_key:<22} "
            f"{get_metric(r, 'pct_ok'):>6.2f} "
            f"{get_metric(r, 'pct_empty_mask'):>8.2f} "
            f"{get_pct_error(r):>7.2f} "
            f"{get_metric(r, 'pct_mask_small'):>8.2f} "
            f"{get_metric(r, 'pct_mask_large'):>8.2f} "
            f"{get_metric(r, 'pct_low_score'):>8.2f} "
            f"{get_metric(r, 'area_ratio_mean'):>10.4f} "
            f"{get_metric(r, 'score_mean'):>11.4f} "
            f"{get_metric(r, 'time_per_image_s'):>10.4f}"
        )
    print("=" * len(header) + "\n")


# ── Graphique 1 — Vue globale ──────────────────────────────────────────────────


def plot_overview(reports: dict, output_dir: Path) -> Path:
    """Statuts pipeline / indicateurs qualité / temps."""
    available = [
        (run_key, run_label) for run_key, run_label in RUNS if run_key in reports
    ]
    n = len(available)
    x = np.arange(n)
    labels = [run_label.replace("\n", " ") for _, run_label in available]
    colors = COLORS[:n]

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    fig.suptitle("Comparaison globale des runs SAM3", fontsize=14, fontweight="bold")

    # 1. Statuts pipeline
    ax = axes[0]
    pct_ok = [get_metric(reports[k], "pct_ok") for k, _ in available]
    pct_empty = [get_metric(reports[k], "pct_empty_mask") for k, _ in available]
    pct_error = [get_pct_error(reports[k]) for k, _ in available]
    ax.bar(x, pct_ok, color="#2ecc71", alpha=0.85, label="% ok")
    ax.bar(
        x,
        pct_empty,
        bottom=np.array(pct_ok),
        color="#e74c3c",
        alpha=0.85,
        label="% empty_mask",
    )
    ax.bar(
        x,
        pct_error,
        bottom=np.array(pct_ok) + np.array(pct_empty),
        color="#7f8c8d",
        alpha=0.85,
        label="% error",
    )
    ax.set_title("Statuts pipeline (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    for i, v in enumerate(pct_ok):
        ax.text(
            i,
            max(v / 2, 2),
            f"{v:.1f}%",
            ha="center",
            va="center",
            fontsize=7,
            color="white",
            fontweight="bold",
        )

    # 2. Indicateurs qualité masque
    ax = axes[1]
    pct_small = [get_metric(reports[k], "pct_mask_small") for k, _ in available]
    pct_large = [get_metric(reports[k], "pct_mask_large") for k, _ in available]
    pct_low = [get_metric(reports[k], "pct_low_score") for k, _ in available]
    w = 0.24
    ax.bar(x - w, pct_small, width=w, color="#3498db", alpha=0.85, label="% mask_small")
    ax.bar(x, pct_large, width=w, color="#f39c12", alpha=0.85, label="% mask_large")
    ax.bar(x + w, pct_low, width=w, color="#9b59b6", alpha=0.85, label="% low_score")
    ax.set_title("Indicateurs de qualité (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    # 3. Temps / image
    ax = axes[2]
    times = [get_metric(reports[k], "time_per_image_s") for k, _ in available]
    ax.bar(x, times, color=colors, alpha=0.85)
    ax.set_title("Temps de traitement / image (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    if times:
        ymax = max(times)
        for i, t in enumerate(times):
            ax.text(i, t + ymax * 0.02, f"{t:.3f}s", ha="center", fontsize=7)

    plt.tight_layout()
    out = output_dir / "comparison_overview.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")
    return out


# ── Graphique 2 — Area ratio et score ─────────────────────────────────────────


def plot_area_score(reports: dict, output_dir: Path) -> Path:
    """Area ratio moyen et score moyen par run."""
    available = [
        (run_key, run_label) for run_key, run_label in RUNS if run_key in reports
    ]
    n = len(available)
    x = np.arange(n)
    labels = [run_label.replace("\n", " ") for _, run_label in available]
    colors = COLORS[:n]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Qualité moyenne des sorties", fontsize=14, fontweight="bold")

    # area_ratio_mean ± std
    ax = axes[0]
    means = [get_metric(reports[k], "area_ratio_mean") for k, _ in available]
    stds = [get_metric(reports[k], "area_ratio_std") for k, _ in available]
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85)
    ax.axhline(
        0.05, color="red", linestyle="--", linewidth=1, alpha=0.7, label="5% (petit)"
    )
    ax.axhline(
        0.85,
        color="orange",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label="85% (grand)",
    )
    ax.set_title("Area ratio moyen (± std)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    # score_mean ± std
    ax = axes[1]
    means = [get_metric(reports[k], "score_mean") for k, _ in available]
    stds = [get_metric(reports[k], "score_std") for k, _ in available]
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85)
    ax.axhline(
        0.3,
        color="red",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label="score faible (0.3)",
    )
    ax.set_title("Score moyen (± std)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = output_dir / "comparison_area_score.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")
    return out


# ── Graphique 3 — Classes difficiles ──────────────────────────────────────────


def plot_per_class(reports: dict, output_dir: Path) -> Path:
    """% empty_mask sur les classes difficiles par run."""
    available = [
        (run_key, run_label) for run_key, run_label in RUNS if run_key in reports
    ]
    n_runs = len(available)
    n_classes = len(CLASSES_DIFFICILES)
    labels = [run_label.replace("\n", " ") for _, run_label in available]
    x = np.arange(n_runs)

    fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 5))
    fig.suptitle("% empty_mask par classe difficile", fontsize=13, fontweight="bold")
    if n_classes == 1:
        axes = [axes]

    for ax, cls in zip(axes, CLASSES_DIFFICILES):
        values = [
            get_class_metric(reports[k], cls, "pct_empty_mask") for k, _ in available
        ]
        bars = ax.bar(x, values, 0.6, color=COLORS[:n_runs], alpha=0.85)
        ax.set_title(cls.replace("tomato_", ""), fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=15)
        ax.set_ylabel("% empty_mask")
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{v:.1f}%",
                ha="center",
                fontsize=7,
            )

    plt.tight_layout()
    out = output_dir / "per_class_empty_mask.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")
    return out


# ── Grille qualitative ─────────────────────────────────────────────────────────


def choose_sample_stem(results_dir: Path, available_runs: list, sample_stem):
    """Choisit un stem commun à tous les runs, en priorité dans les classes difficiles."""
    if sample_stem is not None:
        return sample_stem

    candidates = None
    for run_key, _ in available_runs:
        overlay_root = results_dir / run_key / "overlays"
        if not overlay_root.exists():
            continue
        stems = {
            p.stem.replace("_overlay", "") for p in overlay_root.rglob("*_overlay.png")
        }
        candidates = stems if candidates is None else candidates & stems

    if not candidates:
        return None

    ref_run = available_runs[0][0]
    for cls in CLASSES_DIFFICILES:
        cls_stems = {
            s
            for s in candidates
            if (results_dir / ref_run / "overlays" / cls / f"{s}_overlay.png").exists()
        }
        if cls_stems:
            return sorted(cls_stems)[0]

    return sorted(candidates)[0]


def plot_qualitative_grid(results_dir: Path, output_dir: Path, sample_stem=None):
    """Affiche l'overlay de la même image pour tous les runs disponibles."""
    import cv2

    available = [
        (run_key, run_label)
        for run_key, run_label in RUNS
        if (results_dir / run_key).exists()
    ]
    if not available:
        return None

    chosen = choose_sample_stem(results_dir, available, sample_stem)
    if chosen is None:
        print("[WARNING] Aucun stem commun trouvé pour la grille qualitative.")
        return None

    print(f"[INFO] Grille qualitative sur : {chosen}")
    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    fig.suptitle(f"Comparaison qualitative — {chosen}", fontsize=11, fontweight="bold")
    if n == 1:
        axes = [axes]

    for ax, (run_key, label), color in zip(axes, available, COLORS):
        files = list(
            (results_dir / run_key / "overlays").rglob(f"{chosen}_overlay.png")
        )
        if files:
            img = cv2.imread(str(files[0]))
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.set_facecolor("#dddddd")
            ax.text(
                0.5,
                0.5,
                "absent / empty_mask",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
                color="#555555",
            )
        ax.set_title(
            label.replace("\n", " "), fontsize=8, color=color, fontweight="bold"
        )
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    out = output_dir / "qualitative_grid.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")
    return out


# ── Rapport Markdown ───────────────────────────────────────────────────────────


def build_markdown(reports: dict) -> str:
    md = ["# Comparaison des runs SAM3\n"]

    md.append("## Tableau de synthèse\n")
    md.append(
        "| Run | %ok | %empty | %error | %mask_small | %mask_large "
        "| %low_score | area_mean | score_mean | t/img (s) |"
    )
    md.append(
        "|-----|----:|-------:|-------:|------------:|------------:"
        "|-----------:|----------:|-----------:|----------:|"
    )
    for run_key, _ in RUNS:
        if run_key not in reports:
            continue
        r = reports[run_key]
        md.append(
            f"| {run_key} "
            f"| {get_metric(r, 'pct_ok'):.2f} "
            f"| {get_metric(r, 'pct_empty_mask'):.2f} "
            f"| {get_pct_error(r):.2f} "
            f"| {get_metric(r, 'pct_mask_small'):.2f} "
            f"| {get_metric(r, 'pct_mask_large'):.2f} "
            f"| {get_metric(r, 'pct_low_score'):.2f} "
            f"| {get_metric(r, 'area_ratio_mean'):.4f} "
            f"| {get_metric(r, 'score_mean'):.4f} "
            f"| {get_metric(r, 'time_per_image_s'):.4f} |"
        )

    md.append("\n## Analyse incrémentale\n")

    pairs = [
        ("run_0_baseline", "run_1_prompt_leaves", "Prompt `leaf` → `leaves`"),
        ("run_1_prompt_leaves", "run_2_sliding_only", "Ajout du sliding window"),
        ("run_1_prompt_leaves", "run_3_clahe_only", "Ajout du CLAHE seul (isolation)"),
        (
            "run_2_sliding_only",
            "run_4_sliding_clahe",
            "Ajout CLAHE après sliding window",
        ),
        ("run_4_sliding_clahe", "run_5_fallback", "Ajout du fallback"),
        ("run_5_fallback", "run_6_dual_prompt", "Ajout du dual prompt"),
    ]

    for prev_key, next_key, title in pairs:
        if prev_key not in reports or next_key not in reports:
            continue
        p, n = reports[prev_key], reports[next_key]

        d_empty = get_metric(p, "pct_empty_mask") - get_metric(n, "pct_empty_mask")
        d_small = get_metric(n, "pct_mask_small") - get_metric(p, "pct_mask_small")
        d_large = get_metric(n, "pct_mask_large") - get_metric(p, "pct_mask_large")
        d_low = get_metric(n, "pct_low_score") - get_metric(p, "pct_low_score")
        d_area = get_metric(n, "area_ratio_mean") - get_metric(p, "area_ratio_mean")
        d_score = get_metric(n, "score_mean") - get_metric(p, "score_mean")
        d_time = get_metric(p, "time_per_image_s") - get_metric(n, "time_per_image_s")

        md.append(f"### {title}")
        md.append(
            f"- Δ empty_mask : {'−' if d_empty >= 0 else '+'}{abs(d_empty):.2f} pts"
        )
        md.append(f"- Δ mask_small : {'+' if d_small >= 0 else ''}{d_small:.2f} pts")
        md.append(f"- Δ mask_large : {'+' if d_large >= 0 else ''}{d_large:.2f} pts")
        md.append(f"- Δ low_score  : {'+' if d_low >= 0 else ''}{d_low:.2f} pts")
        md.append(f"- Δ area_mean  : {'+' if d_area >= 0 else ''}{d_area:.4f}")
        md.append(f"- Δ score_mean : {'+' if d_score >= 0 else ''}{d_score:.4f}")
        md.append(f"- Δ t/image    : {'+' if d_time >= 0 else ''}{d_time:.4f} s\n")

    md.append("## Conclusion prudente\n")
    if "run_0_baseline" in reports and "run_6_dual_prompt" in reports:
        b = reports["run_0_baseline"]
        f = reports["run_6_dual_prompt"]
        md.append(
            f"Les améliorations successives réduisent les échecs de segmentation "
            f"de {get_metric(b, 'pct_empty_mask'):.2f}% (baseline) "
            f"à {get_metric(f, 'pct_empty_mask'):.2f}% (version finale). "
            f"La version finale présente {get_metric(f, 'pct_mask_small'):.2f}% de masques petits, "
            f"{get_metric(f, 'pct_mask_large'):.2f}% de masques très grands "
            f"et {get_metric(f, 'pct_low_score'):.2f}% de scores faibles. "
            "Une vérification visuelle complémentaire reste nécessaire pour confirmer "
            "que les erreurs résiduelles sont suffisamment limitées avant passage à la classification."
        )

    return "\n".join(md)


# ── Point d'entrée ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument(
        "--sample_stem", default=None, help="Stem d'image pour la grille qualitative"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = results_dir / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    reports = load_all_reports(results_dir)
    if not reports:
        print("[ERROR] Aucun pipeline_report.json trouvé.")
        return

    print(f"[INFO] {len(reports)} run(s) chargé(s) : {list(reports.keys())}")
    print_summary_table(reports)

    # MLflow : URI depuis variable d'environnement (définie par job_compare.slurm)
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not mlflow_uri:
        mlruns_dir = results_dir.parent / "mlruns"
        mlruns_dir.mkdir(parents=True, exist_ok=True)
        mlflow_uri = f"file://{mlruns_dir.resolve()}"
        os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("sam3_segmentation")

    with mlflow.start_run(run_name="comparison"):
        # Figures
        fig_overview = plot_overview(reports, output_dir)
        fig_area_score = plot_area_score(reports, output_dir)
        fig_per_class = plot_per_class(reports, output_dir)
        fig_qualitative = plot_qualitative_grid(
            results_dir, output_dir, args.sample_stem
        )

        # Rapport Markdown
        md_text = build_markdown(reports)
        md_path = output_dir / "comparison_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_text)
        print(f"[OK] {md_path}")

        # Artifacts MLflow
        for fig in [
            fig_overview,
            fig_area_score,
            fig_per_class,
            fig_qualitative,
            md_path,
        ]:
            if fig and Path(fig).exists():
                mlflow.log_artifact(str(fig))

        # Métriques delta baseline → final
        if "run_0_baseline" in reports and "run_6_dual_prompt" in reports:
            b = reports["run_0_baseline"]
            f = reports["run_6_dual_prompt"]
            mlflow.log_metric(
                "delta_pct_empty_mask",
                get_metric(b, "pct_empty_mask") - get_metric(f, "pct_empty_mask"),
            )
            mlflow.log_metric(
                "delta_pct_mask_small",
                get_metric(b, "pct_mask_small") - get_metric(f, "pct_mask_small"),
            )
            mlflow.log_metric(
                "delta_area_ratio_mean",
                get_metric(f, "area_ratio_mean") - get_metric(b, "area_ratio_mean"),
            )
            mlflow.log_metric(
                "delta_score_mean",
                get_metric(f, "score_mean") - get_metric(b, "score_mean"),
            )
            mlflow.log_metric(
                "delta_time_per_image_s",
                get_metric(b, "time_per_image_s") - get_metric(f, "time_per_image_s"),
            )

        print("\n[DONE] Sorties générées :")
        for fig in [
            fig_overview,
            fig_area_score,
            fig_per_class,
            fig_qualitative,
            md_path,
        ]:
            if fig and Path(fig).exists():
                print(f"  ✓ {fig}")


if __name__ == "__main__":
    main()
