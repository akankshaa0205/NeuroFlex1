from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_figure(figure: plt.Figure, output: Path) -> None:
    figure.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def system_architecture(output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(13.33, 7.5))
    axis.set(xlim=(0, 15), ylim=(0, 8))
    axis.axis("off")
    stages = [
        (0.3, "Webcam", "RGB frames", "#dbeafe"),
        (3.2, "MediaPipe", "33 pose + 21 hand\nlandmarks", "#dcfce7"),
        (6.1, "Motion engine", "angles · ROM · phase\nDTW · repetition", "#fef3c7"),
        (9.0, "Intelligence", "personal rules\ngesture MLP · KIMORE", "#f3e8ff"),
        (11.9, "Patient UI", "live cues · score\nprogress report", "#fee2e2"),
    ]
    for index, (x, title, detail, color) in enumerate(stages):
        axis.add_patch(FancyBboxPatch((x, 3.1), 2.25, 1.75, boxstyle="round,pad=0.16",
                                      facecolor=color, edgecolor="#243044", linewidth=1.6))
        axis.text(x + 1.125, 4.25, title, ha="center", fontsize=14, weight="bold")
        axis.text(x + 1.125, 3.7, detail, ha="center", va="center", fontsize=10.5)
        if index < len(stages) - 1:
            axis.add_patch(FancyArrowPatch((x + 2.3, 3.98), (x + 2.82, 3.98),
                                           arrowstyle="-|>", mutation_scale=18,
                                           linewidth=1.8, color="#c9232c"))
    axis.text(7.5, 6.75, "NeuroFlex end-to-end architecture", ha="center",
              fontsize=23, weight="bold")
    axis.text(7.5, 6.1, "A bounded latest-frame pipeline keeps feedback current instead of building delay",
              ha="center", fontsize=12, color="#475569")
    axis.text(7.5, 1.85,
              "Deterministic safety and scoring remain primary; learned models add touch-free control and experimental quality evidence.",
              ha="center", fontsize=11.5, color="#475569")
    save_figure(figure, output_dir / "system_architecture")


def hybrid_architecture(output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(13.33, 7.5))
    axis.set(xlim=(0, 14), ylim=(0, 8))
    axis.axis("off")
    axis.text(7, 7.45, "Hybrid CPU–GPU execution strategy", ha="center",
              fontsize=23, weight="bold")
    axis.text(7, 6.85, "Automatic device selection keeps the demo portable and enables future acceleration",
              ha="center", fontsize=12, color="#475569")
    boxes = [
        (0.7, 2.1, 5.6, 4.0, "CPU path · active on this laptop",
         "Webcam and desktop UI\nMediaPipe pose and hand inference\nAngles, DTW and repetition state\nGesture MLP and Extra Trees inference\nReports and local storage",
         "#dcfce7", "#14805e"),
        (7.7, 2.1, 5.6, 4.0, "CUDA path · optional",
         "Selected only when PyTorch sees CUDA\nBatched vectorized kinematics\nFuture temporal deep-learning models\nReproducible Colab validation notebook\nAutomatic fallback to CPU",
         "#f3e8ff", "#7048c8"),
    ]
    for x, y, width, height, title, detail, fill, edge in boxes:
        axis.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.2",
                                      facecolor=fill, edgecolor=edge, linewidth=2.2))
        axis.text(x + 0.35, y + height - 0.55, title, fontsize=15, weight="bold", color=edge)
        axis.text(x + 0.45, y + height - 1.2, detail, fontsize=12, va="top", linespacing=1.65)
    axis.add_patch(FancyArrowPatch((6.35, 4.1), (7.65, 4.1), arrowstyle="<->",
                                   mutation_scale=20, linewidth=2, color="#334155"))
    axis.text(7, 3.55, "device-aware\nselection", ha="center", fontsize=10, color="#334155")
    axis.text(7, 1.1,
              "Current review claim: measured CPU operation. GPU support exists, but this host exposed no NVIDIA/CUDA runtime.",
              ha="center", fontsize=11.5, weight="bold", color="#c9232c")
    save_figure(figure, output_dir / "cpu_gpu_hybrid")


def dataset_architecture(output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(13.33, 7.5))
    axis.set(xlim=(0, 14), ylim=(0, 8))
    axis.axis("off")

    def box(x: float, y: float, width: float, height: float, title: str, body: str, color: str):
        patch = FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0.16", facecolor=color,
            edgecolor="#233044", linewidth=1.4,
        )
        axis.add_patch(patch)
        axis.text(x + 0.2, y + height - 0.35, title, fontsize=14, weight="bold", va="top")
        axis.text(x + 0.2, y + height - 0.85, body, fontsize=10.5, va="top", linespacing=1.35)

    def arrow(start: tuple[float, float], end: tuple[float, float], label: str):
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=17,
                                       linewidth=1.6, color="#334155"))
        axis.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.15,
                  label, ha="center", fontsize=9.5, color="#334155")

    box(0.4, 5.1, 3.5, 2.0, "UI-PRMD", "Correct/incorrect rehab repetitions\nKinect + Vicon movement streams\nUsed for DTW/scoring validation", "#dbeafe")
    box(5.2, 5.1, 3.5, 2.0, "HaGRID", "Static hand-gesture images\nMediaPipe 21-point hand landmarks\nUsed to train gesture MLP", "#dcfce7")
    box(10.0, 5.1, 3.5, 2.0, "KIMORE", "77 scored subjects · 5 exercises\n284,882 standardized frames\nClinical-score regression", "#fef3c7")
    box(0.4, 1.5, 3.5, 2.0, "Rehab scoring", "Personal baseline → joint features\n→ DTW distance → quality feedback\nStatus: offline validation only", "#eff6ff")
    box(5.2, 1.5, 3.5, 2.0, "Touch-free controls", "Hand landmarks → normalized\nvector → trained MLP → debounce\nStatus: integrated live", "#f0fdf4")
    box(10.0, 1.5, 3.5, 2.0, "Quality estimation", "Subject-grouped Extra Trees model\nLive cross-sensor estimate in UI\nStatus: integrated · experimental", "#fffbeb")
    arrow((2.15, 5.05), (2.15, 3.58), "validate")
    arrow((6.95, 5.05), (6.95, 3.58), "train + validate")
    arrow((11.75, 5.05), (11.75, 3.58), "train + evaluate")
    axis.text(7, 7.65, "How the three datasets support NeuroFlex", ha="center",
              fontsize=20, weight="bold")
    axis.text(7, 0.55, "HaGRID trains gestures; KIMORE trains experimental quality; UI-PRMD validates scoring offline.",
              ha="center", fontsize=12, color="#475569")
    save_figure(figure, output_dir / "dataset_roles")


def gesture_metrics(metrics: dict, output_dir: Path) -> None:
    labels = metrics["labels"]
    matrix = np.asarray(metrics["test_confusion_matrix_detected"], dtype=float)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_totals, out=np.zeros_like(matrix), where=row_totals != 0)

    figure, axes = plt.subplots(1, 2, figsize=(13.33, 5.6), gridspec_kw={"width_ratios": [0.8, 1.4]})
    values = [metrics["test_detected_accuracy"], metrics["test_end_to_end_accuracy_failures_are_errors"]]
    bars = axes[0].bar(["Detected-hand", "End-to-end"], np.multiply(values, 100),
                       color=["#16a34a", "#2563eb"])
    axes[0].axhline(95, color="#dc2626", linestyle="--", linewidth=1.5, label="95% target")
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Gesture accuracy")
    axes[0].legend(frameon=False)
    axes[0].bar_label(bars, fmt="%.1f%%", padding=3, fontsize=11)

    image = axes[1].imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axes[1].set_yticks(range(len(labels)), labels)
    axes[1].set_xlabel("Predicted label")
    axes[1].set_ylabel("True label")
    axes[1].set_title("Detected-hand confusion matrix (row-normalized)")
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = normalized[row, column]
            axes[1].text(column, row, f"{value:.0%}\n({int(matrix[row, column])})",
                         ha="center", va="center", fontsize=8,
                         color="white" if value > 0.55 else "#0f172a")
    figure.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04, label="Share of true class")
    figure.suptitle("HaGRID gesture-classifier results", fontsize=18, weight="bold")
    figure.tight_layout()
    save_figure(figure, output_dir / "hagrid_gesture_metrics")


def ui_prmd_metrics(diagnosis: dict, output_dir: Path) -> None:
    movements = [f"m{i:02d}" for i in range(1, 11)]
    figure, axes = plt.subplots(2, 1, figsize=(13.33, 7.5), sharex=True)
    x = np.arange(len(movements))
    width = 0.19
    styles = [
        ("Kinect", "global", -1.5 * width, "#94a3b8", "Kinect global"),
        ("Kinect", "personal", -0.5 * width, "#2563eb", "Kinect personal"),
        ("Vicon", "global", 0.5 * width, "#d6a760", "Vicon global"),
        ("Vicon", "personal", 1.5 * width, "#16a34a", "Vicon personal"),
    ]
    for axis, metric, title in zip(axes, ("auc", "balanced_accuracy"),
                                    ("AUC by movement", "Balanced accuracy by movement")):
        for sensor, protocol, offset, color, label in styles:
            values = [diagnosis["sensors"][sensor][movement][protocol][metric] for movement in movements]
            axis.bar(x + offset, values, width, color=color, label=label)
        axis.axhline(0.5, color="#dc2626", linestyle="--", linewidth=1, label="chance" if metric == "auc" else None)
        axis.set_ylim(0.3, 1.03)
        axis.set_ylabel(metric.replace("_", " ").title())
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False)
    axes[1].set_xticks(x, movements)
    axes[1].set_xlabel("UI-PRMD movement ID")
    personal = {
        sensor: np.mean(
            [diagnosis["sensors"][sensor][movement]["personal"]["balanced_accuracy"]
             for movement in movements]
        )
        for sensor in ("Kinect", "Vicon")
    }
    figure.suptitle("Patient-specific baselines outperform global templates", fontsize=18, weight="bold", y=1.02)
    figure.text(
        0.5,
        0.005,
        f"Personalized mean balanced accuracy: Kinect {personal['Kinect']:.1%} · "
        f"Vicon {personal['Vicon']:.1%}",
        ha="center",
        fontsize=12,
        weight="bold",
        color="#0f172a",
    )
    figure.tight_layout()
    save_figure(figure, output_dir / "ui_prmd_personalization")


def runtime_metrics(metrics: dict, output_dir: Path) -> None:
    names = ["Joint-angle batch", "Exact DTW"]
    median = [metrics["kinematics"]["median_ms"], metrics["exact_dtw"]["median_ms"]]
    p95 = [metrics["kinematics"]["p95_ms"], metrics["exact_dtw"]["p95_ms"]]
    figure, axes = plt.subplots(1, 2, figsize=(13.33, 5.4))
    x = np.arange(2)
    width = 0.34
    axes[0].bar(x - width / 2, median, width, label="Median", color="#2563eb")
    bars = axes[0].bar(x + width / 2, p95, width, label="p95", color="#16a34a")
    axes[0].set_xticks(x, names)
    axes[0].set_ylabel("Latency (ms, log scale)")
    axes[0].set_yscale("log")
    axes[0].set_title("CPU algorithm latency")
    axes[0].legend(frameon=False)
    axes[0].bar_label(bars, fmt="%.3f ms", padding=3)

    queue = metrics["bounded_queue"]
    values = [queue["processed"], queue["dropped"], queue["failed"]]
    bars = axes[1].bar(["Processed", "Dropped stale", "Failed"], values,
                       color=["#16a34a", "#f59e0b", "#dc2626"])
    axes[1].set_ylabel("Items")
    axes[1].set_title("Bounded-queue overload behavior")
    axes[1].bar_label(bars, padding=3)
    figure.suptitle("Runtime benchmark — measured on CPU, not full webcam latency",
                    fontsize=18, weight="bold")
    figure.tight_layout()
    save_figure(figure, output_dir / "runtime_metrics")


def kimore_metrics(metrics: dict, output_dir: Path) -> None:
    predictions = metrics["test_predictions"]
    actual = np.asarray([row["actual"] for row in predictions])
    predicted = np.asarray([row["predicted"] for row in predictions])
    candidates = metrics["candidate_cv_mae"]
    figure, axes = plt.subplots(1, 2, figsize=(13.33, 5.4))
    axes[0].scatter(actual, predicted, color="#7c3aed", alpha=0.75, edgecolor="white")
    axes[0].plot([0, 50], [0, 50], "--", color="#dc2626", label="ideal")
    axes[0].set(
        xlim=(0, 50), ylim=(0, 50), xlabel="Clinician cTS (0–50)",
        ylabel="Predicted cTS (0–50)", title="Unseen-subject predictions",
    )
    axes[0].legend(frameon=False)
    names = list(candidates)
    bars = axes[1].bar(
        names, [candidates[name] for name in names],
        color=plt.cm.viridis(np.linspace(0.2, 0.85, len(names))),
    )
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].axhline(
        metrics["baseline_mean_mae"], linestyle="--", color="#dc2626",
        label="mean-score baseline",
    )
    axes[1].set(ylabel="Grouped CV MAE", title="Model selection")
    axes[1].bar_label(bars, fmt="%.2f", padding=3)
    axes[1].legend(frameon=False)
    result = metrics["test_metrics"]
    figure.suptitle(
        f"KIMORE quality regression · test MAE {result['mae']:.2f}/50 · "
        f"R² {result['r2']:.3f} · Spearman ρ {result['spearman_rho']:.3f}",
        fontsize=17, weight="bold",
    )
    figure.tight_layout()
    save_figure(figure, output_dir / "kimore_quality_metrics")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate review-ready NeuroFlex figures.")
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("review_outputs"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    gesture = load_json(args.data_root / "hagrid/training_metrics.json")
    diagnosis = load_json(args.data_root / "ui-prmd/diagnosis_20261025.json")
    runtime = load_json(args.data_root / "runtime/benchmark_20261025.json")
    kimore = load_json(args.data_root / "kimore/quality_metrics_20261029.json")
    dataset_architecture(args.output)
    system_architecture(args.output)
    hybrid_architecture(args.output)
    gesture_metrics(gesture, args.output)
    ui_prmd_metrics(diagnosis, args.output)
    runtime_metrics(runtime, args.output)
    kimore_metrics(kimore, args.output)
    print(f"Wrote presentation-ready PNG figures to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
