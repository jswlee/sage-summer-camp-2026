from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

from matplotlib import pyplot as plt
from PIL import Image
from ultralytics import YOLO


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO classification inference on a test set, save results, "
        "compute aggregate metrics, and generate visualizations."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("runs/classify/train/weights/best.pt"),
        help="Path to the trained YOLO classification model.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("yolo_dataset_224/test"),
        help="Test directory containing one subdirectory per ground-truth class.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write inference results and visualizations. "
        "Defaults to '<model_parent>/../test_inference'.",
    )
    parser.add_argument("--batch", type=int, default=32, help="Inference batch size.")
    parser.add_argument("--imgsz", type=int, default=224, help="Inference image size.")
    parser.add_argument("--device", default=None, help="Inference device, such as cuda, cpu, or 0.")
    parser.add_argument(
        "--sample-count",
        type=int,
        default=25,
        help="Number of sample predictions to render in the visualization grid.",
    )
    return parser.parse_args()


def collect_samples(data_dir: Path) -> list[tuple[Path, str]]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    samples: list[tuple[Path, str]] = []
    for class_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                samples.append((image_path, class_dir.name))
    if not samples:
        raise ValueError(f"No images found under class directories in: {data_dir}")
    return samples


def batch_items(items: list, batch_size: int) -> Iterable[list]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def run_inference(
    model: YOLO,
    samples: list[tuple[Path, str]],
    imgsz: int,
    device,
    batch: int,
) -> list[tuple[Path, str, str, float]]:
    results: list[tuple[Path, str, str, float]] = []
    for chunk in batch_items(samples, batch):
        paths, labels = zip(*chunk)
        outputs = model(list(paths), imgsz=imgsz, device=device, verbose=False)
        for path, truth, output in zip(paths, labels, outputs):
            prediction = str(output.names[output.probs.top1])
            confidence = float(output.probs.top1conf)
            results.append((path, truth, prediction, confidence))
    return results


def class_metrics(
    labels: list[str], actual: list[str], predicted: list[str]
) -> dict[str, tuple[float, float, float, int]]:
    metrics: dict[str, tuple[float, float, float, int]] = {}
    for label in labels:
        true_positive = sum(t == label and p == label for t, p in zip(actual, predicted))
        false_positive = sum(t != label and p == label for t, p in zip(actual, predicted))
        false_negative = sum(t == label and p != label for t, p in zip(actual, predicted))
        support = sum(t == label for t in actual)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[label] = (precision, recall, f1, support)
    return metrics


def write_predictions_csv(
    output_path: Path, predictions: list[tuple[Path, str, str, float]]
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("image", "ground_truth", "prediction", "confidence", "correct"))
        for path, truth, prediction, confidence in predictions:
            writer.writerow((path.as_posix(), truth, prediction, f"{confidence:.6f}", prediction == truth))


def write_metrics_csv(
    output_path: Path,
    labels: list[str],
    metrics: dict[str, tuple[float, float, float, int]],
    accuracy: float,
    macro: tuple[float, float, float],
    total: int,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("class", "precision", "recall", "f1", "support"))
        for label in labels:
            precision, recall, f1, support = metrics[label]
            writer.writerow((label, f"{precision:.6f}", f"{recall:.6f}", f"{f1:.6f}", support))
        writer.writerow(())
        writer.writerow(("accuracy", f"{accuracy:.6f}", "", "", total))
        writer.writerow(("macro_avg", f"{macro[0]:.6f}", f"{macro[1]:.6f}", f"{macro[2]:.6f}", total))


def plot_confusion_matrix(output_path: Path, labels: list[str], confusion: Counter) -> None:
    matrix = [[confusion[truth, prediction] for prediction in labels] for truth in labels]
    figure, axis = plt.subplots(figsize=(max(6, len(labels) * 1.5), max(5, len(labels) * 1.25)))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, label="Images")
    axis.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_xlabel("Prediction")
    axis.set_ylabel("Ground truth")
    axis.set_title("Confusion Matrix")
    threshold = max((value for row in matrix for value in row), default=0) / 2
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            color = "white" if value > threshold else "black"
            axis.text(column_index, row_index, str(value), ha="center", va="center", color=color)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_confidence_histogram(
    output_path: Path, predictions: list[tuple[Path, str, str, float]]
) -> None:
    correct_conf = [c for _, t, p, c in predictions if p == t]
    wrong_conf = [c for _, t, p, c in predictions if p != t]
    figure, axis = plt.subplots(figsize=(8, 5))
    bins = [i / 20 for i in range(21)]
    axis.hist([correct_conf, wrong_conf], bins=bins, stacked=True,
              label=["correct", "incorrect"], color=["#2ca02c", "#d62728"])
    axis.set_xlabel("Top-1 confidence")
    axis.set_ylabel("Images")
    axis.set_title("Prediction Confidence Distribution")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_prediction_grid(
    output_path: Path,
    predictions: list[tuple[Path, str, str, float]],
    sample_count: int,
) -> None:
    if sample_count < 1 or not predictions:
        return
    wrong = [item for item in predictions if item[2] != item[1]]
    right = [item for item in predictions if item[2] == item[1]]
    selection: list[tuple[Path, str, str, float]] = []
    selection.extend(wrong[:sample_count])
    if len(selection) < sample_count:
        selection.extend(right[: sample_count - len(selection)])

    columns = min(5, len(selection))
    rows = math.ceil(len(selection) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 2.6, rows * 2.8))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for index, axis in enumerate(axes):
        axis.axis("off")
        if index >= len(selection):
            continue
        path, truth, prediction, confidence = selection[index]
        try:
            with Image.open(path) as raw:
                axis.imshow(raw.convert("RGB"))
        except OSError:
            continue
        correct = prediction == truth
        axis.set_title(
            f"gt: {truth}\npred: {prediction} ({confidence:.2f})",
            fontsize=8,
            color="green" if correct else "red",
        )
    figure.suptitle("Sample Predictions (mistakes first)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def run(
    model: YOLO | Path | str,
    data_dir: Path | str,
    output_dir: Path | str,
    imgsz: int = 224,
    device=None,
    batch: int = 32,
    sample_count: int = 25,
) -> dict:
    """Run inference on the test set, persist results, and generate visualizations.

    Returns a summary dictionary with the computed aggregate metrics.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not isinstance(model, YOLO):
        model = YOLO(model)

    samples = collect_samples(data_dir)
    predictions = run_inference(model, samples, imgsz, device, batch)

    actual = [truth for _, truth, _, _ in predictions]
    predicted = [prediction for _, _, prediction, _ in predictions]
    labels = sorted(set(actual) | set(predicted))
    metrics = class_metrics(labels, actual, predicted)

    correct = sum(t == p for t, p in zip(actual, predicted))
    accuracy = correct / len(actual)
    macro_precision = sum(m[0] for m in metrics.values()) / len(metrics)
    macro_recall = sum(m[1] for m in metrics.values()) / len(metrics)
    macro_f1 = sum(m[2] for m in metrics.values()) / len(metrics)
    confusion = Counter(zip(actual, predicted))

    predictions_csv = output_dir / "test_predictions.csv"
    metrics_csv = output_dir / "test_metrics.csv"
    confusion_png = output_dir / "confusion_matrix.png"
    confidence_png = output_dir / "confidence_histogram.png"
    grid_png = output_dir / "sample_predictions.png"

    write_predictions_csv(predictions_csv, predictions)
    write_metrics_csv(metrics_csv, labels, metrics, accuracy, (macro_precision, macro_recall, macro_f1), len(actual))
    plot_confusion_matrix(confusion_png, labels, confusion)
    plot_confidence_histogram(confidence_png, predictions)
    plot_prediction_grid(grid_png, predictions, sample_count)

    print(f"output dir: {output_dir}")
    print(f"samples: {len(actual)}")
    print(f"accuracy: {accuracy:.4f} ({correct}/{len(actual)})")
    print(f"macro precision: {macro_precision:.4f}")
    print(f"macro recall: {macro_recall:.4f}")
    print(f"macro f1: {macro_f1:.4f}")
    print("\nPer-class metrics:")
    print(f"{'class':<16} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}")
    for label in labels:
        precision, recall, f1, support = metrics[label]
        print(f"{label:<16} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10}")
    print(f"\nsaved: {predictions_csv.name}, {metrics_csv.name}, "
          f"{confusion_png.name}, {confidence_png.name}, {grid_png.name}")

    return {
        "output_dir": output_dir,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": metrics,
        "samples": len(actual),
    }


def main() -> None:
    args = parse_args()
    if args.batch < 1:
        raise ValueError("--batch must be at least 1")
    if args.imgsz < 1:
        raise ValueError("--imgsz must be at least 1")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model file does not exist: {args.model}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.model.parent.parent / "test_inference"

    run(
        model=args.model,
        data_dir=args.data_dir,
        output_dir=output_dir,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        sample_count=args.sample_count,
    )


if __name__ == "__main__":
    main()
