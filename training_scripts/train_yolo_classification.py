import argparse
from pathlib import Path

from ultralytics import YOLO

import visualize_yolo_classification


DATA = "yolo_dataset_daynight_224"


def main():
    parser = argparse.ArgumentParser(description="Train a YOLO classification model.")
    parser.add_argument("--model", default="yolo26n-cls.pt", help="Model weights path")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--imgsz", type=int, default=224, help="Image size")
    parser.add_argument("--device", default="cuda", help="Device to train on")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    args = parser.parse_args()

    model = YOLO(args.model)
    results = model.train(
        data=DATA,
        epochs=args.epochs,
        seed=42,
        patience=args.patience,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
    )

    save_dir = Path(results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    test_dir = Path(DATA) / "test"

    print("\nRunning inference and visualization on the test set...")
    visualize_yolo_classification.run(
        model=YOLO(best_weights),
        data_dir=test_dir,
        output_dir=save_dir / "test_inference",
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
    )


if __name__ == "__main__":
    main()
