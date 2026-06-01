import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="xView2/xBD YOLO egitim scripti")
    parser.add_argument("--data", required=True, help="Roboflow data.yaml dosya yolu")
    parser.add_argument("--model", default="yolov8n.pt", help="Baslangic YOLO modeli")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="oahts_train")
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset yaml bulunamadi: {data_path}")

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
