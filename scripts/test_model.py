import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv


def parse_args():
    parser = argparse.ArgumentParser(description="Roboflow hosted model test scripti")
    parser.add_argument("--image", required=True, help="Test gorseli")
    parser.add_argument("--model-id", default="xview2-xbd/2", help="Ornek: xview2-xbd/2")
    parser.add_argument("--output", default="test_result.json")
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Gorsel bulunamadi: {image_path}")

    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY .env dosyasinda olmali.")

    from inference_sdk import InferenceHTTPClient

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key,
    )
    result = client.infer(str(image_path), model_id=args.model_id)

    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    print(f"Test sonucu kaydedildi: {args.output}")


if __name__ == "__main__":
    main()
