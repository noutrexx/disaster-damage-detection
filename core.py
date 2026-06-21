"""Streamlit'ten bagimsiz, saf analiz yardimcilari.

Bu modul UI veya disa baglilik icermez; bu sayede `tests/` altinda
dogrudan ve hizlica test edilebilir.
"""

import csv
import io
import random

DAMAGE_COLORS = {
    "destroyed": "#d90429",
    "major-damage": "#f77f00",
    "minor-damage": "#fcbf49",
    "no-damage": "#2a9d8f",
}

# Hasar agirliklari: siddet endeksi hesaplamak icin kullanilir.
DAMAGE_WEIGHTS = {
    "destroyed": 3,
    "major-damage": 2,
    "minor-damage": 1,
    "no-damage": 0,
}

DAMAGE_LABELS_TR = {
    "destroyed": "Yikilmis",
    "major-damage": "Agir hasar",
    "minor-damage": "Hafif hasar",
    "no-damage": "Hasarsiz",
}


def create_demo_predictions(width: int, height: int) -> list[dict]:
    """API key olmadan arayuzu gostermek icin sahte ama sabit tahmin uretir."""
    random.seed(width * 1000 + height)
    classes = list(DAMAGE_COLORS)
    predictions = []

    for index in range(8):
        box_w = random.randint(max(35, width // 14), max(45, width // 7))
        box_h = random.randint(max(35, height // 14), max(45, height // 7))
        x = random.randint(box_w // 2, max(box_w // 2, width - box_w // 2))
        y = random.randint(box_h // 2, max(box_h // 2, height - box_h // 2))
        class_name = classes[index % len(classes)]

        predictions.append(
            {
                "class": class_name,
                "confidence": round(random.uniform(0.55, 0.93), 2),
                "x": x,
                "y": y,
                "width": box_w,
                "height": box_h,
            }
        )

    return predictions


def extract_predictions(result) -> list[dict]:
    """Roboflow yanitindaki ic ice yapidan tahmin listesini cikarir."""
    if isinstance(result, dict) and "predictions" in result:
        return result["predictions"]

    if isinstance(result, list):
        for item in result:
            predictions = extract_predictions(item)
            if predictions:
                return predictions

    if isinstance(result, dict):
        for value in result.values():
            predictions = extract_predictions(value)
            if predictions:
                return predictions

    return []


def count_by_class(predictions: list[dict]) -> dict[str, int]:
    counts = {name: 0 for name in DAMAGE_COLORS}
    for prediction in predictions:
        class_name = prediction.get("class")
        if class_name in counts:
            counts[class_name] += 1
    return counts


def average_confidence(predictions: list[dict]) -> float:
    if not predictions:
        return 0.0
    total = sum(float(item.get("confidence", 0)) for item in predictions)
    return total / len(predictions)


def severity_index(counts: dict[str, int]) -> float:
    """0-100 arasi basit hasar siddet endeksi (yikilmis bolgeler daha agir basar)."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    weighted = sum(DAMAGE_WEIGHTS.get(name, 0) * value for name, value in counts.items())
    max_weight = max(DAMAGE_WEIGHTS.values()) or 1
    return round(100 * weighted / (total * max_weight), 1)


def severity_label(score: float) -> tuple[str, str]:
    """Siddet endeksini etiket ve renge cevirir."""
    if score >= 66:
        return "Kritik", "#d90429"
    if score >= 33:
        return "Yuksek", "#f77f00"
    if score > 0:
        return "Dusuk", "#fcbf49"
    return "Hasar yok", "#2a9d8f"


def predictions_to_csv(predictions: list[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["class", "confidence", "x", "y", "width", "height"])
    for item in predictions:
        writer.writerow(
            [
                item.get("class", ""),
                item.get("confidence", ""),
                item.get("x", ""),
                item.get("y", ""),
                item.get("width", ""),
                item.get("height", ""),
            ]
        )
    return buffer.getvalue()
