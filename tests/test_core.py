"""core.py saf analiz fonksiyonlari icin birim testleri."""

import csv
import io

import pytest

from core import (
    average_confidence,
    count_by_class,
    extract_predictions,
    predictions_to_csv,
    severity_index,
    severity_label,
)


def make(class_name, confidence=0.9):
    return {"class": class_name, "confidence": confidence, "x": 1, "y": 2, "width": 3, "height": 4}


def test_extract_predictions_direct_dict():
    result = {"predictions": [make("destroyed")]}
    assert extract_predictions(result) == [make("destroyed")]


def test_extract_predictions_nested_in_list():
    result = [{"foo": 1}, {"predictions": [make("no-damage")]}]
    assert extract_predictions(result) == [make("no-damage")]


def test_extract_predictions_deeply_nested():
    result = {"outputs": {"model": {"predictions": [make("minor-damage")]}}}
    assert extract_predictions(result) == [make("minor-damage")]


def test_extract_predictions_missing_returns_empty():
    assert extract_predictions({"foo": "bar"}) == []
    assert extract_predictions([]) == []
    assert extract_predictions("beklenmeyen") == []


def test_count_by_class_counts_known_classes():
    preds = [make("destroyed"), make("destroyed"), make("no-damage")]
    counts = count_by_class(preds)
    assert counts == {
        "destroyed": 2,
        "major-damage": 0,
        "minor-damage": 0,
        "no-damage": 1,
    }


def test_count_by_class_ignores_unknown_class():
    counts = count_by_class([make("destroyed"), make("bilinmeyen")])
    assert counts["destroyed"] == 1
    assert sum(counts.values()) == 1


def test_count_by_class_empty():
    assert sum(count_by_class([]).values()) == 0


def test_average_confidence_empty_is_zero():
    assert average_confidence([]) == 0.0


def test_average_confidence_mean():
    preds = [make("destroyed", 0.4), make("no-damage", 0.6)]
    assert average_confidence(preds) == pytest.approx(0.5)


def test_severity_index_all_destroyed_is_100():
    counts = count_by_class([make("destroyed")])
    assert severity_index(counts) == 100.0


def test_severity_index_all_no_damage_is_zero():
    counts = count_by_class([make("no-damage"), make("no-damage")])
    assert severity_index(counts) == 0.0


def test_severity_index_empty_is_zero():
    assert severity_index(count_by_class([])) == 0.0


def test_severity_index_mixed_is_between():
    counts = count_by_class([make("destroyed"), make("no-damage")])
    score = severity_index(counts)
    assert 0 < score < 100


@pytest.mark.parametrize(
    "score,label",
    [(0, "Hasar yok"), (10, "Dusuk"), (33, "Yuksek"), (66, "Kritik"), (100, "Kritik")],
)
def test_severity_label_thresholds(score, label):
    assert severity_label(score)[0] == label


def test_predictions_to_csv_header_and_rows():
    preds = [make("destroyed", 0.7), make("minor-damage", 0.5)]
    text = predictions_to_csv(preds)
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == ["class", "confidence", "x", "y", "width", "height"]
    assert len(rows) == 3
    assert rows[1][0] == "destroyed"


def test_predictions_to_csv_empty_has_header_only():
    rows = list(csv.reader(io.StringIO(predictions_to_csv([]))))
    assert len(rows) == 1
