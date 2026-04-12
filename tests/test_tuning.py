import os
import pandas as pd
import pytest

from generate_labeled import generate_labeled_data
from tune_threshold import evaluate_threshold, run_sweep
from detector import compute_z_scores

MIN_RECORDS = 5


def test_labeled_data_has_anomalies():
    df = generate_labeled_data(n_clients=10)
    n_anomalies = df["is_anomaly"].sum()
    assert n_anomalies > 0, "No anomalies generated"
    assert n_anomalies < len(df), "All records are anomalies"


def test_labeled_data_columns():
    df = generate_labeled_data(n_clients=5)
    for col in ["client_id", "date", "amount", "is_anomaly"]:
        assert col in df.columns, f"Missing column: {col}"


def test_labeled_data_deterministic():
    df1 = generate_labeled_data(n_clients=10, seed=99)
    df2 = generate_labeled_data(n_clients=10, seed=99)
    pd.testing.assert_frame_equal(df1, df2)


def test_recall_decreases_with_higher_threshold():
    df = generate_labeled_data(n_clients=15)
    scored = compute_z_scores(df, MIN_RECORDS)
    low = evaluate_threshold(scored, threshold=1.5)
    high = evaluate_threshold(scored, threshold=3.5)
    assert low["recall"] >= high["recall"], (
        f"Recall at t=1.5 ({low['recall']}) should be >= recall at t=3.5 ({high['recall']})"
    )


def test_precision_increases_with_higher_threshold():
    df = generate_labeled_data(n_clients=15)
    scored = compute_z_scores(df, MIN_RECORDS)
    low = evaluate_threshold(scored, threshold=1.5)
    high = evaluate_threshold(scored, threshold=3.5)
    assert high["precision"] >= low["precision"], (
        f"Precision at t=3.5 ({high['precision']}) should be >= precision at t=1.5 ({low['precision']})"
    )


def test_sweep_produces_all_thresholds(tmp_path):
    labeled_path = tmp_path / "labeled.csv"
    df = generate_labeled_data(n_clients=10)
    df.to_csv(labeled_path, index=False)
    results = run_sweep(str(labeled_path))
    assert len(results) == 9  # 9 thresholds in THRESHOLDS list
    assert "precision" in results.columns
    assert "recall" in results.columns
    assert "f1" in results.columns
