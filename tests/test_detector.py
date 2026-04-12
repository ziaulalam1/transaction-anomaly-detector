import pandas as pd

from detector import compute_z_scores, flag_outliers


def _df(rows):
    return pd.DataFrame(rows, columns=["client_id", "date", "amount"])


def test_z_score_outlier_flagged():
    rows = [("A", "2025-01-01", 100)] * 9 + [("A", "2025-01-05", 900)]
    df = _df(rows)
    result = compute_z_scores(df, min_records=5)
    assert result.loc[result["amount"] == 900, "z_score"].values[0] > 2.0


def test_min_records_excludes_client():
    rows = [("A", "2025-01-01", 100), ("A", "2025-01-02", 200)]
    df = _df(rows)
    result = compute_z_scores(df, min_records=5)
    assert result["z_score"].isna().all()


def test_flag_outliers_high_severity():
    df = pd.DataFrame({
        "client_id": ["A"],
        "amount": [500.0],
        "z_score": [4.5],
    })
    flagged = flag_outliers(df, threshold=2.5)
    assert len(flagged) == 1
    assert flagged.iloc[0]["severity"] == "HIGH"


def test_flag_outliers_medium_severity():
    df = pd.DataFrame({
        "client_id": ["A"],
        "amount": [200.0],
        "z_score": [3.0],
    })
    flagged = flag_outliers(df, threshold=2.5)
    assert flagged.iloc[0]["severity"] == "MEDIUM"


def test_no_flags_below_threshold():
    df = pd.DataFrame({
        "client_id": ["A"],
        "amount": [100.0],
        "z_score": [1.5],
    })
    flagged = flag_outliers(df, threshold=2.5)
    assert len(flagged) == 0


def test_per_client_grouping():
    rows = (
        [("small", f"2025-01-{i:02d}", 100) for i in range(1, 11)]
        + [("small", "2025-01-15", 5000)]
        + [("large", f"2025-01-{i:02d}", 50000 + i * 100) for i in range(1, 11)]
        + [("large", "2025-01-15", 51500)]
    )
    df = _df(rows)
    result = compute_z_scores(df, min_records=5)
    flagged = flag_outliers(result, threshold=2.5)
    client_ids = set(flagged["client_id"].tolist())
    assert "small" in client_ids
    assert "large" not in client_ids


def test_z_score_column_present():
    rows = [("A", "2025-01-01", 100)] * 5
    df = _df(rows)
    result = compute_z_scores(df, min_records=5)
    assert "z_score" in result.columns
