import json
import subprocess
import sys

from stream_detector import StreamDetector, ClientState


def test_welford_mean_and_std():
    state = ClientState()
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    for v in values:
        state.update(v)
    assert abs(state.mean - 30.0) < 1e-10
    assert abs(state.std - 15.8114) < 0.001


def test_welford_z_score():
    state = ClientState()
    for v in [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]:
        state.update(v)
    z = state.z_score(500.0)
    assert z is None  # std is 0 for identical values


def test_warmup_skips_early_records():
    detector = StreamDetector(threshold=2.5, warmup=5)
    record = {"client_id": "A", "date": "2025-01-01", "amount": 99999}
    result = detector.process(record)
    assert result["flagged"] is False
    assert result["z_score"] is None


def test_flags_anomaly_after_warmup():
    detector = StreamDetector(threshold=2.5, warmup=5)
    for i in range(10):
        detector.process({"client_id": "A", "date": f"2025-01-{i+1:02d}", "amount": 100 + i})
    result = detector.process({"client_id": "A", "date": "2025-02-01", "amount": 5000})
    assert result["flagged"] is True
    assert result["severity"] in ("HIGH", "MEDIUM")


def test_per_client_isolation():
    detector = StreamDetector(threshold=2.5, warmup=5)
    # client A: small transactions with slight variance
    for i in range(10):
        detector.process({"client_id": "A", "date": f"2025-01-{i+1:02d}", "amount": 100 + i})
    # client B: large transactions with slight variance
    for i in range(10):
        detector.process({"client_id": "B", "date": f"2025-01-{i+1:02d}", "amount": 50000 + i * 100})
    # 5000 is anomalous for A but normal for B
    result_a = detector.process({"client_id": "A", "date": "2025-02-01", "amount": 5000})
    result_b = detector.process({"client_id": "B", "date": "2025-02-01", "amount": 50500})
    assert result_a["flagged"] is True
    assert result_b["flagged"] is False


def test_stream_batch_agreement():
    """Streaming results should flag the same records as batch mode for equivalent data."""
    from detector import compute_z_scores, flag_outliers
    import pandas as pd

    records = []
    for i in range(30):
        records.append({"client_id": "X", "date": f"2025-01-{(i % 28) + 1:02d}", "amount": 100 + i})
    records.append({"client_id": "X", "date": "2025-02-01", "amount": 5000})

    # streaming
    detector = StreamDetector(threshold=2.5, warmup=5)
    stream_flagged = []
    for r in records:
        result = detector.process(r)
        if result["flagged"]:
            stream_flagged.append(result["amount"])

    # batch
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = df["amount"].astype(float)
    scored = compute_z_scores(df, min_records=5)
    batch_flagged = flag_outliers(scored, threshold=2.5)
    batch_amounts = sorted(batch_flagged["amount"].tolist())

    # the extreme outlier (5000) should be flagged by both
    assert 5000.0 in stream_flagged
    assert 5000.0 in batch_amounts
