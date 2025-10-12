import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_generate_sample_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent.parent / "generate_sample.py")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert (tmp_path / "sample_transactions.csv").exists()


def test_generate_sample_columns(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(
        [sys.executable, str(Path(__file__).parent.parent / "generate_sample.py")],
        capture_output=True,
    )
    df = pd.read_csv(tmp_path / "sample_transactions.csv")
    assert {"client_id", "date", "amount", "description"}.issubset(df.columns)
    assert (df["amount"] >= 0).all()


def test_generate_sample_no_negative_amounts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(
        [sys.executable, str(Path(__file__).parent.parent / "generate_sample.py")],
        capture_output=True,
    )
    df = pd.read_csv(tmp_path / "sample_transactions.csv")
    assert (df["amount"] >= 0).all()
