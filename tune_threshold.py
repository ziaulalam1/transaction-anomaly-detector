"""
Sweep z-score thresholds against labeled data to find the operating point
that balances precision and recall.

Reads a labeled CSV (with `is_anomaly` column), runs the detector at each
threshold, and outputs:
  - reports/threshold_tuning.csv   (precision, recall, F1 per threshold)
  - reports/precision_recall.png   (visualization)

Usage:
    python tune_threshold.py [--input labeled_transactions.csv]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from detector import compute_z_scores, flag_outliers

THRESHOLDS = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0]
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
DEFAULT_INPUT = "labeled_transactions.csv"
MIN_RECORDS = 5


def evaluate_threshold(scored_df: pd.DataFrame, threshold: float, label_col: str = "is_anomaly") -> dict:
    flagged = flag_outliers(scored_df, threshold)
    flagged_indices = set(flagged.index)

    # only evaluate clients that had enough records to be scored
    scorable = scored_df[scored_df["z_score"].notna()]
    actual_positive = set(scorable[scorable[label_col]].index)
    actual_negative = set(scorable[~scorable[label_col]].index)

    tp = len(flagged_indices & actual_positive)
    fp = len(flagged_indices & actual_negative)
    fn = len(actual_positive - flagged_indices)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def run_sweep(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path, parse_dates=["date"])
    df["is_anomaly"] = df["is_anomaly"].astype(bool)

    scored = compute_z_scores(df, MIN_RECORDS)

    results = []
    for t in THRESHOLDS:
        results.append(evaluate_threshold(scored, t))

    return pd.DataFrame(results)


def write_reports(results_df: pd.DataFrame) -> None:
    os.makedirs(REPORTS_DIR, exist_ok=True)

    csv_path = os.path.join(REPORTS_DIR, "threshold_tuning.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    # chart
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results_df["threshold"], results_df["precision"], "o-", color="#1a73e8", label="Precision", linewidth=2)
    ax.plot(results_df["threshold"], results_df["recall"], "s-", color="#d93025", label="Recall", linewidth=2)
    ax.plot(results_df["threshold"], results_df["f1"], "^-", color="#0d652d", label="F1", linewidth=2)

    # mark the chosen operating point
    best_idx = results_df["f1"].idxmax()
    best_t = results_df.loc[best_idx, "threshold"]
    best_f1 = results_df.loc[best_idx, "f1"]
    ax.axvline(best_t, color="#f29900", linestyle="--", linewidth=1.5, alpha=0.7,
               label=f"Best F1 = {best_f1:.2f} @ t={best_t}")

    ax.set_xlabel("Z-Score Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Tuning: Precision / Recall / F1")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="center left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    png_path = os.path.join(REPORTS_DIR, "precision_recall.png")
    fig.savefig(png_path, dpi=150)
    print(f"Wrote {png_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    args = parser.parse_args()

    results_df = run_sweep(args.input)

    print("\n--- Threshold Sweep ---")
    print(results_df.to_string(index=False))

    best_idx = results_df["f1"].idxmax()
    best = results_df.loc[best_idx]
    print(f"\nBest F1: {best['f1']} at threshold={best['threshold']} "
          f"(precision={best['precision']}, recall={best['recall']})")

    write_reports(results_df)


if __name__ == "__main__":
    main()
