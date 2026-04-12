"""
Generate a labeled transaction dataset with known anomalies for threshold tuning.

Each transaction has an `is_anomaly` ground-truth column. Anomalies are injected
by multiplying the client's normal amount by a large factor, guaranteeing the
z-score will exceed any reasonable threshold. Normal transactions are drawn from
the client's baseline distribution and labeled False.

Usage:
    python generate_labeled.py [--clients 10] [--output labeled_transactions.csv]
"""

import argparse
import random
from datetime import date, timedelta

import numpy as np
import pandas as pd

DEFAULT_CLIENTS = 20
DEFAULT_OUTPUT = "labeled_transactions.csv"
SEED = 42
START_DATE = date(2024, 1, 1)
DAYS = 365


def generate_labeled_data(n_clients: int, seed: int = SEED) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    clients = [f"CLIENT_{i:03d}" for i in range(1, n_clients + 1)]
    rows = []

    for client in clients:
        n_normal = random.randint(30, 60)
        base_amount = random.uniform(1000, 20000)
        std = base_amount * random.uniform(0.12, 0.25)

        # normal transactions
        for _ in range(n_normal):
            tx_date = START_DATE + timedelta(days=random.randint(0, DAYS))
            amount = round(abs(np.random.normal(base_amount, std)), 2)
            rows.append({
                "client_id": client,
                "date": tx_date,
                "amount": amount,
                "is_anomaly": False,
            })

        # legitimate large transactions (equipment purchases, year-end
        # settlements, retainer deposits) -- NOT anomalies, but the z-score
        # detector will flag them at low thresholds. These are the false
        # positives a reviewer would clear.
        n_legit_large = random.randint(0, 2)
        for _ in range(n_legit_large):
            tx_date = START_DATE + timedelta(days=random.randint(0, DAYS))
            amount = round(base_amount * random.uniform(2.5, 4.0), 2)
            rows.append({
                "client_id": client,
                "date": tx_date,
                "amount": amount,
                "is_anomaly": False,
            })

        # inject anomalies at varying severity:
        #   - "clear" anomalies: 3.5-5x multiplier (should always be caught)
        #   - "borderline" anomalies: 2.0-3.0x multiplier (caught at low threshold,
        #     missed at high threshold -- this is what creates the recall tradeoff)
        n_clear = random.randint(1, 3)
        n_borderline = random.randint(1, 3)

        for _ in range(n_clear):
            tx_date = START_DATE + timedelta(days=random.randint(0, DAYS))
            multiplier = random.uniform(3.5, 5.0)
            amount = round(abs(np.random.normal(base_amount, std)) * multiplier, 2)
            rows.append({
                "client_id": client,
                "date": tx_date,
                "amount": amount,
                "is_anomaly": True,
            })

        for _ in range(n_borderline):
            tx_date = START_DATE + timedelta(days=random.randint(0, DAYS))
            multiplier = random.uniform(2.0, 3.0)
            amount = round(abs(np.random.normal(base_amount, std)) * multiplier, 2)
            rows.append({
                "client_id": client,
                "date": tx_date,
                "amount": amount,
                "is_anomaly": True,
            })

    df = pd.DataFrame(rows).sort_values(["client_id", "date"]).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate labeled transaction data for threshold tuning.")
    parser.add_argument("--clients", type=int, default=DEFAULT_CLIENTS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    df = generate_labeled_data(args.clients)
    df.to_csv(args.output, index=False)

    n_anomalies = df["is_anomaly"].sum()
    print(f"Generated {len(df)} transactions for {args.clients} clients.")
    print(f"Anomalies: {n_anomalies} ({n_anomalies / len(df) * 100:.1f}%)")


if __name__ == "__main__":
    main()
