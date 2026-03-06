"""
Generates a synthetic transaction dataset with seeded outliers for testing purposes.
"""
import random
from datetime import date, timedelta

import pandas as pd
import numpy as np

CLIENTS = [f"CLIENT_{i:03d}" for i in range(1, 21)]
START_DATE = date(2024, 1, 1)
DAYS = 365
OUTLIER_RATE = 0.03

random.seed(42)
np.random.seed(42)

rows = []
for client in CLIENTS:
    n_transactions = random.randint(10, 60)
    base_amount = random.uniform(500, 10000)
    std = base_amount * 0.15

    for _ in range(n_transactions):
        tx_date = START_DATE + timedelta(days=random.randint(0, DAYS))
        amount = round(np.random.normal(base_amount, std), 2)

        # seed outliers
        if random.random() < OUTLIER_RATE:
            amount = round(amount * random.choice([4.5, 5.5, -3.0]), 2)

        rows.append({
            "client_id": client,
            "date": tx_date,
            "amount": abs(amount),
            "description": random.choice([
                "Invoice payment", "Retainer fee", "Consulting",
                "Equipment", "Software license", "Misc expense"
            ])
        })

df = pd.DataFrame(rows).sort_values(["client_id", "date"]).reset_index(drop=True)
df.to_csv("sample_transactions.csv", index=False)
print(f"Generated {len(df)} transactions for {len(CLIENTS)} clients.")
print(f"Approximate outliers seeded: {int(len(df) * OUTLIER_RATE)}")
