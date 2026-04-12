"""
Streaming anomaly detector for financial transaction streams.

Processes records one at a time using Welford's online algorithm for
rolling mean/variance, so it never needs to store the full history.
Each record receives a flag decision immediately upon arrival.

Input: newline-delimited JSON (NDJSON) on stdin or from a file.
Each line must have: client_id, date, amount

Usage:
    # pipe from stdin
    cat transactions.ndjson | python stream_detector.py

    # read from file
    python stream_detector.py --input transactions.ndjson

    # adjust threshold and warmup
    python stream_detector.py --input data.ndjson --threshold 3.0 --warmup 10
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, field


@dataclass
class ClientState:
    """Welford's online algorithm for running mean and variance."""
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def z_score(self, value: float) -> float | None:
        if self.count < 2 or self.std == 0:
            return None
        return (value - self.mean) / self.std


class StreamDetector:
    def __init__(self, threshold: float = 2.5, warmup: int = 5):
        self.threshold = threshold
        self.warmup = warmup
        self.clients: dict[str, ClientState] = {}

    def process(self, record: dict) -> dict:
        client_id = record["client_id"]
        amount = float(record["amount"])

        if client_id not in self.clients:
            self.clients[client_id] = ClientState()

        state = self.clients[client_id]

        # compute z-score BEFORE updating (score against prior history)
        z = None
        flagged = False
        severity = None

        if state.count >= self.warmup:
            z = state.z_score(amount)
            if z is not None and abs(z) > self.threshold:
                flagged = True
                severity = "HIGH" if abs(z) > self.threshold * 1.5 else "MEDIUM"

        # update running statistics with new value
        state.update(amount)

        result = {
            "client_id": client_id,
            "date": record.get("date", ""),
            "amount": amount,
            "z_score": round(z, 4) if z is not None else None,
            "flagged": flagged,
            "severity": severity,
            "records_seen": state.count,
        }
        return result


def main():
    parser = argparse.ArgumentParser(description="Streaming anomaly detector.")
    parser.add_argument("--input", default=None, help="NDJSON file (default: stdin)")
    parser.add_argument("--threshold", type=float, default=2.5)
    parser.add_argument("--warmup", type=int, default=5,
                        help="Minimum records per client before scoring (default: 5)")
    args = parser.parse_args()

    detector = StreamDetector(threshold=args.threshold, warmup=args.warmup)

    source = open(args.input) if args.input else sys.stdin
    flagged_count = 0
    total = 0

    try:
        for line in source:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            result = detector.process(record)
            total += 1

            if result["flagged"]:
                flagged_count += 1
                print(json.dumps(result))
    finally:
        if args.input and source is not sys.stdin:
            source.close()

    print(f"\n--- Stream Summary ---", file=sys.stderr)
    print(f"  Total records:  {total}", file=sys.stderr)
    print(f"  Flagged:        {flagged_count}", file=sys.stderr)
    print(f"  Clients:        {len(detector.clients)}", file=sys.stderr)


if __name__ == "__main__":
    main()
