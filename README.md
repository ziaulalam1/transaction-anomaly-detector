# Transaction Anomaly Detector

When working with financial records at Condata, there was a single shared master list to track all records and it was intended to be the source of truth, however, the records were incorrect enough that you could not fully rely upon them for manual reviews. There was no automated step between the raw data and a human reviewer to verify the accuracy of the information. That lack of an automated step has stayed with me; it seemed to be the perfect example of something that should be checked programmatically. This project represents my attempt to work through the design of such an automated process. This project performs z-score analysis on a per-client basis to identify statistically unusual clients -- based upon their individual transaction histories -- to reduce the risk of false positives in terms of identifying anomalies.

**Stack:** Python · pandas · NumPy · openpyxl

---

## How Does It Work

Each client's transaction record history is rated on its own merits. The z score for each transaction is calculated using that client's own mean and standard deviation, not the global mean, so both a small client and a large client will be compared to their own baseline. If clients have less than `--min-records` transactions they will be removed from scoring. If there are less than 5 records for a client then the standard deviation of those records cannot be used reliably as an estimate of the population standard deviation; therefore, flagged records will be categorized into two levels of severity:

| Severity | Condition |
|----------|-----------|
| HIGH | `\|z\|` > threshold × 1.5 |
| MEDIUM | `\|z\|` > threshold |

---

## Quickstart

```bash
pip install -r requirements.txt

# Generate sample data to test with
python generate_sample.py

# Run the detector
python detector.py sample_transactions.csv
```

Output is written to `flagged_transactions.xlsx` by default.

---

## Usage

```bash
python detector.py <input> [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `-o`, `--output` | `flagged_transactions.xlsx` | Output report path |
| `-t`, `--threshold` | `2.5` | Z-score cutoff for flagging |
| `--min-records` | `5` | Minimum transactions per client to include |

**Examples:**

```bash
# Default threshold
python detector.py transactions.csv

# Stricter threshold, custom output
python detector.py transactions.csv -t 3.0 -o review_queue.xlsx

# Lower minimum record requirement
python detector.py transactions.csv --min-records 3
```

---

## Input Format

CSV or Excel file with at minimum these columns (case-insensitive):

| Column | Type | Description |
|--------|------|-------------|
| `client_id` | string | Unique identifier per client |
| `date` | date | Transaction date |
| `amount` | numeric | Transaction amount |

Additional columns are passed through to the output unchanged.

---

## Output

Color-coded Excel report:

- **Red rows** -- HIGH severity (z-score > threshold x 1.5)
- **Yellow rows** -- MEDIUM severity (z-score > threshold)
- Sorted by absolute z-score descending so the most anomalous records are at the top
- Includes a `z_score` and `severity` column alongside the original data

---

## Trade-offs

**Per Client Grouping vs Global Threshold**
Using a global z score would result in a $50,000 transaction being flagged for both a small client and a large client, which would produce irrelevant results. By using a per client group by client_id each client will be compared to their own history.

**Why Use Z-Score Instead Of IQR Or Isolation Forest**
Z-Score can be interpreted and explained to a non-technical reviewer at the data sizes commonly found in accounting work flows (10s to 100s of transactions per client). While isolation forest would provide a better solution at larger scales or when there are multiple input variables it would also add additional complexity that is not warranted in this use case.

**Default Threshold Value of 2.5**
While a z-score of 3.0 is the generally accepted value in academic research it is often too conservative and misses too many cases that are later determined to be valid exceptions. On the other hand a z-score of 2.5 may produce a greater number of flags but the high/medium flag designation allows reviewers to triage flags versus treating all flags as equal.
