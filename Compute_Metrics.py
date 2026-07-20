"""
compute_metrics.py

Compute Top-k Recall (R@k), Top-k Group Recall (GR@k), and
Top-1 Discriminative Score (DS@1) for a single output JSON file.

Usage:
    python compute_metrics.py \
        --input  /path/to/output.json \
        --output /path/to/metrics.csv \
        --mode   Embedding   # or Rerank
"""

import argparse
import ast
import json
import os

import pandas as pd


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate inter-table retrieval metrics from a single JSON file."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the input JSON file (output.json)."
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to write the resulting CSV file."
    )
    parser.add_argument(
        "--mode", required=True, choices=["Embedding", "Rerank"],
        help="Retrieval mode: 'Embedding' uses output_text; "
             "'Rerank' uses output_text_rerank."
    )
    parser.add_argument(
        "--max_k", type=int, default=5,
        help="Maximum k for top-k metrics (default: 5)."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Prediction parsing helpers
# ---------------------------------------------------------------------------

def parse_pred_list(raw_text):
    """
    Parse a stringified Python list into a list of (db_id, table_name) tuples.
    Entries that do not contain '::' are stored as (None, None).
    """
    try:
        pred_list = ast.literal_eval(raw_text)
    except (ValueError, SyntaxError):
        return []

    parsed = []
    for p in pred_list:
        if "::" not in p:
            parsed.append((None, None))
        else:
            db_part, table_part = p.split("::", 1)
            parsed.append((db_part, table_part))
    return parsed


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(data, pred_key, max_k=5):
    """
    Compute R@k, GR@k, and DS@1 for each query and aggregate.

    Parameters
    ----------
    data     : list of dict  — loaded JSON records
    pred_key : str           — field name that holds the ranked prediction list
    max_k    : int           — maximum k to evaluate

    Returns
    -------
    per_query : pd.DataFrame with one row per query and columns for each metric
    summary   : dict with averaged metric values
    """
    records = []

    for item in data:
        db_id      = item["db_id"]
        # Ground-truth: the single target table for answerability (R@k)
        target_table = item["highlighted_table"][0]          # e.g. "address_T0"
        # Ground-truth: the full sibling group for coarse relevance (GR@k)
        group_tables = set(item["all_tables_in_group"])      # e.g. {"address_T0", ...}
        group_size   = len(group_tables)

        if group_size == 0:
            continue

        # Parse the ranked prediction list for this mode
        parsed = parse_pred_list(item.get(pred_key, "[]"))

        row = {"db_id": db_id, "target_table": target_table,
               "group_size": group_size}

        # ---- R@k : has the exact target table appeared in top-k? ----
        target_found_at = None   # first rank (1-based) where target appears
        for k in range(1, max_k + 1):
            if k - 1 < len(parsed):
                p_db, p_table = parsed[k - 1]
                if p_db == db_id and p_table == target_table:
                    target_found_at = k
            row[f"R@{k}"] = 1.0 if (target_found_at is not None and
                                      target_found_at <= k) else 0.0

        # ---- GR@k : fraction of the sibling group covered in top-k ----
        # We accumulate hits from the ranked list for each k.
        seen_group_hits = 0
        for k in range(1, max_k + 1):
            if k - 1 < len(parsed):
                p_db, p_table = parsed[k - 1]
                if p_db == db_id and p_table in group_tables:
                    seen_group_hits += 1
            # denominator: min(|G_i|, k)  — as defined in the paper
            denom = min(group_size, k)
            row[f"GR@{k}"] = seen_group_hits / denom

        # ---- DS@1 : R@1 / GR@1 (discriminative ability at rank 1) ----
        gr1 = row["GR@1"]
        row["DS@1"] = (row["R@1"] / gr1) if gr1 > 0 else 0.0

        records.append(row)

    per_query = pd.DataFrame(records)

    # Aggregate: mean over all queries
    metric_cols = (
        [f"R@{k}"  for k in range(1, max_k + 1)] +
        [f"GR@{k}" for k in range(1, max_k + 1)] +
        ["DS@1"]
    )
    summary = {col: per_query[col].mean() for col in metric_cols
               if col in per_query.columns}

    return per_query, summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Validate input file
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # Choose the prediction field based on the selected mode
    pred_key = "output_text" if args.mode == "Embedding" else "output_text_rerank"

    # Load data
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[INFO] Loaded {len(data)} records from {args.input}")
    print(f"[INFO] Mode: {args.mode}  →  using field '{pred_key}'")

    # Compute metrics
    per_query, summary = compute_metrics(data, pred_key, max_k=args.max_k)

    # Print summary to console
    print("\n===== Metric Summary =====")
    for metric, val in summary.items():
        print(f"  {metric:8s}: {val:.4f}")
    print("==========================\n")

    # Build output DataFrame: a single summary row with averaged metrics
    summary_row = {metric: round(val, 6) for metric, val in summary.items()}
    output_df = pd.DataFrame([summary_row])

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    output_df.to_csv(args.output, index=False)
    print(f"[OK] Results written to {args.output}")


if __name__ == "__main__":
    main()