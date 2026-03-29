#!/usr/bin/env python3
"""Build a refined final submission from top-ranked breakthrough candidates.

This script blends the continuous TargetRAUC scores from the top files listed in
FINAL_BT_RANKING.txt, then derives TargetF1 using a quantile threshold that
matches a target positive rate.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_ranked_filenames(ranking_path: Path, top_k: int) -> list[str]:
    files: list[str] = []
    for line in ranking_path.read_text(encoding="utf-8").splitlines():
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        if not re.match(r"^\d+$", parts[0]):
            continue
        filename = parts[1]
        if filename.lower().endswith(".csv"):
            files.append(filename)
        if len(files) >= top_k:
            break
    if not files:
        raise ValueError(f"No ranked CSV files found in {ranking_path}")
    return files


def rank_normalize(values: pd.Series) -> pd.Series:
    # Rank-normalization makes scales comparable across different source files.
    return values.rank(method="average", pct=True)


def build_refined_submission(
    data_dir: Path,
    ranking_file: str,
    top_k: int,
    decay: float,
    target_pos_rate: float,
    output_name: str,
) -> tuple[Path, float]:
    ranking_path = data_dir / ranking_file
    ranked_files = parse_ranked_filenames(ranking_path, top_k)

    base = pd.read_csv(data_dir / ranked_files[0])[["ID"]].copy()
    blend_rank = np.zeros(len(base), dtype=float)
    blend_raw = np.zeros(len(base), dtype=float)

    weights = np.array([math.exp(-decay * i) for i in range(len(ranked_files))], dtype=float)
    weights = weights / weights.sum()

    for idx, filename in enumerate(ranked_files):
        cur = pd.read_csv(data_dir / filename)[["ID", "TargetRAUC"]]
        merged = base.merge(cur, on="ID", how="left", validate="one_to_one")
        if merged["TargetRAUC"].isna().any():
            raise ValueError(f"Missing IDs after merging {filename}")

        raw = merged["TargetRAUC"].astype(float)
        blend_raw += weights[idx] * raw.values
        blend_rank += weights[idx] * rank_normalize(raw).values

    # Combine raw and rank blend for both calibration and robust ordering.
    refined_rauc = 0.55 * blend_rank + 0.45 * rank_normalize(pd.Series(blend_raw))

    threshold = float(np.quantile(refined_rauc, 1.0 - target_pos_rate, method="higher"))
    target_f1 = (refined_rauc >= threshold).astype(int)

    submission = pd.DataFrame(
        {
            "ID": base["ID"],
            "TargetF1": target_f1,
            "TargetRAUC": refined_rauc,
        }
    )

    out_path = data_dir / output_name
    submission.to_csv(out_path, index=False)

    achieved_pos_rate = float(submission["TargetF1"].mean())
    return out_path, achieved_pos_rate


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine a final submission from ranked candidates")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--ranking-file", default="FINAL_BT_RANKING.txt")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--decay", type=float, default=0.18)
    parser.add_argument("--target-pos-rate", type=float, default=0.7553)
    parser.add_argument("--output-name", default="final_refined_lastchance.csv")
    args = parser.parse_args()

    out_path, pos_rate = build_refined_submission(
        data_dir=args.data_dir,
        ranking_file=args.ranking_file,
        top_k=args.top_k,
        decay=args.decay,
        target_pos_rate=args.target_pos_rate,
        output_name=args.output_name,
    )

    print(f"Saved: {out_path}")
    print(f"Achieved positive rate: {pos_rate:.4f}")


if __name__ == "__main__":
    main()
