#!/usr/bin/env python3
"""Select the N SAR frames with the most eddies in the Western‑Med Excel sheet.

Usage
-----
python scripts/00_select_top_files.py \
    --excel $SCRATCH/WESTMEDEddies_Gade_engl_orig.xlsx \
    --top 2 \
    --out cfg/preprocess_sar/selected_files.json
"""
import argparse
import json

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--excel", type=str, default="data/WESTMEDEddies_Gade_engl_orig.xlsx"
    )
    p.add_argument("--top", type=int, default=2)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    df = pd.read_excel(args.excel)
    counts = df["File Name"].value_counts().head(args.top).to_dict()
    print("Top files (eddy count):")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    with open(args.out, "w") as fp:
        json.dump({"files": list(counts.keys())}, fp, indent=2)


if __name__ == "__main__":
    main()
