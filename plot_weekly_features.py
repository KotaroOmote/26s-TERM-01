#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features-csv",
        default="cache/fujisawa_demo/derived/features_weekly.csv",
        help="Path to features_weekly.csv",
    )
    parser.add_argument(
        "--save",
        default="",
        help="Optional output image path. If omitted, show interactive plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feat = pd.read_csv(Path(args.features_csv))
    feat["week_start"] = pd.to_datetime(feat["week_start"])

    num_cols = [c for c in feat.columns if c != "week_start"]
    z = feat[num_cols].copy()
    z = (z - z.mean()) / z.std().replace(0, 1)

    plt.figure(figsize=(14, 5))
    for c in num_cols:
        plt.plot(feat["week_start"], z[c], label=c, alpha=0.9)
    plt.title("Weekly Features (Z-score normalized)")
    plt.xlabel("Week")
    plt.ylabel("Z-score")
    plt.grid(alpha=0.3)
    plt.legend(ncol=3, fontsize=9)

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[ok] saved: {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
