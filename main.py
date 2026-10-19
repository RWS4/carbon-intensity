import argparse
import asyncio
from datetime import datetime, timezone

import polars as pl

import analysis
from gridcog_carbontest import DATA_DIR, REGION_IDS, fetch_range


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)

#TODO: move this to own module
def combine_parquet_files(output_name: str = "combined.parquet") -> pl.DataFrame:
    files = sorted(DATA_DIR.glob("clean_slice_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no clean_slice_*.parquet files in {DATA_DIR}")

    df = (
        pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
        .unique(subset=["regionid", "interval_start"], keep="first")
        .sort(["regionid", "interval_start"])
    )
    df.write_parquet(DATA_DIR / output_name)
    return df

#NOTE: primitive CLI functionality
#TODO: add "cleanup" option to get rid of intermediate files
#TODO: analysis options are still hardcoded, change to accept user inputs
def main():
    parser = argparse.ArgumentParser(description="Download regional carbon intensity data in 14-day windows.")
    parser.add_argument("--start", type=parse_date, default="2025-01-16", help="start date, YYYY-MM-DD (default: 2025-01-16)")
    parser.add_argument("--end", type=parse_date, default="2026-10-16", help="end date, YYYY-MM-DD (default: 2026-10-16)")
    parser.add_argument("--region", type=int, nargs="+", default=list(REGION_IDS), help="one or more region ids (default: all, 1-17)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--combine", action="store_true", help="combine the parquet files already in data/ into combined.parquet instead of downloading")
    mode.add_argument("--analyse", action="store_true", help="train the model on data/combined.parquet and save figures instead of downloading")
    args = parser.parse_args()

    if args.combine:
        combine_parquet_files()
    elif args.analyse:
        analysis.run()
    else:
        asyncio.run(fetch_range(args.start, args.end, args.region))


if __name__ == "__main__":
    main()
