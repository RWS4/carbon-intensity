# carbon-intensity

A prototype pipeline that downloads half-hourly regional carbon intensity data for Great Britain from the [Carbon Intensity API](https://carbonintensity.org.uk/), combines it into a single dataset, and fits an XGBoost regression model to predict carbon intensity by region, day of year and time of day.

## Requirements

- Python 3.14
- [uv](https://docs.astral.sh/uv/)

Install dependencies:

```bash
uv sync
```

## Usage

Everything runs through `main.py`, which has three modes. Run them in this order.

### 1. Download

```bash
uv run main.py
```

Fetches data for each region in 14-day windows (requests run concurrently) and writes one parquet file per region and window to `data/`, e.g. `data/clean_slice_r13_20250116_20250130.parquet`. When finished it reports any failed windows and checks that every expected file is present.

Options:

| Option | Default | Description |
|---|---|---|
| `--start YYYY-MM-DD` | `2025-01-16` | Start date (UTC) |
| `--end YYYY-MM-DD` | `2026-10-16` | End date (UTC) |
| `--region ID [ID ...]` | all (1–17) | One or more region ids |

```bash
uv run main.py --start 2026-01-01 --end 2026-03-01 --region 1 13
```

### 2. Combine

```bash
uv run main.py --combine
```

Merges all `data/clean_slice_*.parquet` files into `data/combined.parquet`, removing duplicate intervals within each region.

### 3. Analyse

```bash
uv run main.py --analyse
```

Trains the model on `data/combined.parquet` (run from the project folder), prints predictions and the test RMSE, and saves figures to `figures/`:

- `day_180.png`: predicted carbon intensity across the half-hours of day 180
- `year_tod_24.png`: predicted carbon intensity across the year at half-hour slot 24 (11:30 UTC)

Both plots compare regions 1 (North Scotland) and 13 (London).

## Project layout

| File | Purpose |
|---|---|
| `main.py` | Command-line entry point; also combines the downloaded files |
| `gridcog_carbontest.py` | Async API download, cleaning and per-window parquet output |
| `analysis.py` | Feature engineering, XGBoost training and plotting |

## Limitations

This is a prototype. Time of day is encoded as an integer slot (1–48, UTC), so the model doesn't know that 23:30 is next to 00:00. The train/test split is random rather than time-based, so the reported RMSE is optimistic. See the notes in the source files for planned improvements.
