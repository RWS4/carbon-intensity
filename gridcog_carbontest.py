import polars as pl
import asyncio
import httpx
import json
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Iterable
import matplotlib.pyplot as mpl

#TODO: move this to a workflow config
WINDOW = timedelta(days=14)
DT_FMT = "%Y-%m-%dT%H:%MZ"
MAX_CONCURRENT = 5
DATA_DIR = Path(__file__).parent / "data"
REGION_IDS = range(1, 18)

#NOTE: I probably bit off more than I could chew trying to make this asynchronous. It is pretty slow for the volume of data, and I suspect that either the time windows are too small or the generators are waiting in the wrong step. Could do a simple diagnosis with a combination of Claude/verbose output/timeits but I didn't want to risk refactoring halfway through.
async def fetch_window(client: httpx.AsyncClient, from_dt: datetime, to_dt: datetime, region_id: int) -> pl.DataFrame:
    #NOTE: this was what I initially came up with, without AI, just trying to get my head around the API. My thinking was that having polars handle the unpacking as a vector operation would be quicker than validating with pydantic then loading into the frame. This turned out to be the incorrect approach due to the return limits on the API, which negated the potential benefits of the vector operations.
    url = f"https://api.carbonintensity.org.uk/regional/intensity/{from_dt.strftime(DT_FMT)}/{to_dt.strftime(DT_FMT)}/regionid/{region_id}"
    headers = {
      'Accept': 'application/json'
    }

    r = await client.get(url, headers = headers)
    r.raise_for_status()
    return ( pl.read_json(StringIO(r.text))
            .unnest("data")
            .explode("data", empty_as_null=True)
            .unnest("data")
            .drop(["dnoregion","shortname","to"])
            .unnest("intensity")
            .with_columns(pl.col("from").str.to_datetime(format=DT_FMT, time_zone='UTC').alias("interval_start"))
            .drop(["from","index"])
            .explode("generationmix", empty_as_null=True)
            .unnest("generationmix")
            .pivot(on="fuel", values="perc")
          )

#NOTE: handles creation of output files paths.
#TODO: detangle data checking/file names conceptually.
def slice_path(region_id: int, from_dt: datetime, to_dt: datetime) -> Path:
    return DATA_DIR / f"clean_slice_r{region_id:02d}_{from_dt:%Y%m%d}_{to_dt:%Y%m%d}.parquet"

#NOTE: this functions handles the file write operation of the slices
async def fetch_and_write(client: httpx.AsyncClient, sem: asyncio.Semaphore, from_dt: datetime, to_dt: datetime, region_id: int):
    async with sem:
        df = await fetch_window(client, from_dt, to_dt, region_id)
    print(f"region {region_id:>2}, {from_dt:%Y-%m-%d} - {to_dt:%Y-%m-%d}: {df.height} rows")

    df.write_parquet(slice_path(region_id, from_dt, to_dt))

#NOTE: entry point from main
async def fetch_range(start: datetime, end: datetime, region_ids: Iterable[int] = REGION_IDS):
    windows = []
    window_start = start
    while window_start < end:
        window_end = min(window_start + WINDOW, end)
        windows.append((window_start, window_end))
        window_start = window_end

    jobs = [(region_id, f, t) for region_id in region_ids for f, t in windows]
    
    DATA_DIR.mkdir(exist_ok=True)
    #NOTE:creates a manifest of expected date range slices. Not adequate in long run. Should be partially baked into the data model, but at a minimum this functionality should be moved to separate checking module to read through resultant dataset and assess for gaps/missing data. This could then create a feedback loop to requery the API at relevant sections.
    manifest = [slice_path(r, f, t) for r, f, t in jobs]
    print(f"manifest: {len(manifest)} files expected")
    #TODO: play around with MAX_CONCURRENT to see if that speeds things up
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    async with httpx.AsyncClient(timeout=30) as client:
        results = await asyncio.gather(
            *(fetch_and_write(client, sem, f, t, r) for r, f, t in jobs),
            return_exceptions=True,
        )
    #NOTE: checks results of manifest of expected slices
    for (r, f, t), result in zip(jobs, results):
        if isinstance(result, Exception):
            print(f"FAILED region {r:>2}, {f:%Y-%m-%d} - {t:%Y-%m-%d}: {result!r}")

    missing = [path for path in manifest if not path.exists()]
    print(f"manifest check: {len(manifest) - len(missing)}/{len(manifest)} files present")
    for path in missing:
        print(f"  missing: {path.name}")

#NOTE: test function unused in prod
def test_data_get():
    start = datetime(2025, 1, 16, tzinfo=timezone.utc)
    end = datetime(2026, 10, 16, tzinfo=timezone.utc)

    asyncio.run(fetch_range(start, end, [13]))
