"""Download EU macro series via pandas-datareader + FRED.

VSTOXX is not on FRED; we approximate EU vol with VIX and rely on DGS10 as
a rates proxy (real Bund yields can be added later from the ECB SDW API).
"""
import pandas as pd
from ..config import DATA_RAW, log
from ..bootstrap import ensure

FRED_SERIES = {
    "DGS10": "us_10y",
    "DGS5": "us_5y",
    "DGS1": "us_1y",
    "FEDFUNDS": "fed_funds",
    "TEDRATE": "ted_spread",
    "VIXCLS": "vix",
    "DEXUSEU": "usd_eur",
}

def download(start="2005-01-01", end=None):
    ensure({"pandas_datareader": "pandas-datareader"})
    import pandas_datareader.data as web
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
    log.info(f"fred download {len(FRED_SERIES)} series")
    out = None
    for code, name in FRED_SERIES.items():
        try:
            s = web.DataReader(code, "fred", start, end).rename(columns={code: name})
            out = s if out is None else out.join(s, how="outer")
        except Exception as e:
            log.warning(f"  fred {code} failed: {e}")
    out = out.reset_index().rename(columns={"DATE": "Date"})
    out.to_parquet(DATA_RAW / "macro_fred.parquet", index=False)
    log.info(f"macro: {len(out):,} rows")
    return out

if __name__ == "__main__":
    download()
