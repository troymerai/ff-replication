"""
ff_data_us.py — Ken French US data reader (US only)

What it does:
  1) Extracts only the 'monthly' section from Ken French zip/CSV files
       - 3-factor file       : monthly Mkt-RF, SMB, HML, RF
       - 25-portfolio file   : only the 'monthly Value Weighted' section
  2) Percent -> decimal (/100)
  3) Missing data (-99.99, -999) -> NaN
  4) Aligns the dates of the two files (common months only)
  5) Subtracts the risk-free rate (RF) from each portfolio return to form 'excess returns'

The return shape is exactly what the engine (ff_core.summarize) expects:
    excess_returns : (T x 25) excess returns
    factors        : (T x 3)  [Mkt-RF, SMB, HML]
    rf             : (T,)     risk-free rate (for reference)

Design note: line numbers are not hard-coded. Ken French appends a new row every
month, which shifts line numbers, so sections are located purely by the 'section
title' and the 'YYYYMM row' pattern.
"""

from __future__ import annotations

import io
import os
import re
import zipfile
import urllib.request

import numpy as np
import pandas as pd

# Ken French direct download URLs (only used when fetching fresh data locally)
FACTORS_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/"
               "ken.french/ftp/F-F_Research_Data_Factors_CSV.zip")
PORTFOLIOS_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/"
                  "ken.french/ftp/25_Portfolios_5x5_CSV.zip")

_DATE_ROW = re.compile(r"^\s*\d{6}\s*,")   # monthly data row starting with 'YYYYMM,'
_MISSING = [-99.99, -999]                  # Ken French missing-data markers


def _read_text(path: str) -> str:
    """If the path is a .zip, read the single CSV inside it; otherwise read the file as-is."""
    if str(path).lower().endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
            return z.read(name).decode("latin-1")
    with open(path, "r", encoding="latin-1") as f:
        return f.read()


def _collect_block(raw_lines, header_idx):
    """Below the column-name row (header_idx), collect consecutive 'YYYYMM,' rows.
       Once data has started, stop at the first non-data row (blank, next section title, annual row)."""
    header_line = raw_lines[header_idx].strip()
    data = []
    for ln in raw_lines[header_idx + 1:]:
        if _DATE_ROW.match(ln):
            data.append(ln.strip())
        elif data:
            break
    return header_line, data


def _parse_block(header_line, data_lines):
    """Column-name row + monthly data rows -> DataFrame.
       Dates become a monthly (Period) index; values are percent -> decimal."""
    csv_text = header_line + "\n" + "\n".join(data_lines)
    df = pd.read_csv(io.StringIO(csv_text), skipinitialspace=True,
                     na_values=_MISSING)
    df = df.rename(columns={df.columns[0]: "date"})
    idx = pd.to_datetime(df["date"].astype(int).astype(str),
                         format="%Y%m").dt.to_period("M")
    df = df.drop(columns="date")
    df.index = idx
    df = df.replace(_MISSING, np.nan)        # catch any leftover markers once more
    return df / 100.0                        # percent -> decimal


def _load_factors(path: str):
    """Only the monthly section of the 3-factor file. Returns (factors[Mkt-RF,SMB,HML], rf)."""
    raw = _read_text(path).replace("\r", "").split("\n")
    # Find the monthly column-name row ',Mkt-RF,SMB,HML,RF' (everything above it is descriptive text)
    h = next(i for i, ln in enumerate(raw) if ln.strip().startswith(",Mkt-RF"))
    header_line, data = _collect_block(raw, h)
    df = _parse_block(header_line, data)
    return df[["Mkt-RF", "SMB", "HML"]], df["RF"]


def _load_portfolios(path: str,
                     section="Average Value Weighted Returns -- Monthly"):
    """Extract the given section (default = monthly value-weighted) from the 25-portfolio file."""
    raw = _read_text(path).replace("\r", "").split("\n")
    s = next(i for i, ln in enumerate(raw) if section in ln)      # section title row
    h = next(i for i in range(s + 1, len(raw)) if raw[i].strip())  # the next column-name row
    header_line, data = _collect_block(raw, h)
    return _parse_block(header_line, data)


def load_ff_us(factors_path: str, portfolios_path: str,
               start: str | None = None, end: str | None = None):
    """
    Read the Ken French US data and return it in the engine's input shape.

    factors_path, portfolios_path : .zip or .csv paths
    start, end                    : 'YYYY-MM' sample-window trimming (optional)

    Returns: (excess_returns[T x 25], factors[T x 3], rf[T])
    """
    factors, rf = _load_factors(factors_path)
    ports = _load_portfolios(portfolios_path)

    # Align dates (common months only)
    idx = factors.index.intersection(ports.index).sort_values()
    factors, rf, ports = factors.loc[idx], rf.loc[idx], ports.loc[idx]

    # excess return = portfolio return - risk-free rate
    excess = ports.sub(rf, axis=0)

    # Sample-window trimming (optional)
    if start is not None:
        excess, factors, rf = excess.loc[start:], factors.loc[start:], rf.loc[start:]
    if end is not None:
        excess, factors, rf = excess.loc[:end], factors.loc[:end], rf.loc[:end]

    return excess, factors, rf


def download_ff_us(dest_dir: str = "."):
    """(Local only) Download the two zip files directly from Ken French. Returns the 2 saved paths.
       Note: this function only works on a machine with internet access."""
    os.makedirs(dest_dir, exist_ok=True)
    paths = []
    for url in (FACTORS_URL, PORTFOLIOS_URL):
        fn = os.path.join(dest_dir, url.rsplit("/", 1)[1])
        urllib.request.urlretrieve(url, fn)
        paths.append(fn)
    return paths[0], paths[1]


def find_files(factors_name="F-F_Research_Data_Factors_CSV.zip",
               ports_name="25_Portfolios_5x5_CSV.zip"):
    """Find the two zips in data/ or the current folder; return (factors_path, ports_path)."""
    def _find(name):
        for d in [".", "data", os.path.join("..", "data")]:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"'{name}' not found. Place it in the data/ folder.")
    return _find(factors_name), _find(ports_name)
