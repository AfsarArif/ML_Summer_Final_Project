from pathlib import Path

import numpy as np
import pandas as pd


# position / cash_ratio get overwritten each day by Book
FEATS = [
    "return_1d",
    "log_volume",
    "rsi_14",
    "sma_ratio_5",
    "sma_ratio_20",
    "ema_ratio_12",
    "macd",
    "macd_signal",
    "macd_hist",
    "volatility_10",
    "position",
    "cash_ratio",
]


def rsi(close, w=14):
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = -d.clip(upper=0.0)
    au = up.ewm(alpha=1.0 / w, min_periods=w, adjust=False).mean()
    ad = dn.ewm(alpha=1.0 / w, min_periods=w, adjust=False).mean()
    rs = au / ad.replace(0.0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)


def add_inds(df):
    out = df.copy()
    c = out["Close"].astype(float)
    v = out["Volume"].astype(float)
    out["return_1d"] = c.pct_change().fillna(0.0)
    out["log_volume"] = np.log1p(v)
    out["rsi_14"] = rsi(c, 14)

    s5 = c.rolling(5, min_periods=1).mean()
    s20 = c.rolling(20, min_periods=1).mean()
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    # ratios so 2010 vs 2024 price level doesn't dominate
    out["sma_ratio_5"] = (c / s5) - 1.0
    out["sma_ratio_20"] = (c / s20) - 1.0
    out["ema_ratio_12"] = (c / e12) - 1.0

    m = e12 - e26
    sig = m.ewm(span=9, adjust=False).mean()
    out["macd"] = m
    out["macd_signal"] = sig
    out["macd_hist"] = m - sig
    out["volatility_10"] = out["return_1d"].rolling(10, min_periods=1).std().fillna(0.0)
    out["position"] = 0.0
    out["cash_ratio"] = 1.0
    return out


def pull_spx(start="2010-01-01", end=None):
    import yfinance as yf
    data = yf.download("^GSPC", start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.reset_index()
    need = ["Date", "Open", "High", "Low", "Close", "Volume"]
    miss = [c for c in need if c not in data.columns]
    if miss:
        raise RuntimeError("missing cols " + str(miss))
    return data[need].dropna().reset_index(drop=True)


def make_split(raw, frac=0.8):
    feat = add_inds(raw)
    feat = feat.iloc[30:].reset_index(drop=True)  # burn RSI/MACD warmup
    cut = int(len(feat) * frac)
    # chronological — random shuffle would peek at future closes
    tr = feat.iloc[:cut].copy().reset_index(drop=True)
    te = feat.iloc[cut:].copy().reset_index(drop=True)
    mkt = [c for c in FEATS if c not in ("position", "cash_ratio")]
    mu = tr[mkt].mean()
    sd = tr[mkt].std().replace(0.0, 1.0)
    for fr in (tr, te):
        fr[mkt] = (fr[mkt] - mu) / sd
    return tr, te, {"means": mu.to_dict(), "stds": sd.to_dict()}


def dump_csvs(tr, te, raw, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    raw.to_csv(folder / "sp500_raw.csv", index=False)
    tr.to_csv(folder / "sp500_train.csv", index=False)
    te.to_csv(folder / "sp500_test.csv", index=False)
    pd.concat([tr.assign(split="train"), te.assign(split="test")], ignore_index=True).to_csv(
        folder / "sp500_processed.csv", index=False
    )


def get_spx(folder, force=False):
    folder = Path(folder)
    tp, vp = folder / "sp500_train.csv", folder / "sp500_test.csv"
    if (not force) and tp.exists() and vp.exists():
        return pd.read_csv(tp), pd.read_csv(vp)
    raw_p = folder / "sp500_raw.csv"
    if (not force) and raw_p.exists():
        raw = pd.read_csv(raw_p, parse_dates=["Date"])
    else:
        raw = pull_spx()
    tr, te, _ = make_split(raw)
    dump_csvs(tr, te, raw, folder)
    return tr, te
