"""Post-earnings log-volatility labels.

    v_tau = log( sqrt( (1/tau) * sum_{i=0}^{tau-1} (r_{t+i} - rbar)^2 ) )

r are *simple* daily close-to-close returns, rbar is their mean over the window,
and t is the first trading session whose close reflects the release:
  - release before the open (BeforeMarket)  -> t = release day
  - release after the close (AfterMarket)   -> t = next trading day

This matches DEC exactly (lv{tau}_future_{tau} from p_future_1..tau, see
tests/test_labels.py) and Qin & Yang (2019). Note the window has tau returns
(i = 0..tau-1) and the 1/tau factor; the proposal's formula omitted 1/tau and
had an off-by-one upper limit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (3, 7, 15, 30)
BMO, AMC = "BeforeMarket", "AfterMarket"


def log_vol(returns) -> float:
    """log of the population std of a window of returns."""
    x = np.asarray(returns, dtype=float)
    if len(x) == 0 or np.isnan(x).any():
        return float("nan")
    return float(np.log(np.sqrt(np.mean((x - x.mean()) ** 2))))


def first_session(release_day, timing: str, sessions: pd.DatetimeIndex) -> pd.Timestamp | None:
    """First trading session that prices in the release."""
    day = pd.Timestamp(release_day).normalize()
    if timing == BMO:
        idx = sessions.searchsorted(day, side="left")   # same day if it trades
    elif timing == AMC:
        idx = sessions.searchsorted(day, side="right")  # strictly after
    else:
        raise ValueError(f"unknown timing {timing!r}")
    return sessions[idx] if idx < len(sessions) else None


def labels_for_call(close: pd.Series, release_day, timing: str,
                    horizons=HORIZONS) -> dict[str, float]:
    """Labels lv{tau} for one call, given a ticker's close series (DatetimeIndex)."""
    close = close.dropna().sort_index()
    ret = close.pct_change()
    t = first_session(release_day, timing, close.index)
    out = {"t": t}
    if t is None:
        return out | {f"lv{h}": float("nan") for h in horizons}
    i = close.index.get_loc(t)
    for h in horizons:
        window = ret.iloc[i:i + h]
        out[f"lv{h}"] = log_vol(window.values) if len(window) == h else float("nan")
    return out
