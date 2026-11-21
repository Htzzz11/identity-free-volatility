import numpy as np
import pandas as pd
import pytest

from identity_free_volatility import labels
from identity_free_volatility.paths import DEC_CSV

dec_available = pytest.mark.skipif(not DEC_CSV.exists(), reason="DEC.csv not downloaded")


def test_formula_matches_dec_columns():
    """DEC's lv{tau}_future_{tau} == log std (population, 1/tau) of p_future_1..tau."""
    if not DEC_CSV.exists():
        pytest.skip("DEC.csv not downloaded")
    dec = pd.read_csv(DEC_CSV)
    for tau in labels.HORIZONS:
        ours = dec[[f"p_future_{k}" for k in range(1, tau + 1)]].apply(
            lambda row: labels.log_vol(row.values), axis=1)
        theirs = dec[f"lv{tau}_future_{tau}"]
        assert np.allclose(ours, theirs, atol=1e-9), tau


def test_first_session():
    sessions = pd.DatetimeIndex(["2017-11-14", "2017-11-15", "2017-11-16", "2017-11-20"])
    assert labels.first_session("2017-11-15", labels.BMO, sessions) == pd.Timestamp("2017-11-15")
    assert labels.first_session("2017-11-15", labels.AMC, sessions) == pd.Timestamp("2017-11-16")
    # release on a non-trading day (Fri 17th / weekend) -> next session either way
    assert labels.first_session("2017-11-17", labels.BMO, sessions) == pd.Timestamp("2017-11-20")
    assert labels.first_session("2017-11-18", labels.AMC, sessions) == pd.Timestamp("2017-11-20")


def test_target_q3_2017_label():
    """Target FY2017 Q3 release, 2017-11-15 before the open: tau=3 label is -2.726."""
    from identity_free_volatility import prices
    close = prices.cache_closes(["TGT"])["TGT"]
    out = labels.labels_for_call(close, "2017-11-15", labels.BMO)
    assert out["t"] == pd.Timestamp("2017-11-15")
    assert abs(out["lv3"] - (-2.726)) < 2e-3, out["lv3"]


@dec_available
def test_pipeline_reproduces_dec_labels():
    """End to end on a few DEC calls: yfinance closes + DEC timing -> DEC's labels.

    Loose tolerance because Yahoo's adjusted closes drift slightly from DEC's
    price vendor (dividend adjustment), not because the formula differs.
    """
    from identity_free_volatility import prices
    dec = pd.read_csv(DEC_CSV)
    sample = dec[dec.ticker.isin(["AAPL", "TGT", "JPM"])]
    close = prices.cache_closes(["AAPL", "TGT", "JPM"])
    diffs = []
    for r in sample.itertuples(index=False):
        ours = labels.labels_for_call(close[r.ticker], r.day_earnings, r.beforeAfterMarket)
        diffs.append(abs(ours["lv3"] - r.lv3_future_3))
    assert np.nanmedian(diffs) < 0.01
    assert np.nanmean(np.array(diffs) < 0.05) > 0.95


@dec_available
def test_dec_flag_is_read_from_the_label_window():
    """EOG reports after the close and holds the call next morning. DEC lists the
    call date (2022-08-05) with AfterMarket, but its returns start on 08-05, so
    relative to the call date the flag must be BeforeMarket."""
    from identity_free_volatility import prices, strux
    close = prices.cache_closes(["EOG", "AAPL"])
    calls = pd.DataFrame({"call_id": ["EOG_20220805", "AAPL_20200430"], "ticker": ["EOG", "AAPL"],
                          "date": pd.to_datetime(["2022-08-05", "2020-04-30"]), "header": ["", ""]})
    out = strux.recover_release(calls, None, close)
    assert out["dec"].tolist() == [labels.BMO, labels.AMC]
