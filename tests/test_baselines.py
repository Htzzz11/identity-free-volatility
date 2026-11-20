import numpy as np
import pandas as pd
import pytest

from identity_free_volatility import data
from identity_free_volatility.baselines import historical
from identity_free_volatility.paths import DEC_CSV

pytestmark = pytest.mark.skipif(not DEC_CSV.exists(), reason="DEC.csv not downloaded")


def _mse(corpus, preds, test):
    q = preds[preds.test == test].merge(corpus.frame, on="call_id", suffixes=("_p", ""))
    return [round(float(((q[f"{c}_p"] - q[c]) ** 2).mean()), 3) for c in data.LABELS]


def test_reproduces_yu_et_al_table_4():
    """PEV(Mean) and STPEV(Mean) on DEC, test quarter 2021 Q1."""
    dec = data.load_dec()
    assert _mse(dec, historical.pev(dec), "2021Q1") == [0.433, 0.209, 0.115, 0.105]
    assert _mse(dec, historical.stpev(dec), "2021Q1") == [0.368, 0.149, 0.067, 0.041]


def test_stpev_matches_dec_column():
    dec = data.load_dec()
    raw = pd.read_csv(DEC_CSV, usecols=["id", "stpev_mean_lv3f3"]).set_index("id")
    ours = historical.stpev(dec).set_index("call_id")["lv3"]
    assert np.allclose(ours, raw.loc[ours.index, "stpev_mean_lv3f3"], atol=1e-9)


def test_rolling_splits_never_look_ahead():
    frame = pd.DataFrame({"year": [2018, 2018, 2019, 2019, 2020], "quarter": [1, 4, 1, 2, 1]})
    splits = data.rolling_splits(frame, first_test="2019Q1")
    assert list(splits) == ["2019Q1", "2019Q2", "2020Q1"]
    s = splits["2019Q2"]
    assert s[3] == "test" and s[4] == "none" and set(s[:3]) <= {"train", "val"}
