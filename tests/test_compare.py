import numpy as np
import pandas as pd
import pytest

from src.classify.compare import bootstrap_macro_f1, macro_f1_codes, mcnemar_exact
from src.classify.evaluate import summary_scores


def test_mcnemar_counts_and_p_value():
    a = [True, True, True, False, True]
    b = [True, False, False, False, False]
    out = mcnemar_exact(a, b)
    assert out["only_a"] == 3 and out["only_b"] == 0
    assert out["p_value"] == pytest.approx(0.25)  # 2 * 0.5**3


def test_mcnemar_no_discordant_pairs():
    assert mcnemar_exact([True, False], [True, False])["p_value"] == 1.0


def test_macro_f1_codes_matches_evaluate():
    labels = ["a", "b", "c"]
    t = ["a", "a", "b", "b", "c"]
    p = ["a", "b", "b", "b", "a"]
    code = {lab: i for i, lab in enumerate(labels)}
    fast = macro_f1_codes(
        np.array([code[x] for x in t]), np.array([code[x] for x in p]), 3
    )
    slow = summary_scores(pd.Series(t), pd.Series(p), labels)["macro_f1"]
    assert fast == pytest.approx(slow)


def test_bootstrap_identical_models_and_reproducible():
    t = ["a", "b", "a", "b", "a", "b"]
    p = ["a", "b", "b", "b", "a", "a"]
    out = bootstrap_macro_f1(t, p, p, ["a", "b"], n_boot=200, seed=0)
    assert out["diff"] == 0 and out["diff_ci"] == (0.0, 0.0)
    assert out["a_ci"][0] <= out["a"] <= out["a_ci"][1]
    again = bootstrap_macro_f1(t, p, p, ["a", "b"], n_boot=200, seed=0)
    assert out == again
