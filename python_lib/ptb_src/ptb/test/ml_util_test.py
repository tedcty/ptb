"""MLOperations.extract_features_from_x honours the feature set it's given.

Up to 0.3.27 a tsfresh settings object such as EfficientFCParameters() silently fell back
to MinimalFCParameters, then raised AttributeError on .name. Run with pytest, or directly.
"""
import numpy as np
import pandas as pd
from tsfresh.feature_extraction import (extract_features, ComprehensiveFCParameters,
                                        EfficientFCParameters, MinimalFCParameters)
from tsfresh.utilities.dataframe_functions import impute

from ptb.ml.ml_util import MLOperations
from ptb.ml.tags import MLKeys


def _windows():
    """Two 50-sample windows of one channel, in the id/time layout MLOperations expects."""
    rng = np.random.default_rng(0)
    t = np.arange(50) / 50.0
    return pd.DataFrame({"id": np.repeat([1, 2], 50),
                         "time": np.tile(t, 2),
                         "acc": np.concatenate([np.sin(8 * t), np.cos(5 * t)]) + rng.normal(0, 0.1, 100)})


def _direct(x, settings):
    """What tsfresh itself returns for these settings."""
    return extract_features(x, column_id="id", column_sort="time", default_fc_parameters=settings,
                            impute_function=impute, n_jobs=0)


def test_a_settings_object_is_used_as_given():
    x = _windows()
    efx, param = MLOperations.extract_features_from_x(x, fc_parameters=EfficientFCParameters(), n_jobs=0)

    expected = _direct(x, EfficientFCParameters())
    assert list(efx.columns) == list(expected.columns)
    assert len(efx.columns) > len(_direct(x, MinimalFCParameters()).columns)
    assert param["fc_parameters"] == "EfficientFCParameters"


def test_a_plain_dict_of_features_is_a_feature_set_too():
    x = _windows()
    efx, param = MLOperations.extract_features_from_x(x, fc_parameters={"mean": None, "maximum": None},
                                                      n_jobs=0)

    assert sorted(efx.columns) == ["acc__maximum", "acc__mean"]
    assert param["fc_parameters"] == "dict"


def test_the_mlkeys_members_keep_their_meaning():
    x = _windows()
    comprehensive, p_c = MLOperations.extract_features_from_x(x, fc_parameters=MLKeys.CFCParameters, n_jobs=0)
    minimal, p_m = MLOperations.extract_features_from_x(x, fc_parameters=MLKeys.MFCParameters, n_jobs=0)

    assert list(comprehensive.columns) == list(_direct(x, ComprehensiveFCParameters()).columns)
    assert list(minimal.columns) == list(_direct(x, MinimalFCParameters()).columns)
    assert (p_c["fc_parameters"], p_m["fc_parameters"]) == ("CFCParameters", "MFCParameters")


def test_anything_else_is_refused_by_name():
    try:
        MLOperations.fc_settings("efficient")
    except TypeError as e:
        assert "EfficientFCParameters()" in str(e)
    else:
        raise AssertionError("a string was accepted as a feature set")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
