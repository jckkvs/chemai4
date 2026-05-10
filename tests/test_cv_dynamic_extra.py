import pytest
import numpy as np
from backend.models.cv_manager import CVConfig, get_cv
from sklearn.model_selection import LeaveOneOut, RepeatedKFold

def test_dynamic_cv_lookup():
    # レジストリにない手法を名前で取得
    cfg = CVConfig(cv_key="LeaveOneOut")
    cv = get_cv(cfg)
    assert isinstance(cv, LeaveOneOut)

def test_dynamic_cv_extra_params():
    # RepeatedKFold に n_repeats を渡す
    cfg = CVConfig(
        cv_key="RepeatedKFold", 
        n_splits=5, 
        extra_params={"n_repeats": 10}
    )
    cv = get_cv(cfg)
    assert isinstance(cv, RepeatedKFold)
    assert cv.n_repeats == 10

def test_invalid_cv_param_warning(caplog):
    # 無効な引数を渡した場合の警告確認
    import logging
    with caplog.at_level(logging.WARNING, logger="backend.models.cv_manager"):
        cfg = CVConfig(
            cv_key="KFold",
            extra_params={"invalid_arg": 123}
        )
        cv = get_cv(cfg)
    assert "無効な引数が指定されました" in caplog.text
