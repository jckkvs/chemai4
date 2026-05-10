import pytest
import numpy as np
import pandas as pd
from backend.models.cv_manager import CVConfig, get_cv, WalkForwardSplit
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut, GroupKFold, StratifiedKFold

def test_cv_aliases():
    # loo
    cfg = CVConfig(cv_key="loo")
    cv = get_cv(cfg)
    assert isinstance(cv, LeaveOneOut)

    # logo
    cfg = CVConfig(cv_key="logo")
    cv = get_cv(cfg)
    assert isinstance(cv, LeaveOneGroupOut)

    # groupfold
    cfg = CVConfig(cv_key="groupfold", n_splits=3)
    cv = get_cv(cfg)
    assert isinstance(cv, GroupKFold)
    assert cv.get_n_splits() == 3

    # stratifiedfold
    cfg = CVConfig(cv_key="stratifiedfold", n_splits=4)
    cv = get_cv(cfg)
    assert isinstance(cv, StratifiedKFold)
    assert cv.get_n_splits() == 4

    # walkthrough
    cfg = CVConfig(cv_key="walkthrough", n_splits=5, extra_params={"gap": 10})
    cv = get_cv(cfg)
    assert isinstance(cv, WalkForwardSplit)
    assert cv.n_splits == 5
    assert cv.gap == 10

def test_type_conversion():
    # 文字列から数値への変換テスト (UI入力を想定)
    cfg = CVConfig(
        cv_key="walkthrough",
        extra_params={
            "n_splits": "7",  # 文字列
            "gap": "5",       # 文字列
            "min_train_size": "100"
        }
    )
    cv = get_cv(cfg)
    assert cv.n_splits == 7
    assert cv.gap == 5
    assert cv.min_train_size == 100

def test_bool_conversion():
    # 文字列から真偽値への変換
    cfg = CVConfig(
        cv_key="kfold",
        extra_params={"shuffle": "true"}
    )
    cv = get_cv(cfg)
    assert cv.shuffle is True

    cfg = CVConfig(
        cv_key="kfold",
        extra_params={"shuffle": "False", "random_state": 42}
    )
    cv = get_cv(cfg)
    # random_state があるので shuffle=True に自動変換されるはず
    assert cv.shuffle is True
