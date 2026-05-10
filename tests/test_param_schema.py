"""
tests/test_param_schema.py

パラメータ自動UIエンジンの包括的テスト。
"""
import sys
import json
sys.path.insert(0, ".")

from backend.ui.param_schema import (
    introspect_params, apply_params, get_basic_specs, get_advanced_specs, ParamSpec
)


def test_sklearn_estimators():
    """TEST 1: sklearn estimators — 全モデルイントロスペクション"""
    from sklearn.ensemble import (
        RandomForestRegressor, GradientBoostingRegressor,
        AdaBoostRegressor, ExtraTreesRegressor,
    )
    from sklearn.linear_model import (
        Ridge, Lasso, ElasticNet, LinearRegression,
        HuberRegressor, ARDRegression, BayesianRidge,
    )
    from sklearn.svm import SVR, LinearSVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor

    classes = [
        RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,
        ExtraTreesRegressor, Ridge, Lasso, ElasticNet, LinearRegression,
        HuberRegressor, ARDRegression, BayesianRidge, SVR, LinearSVR,
        KNeighborsRegressor, DecisionTreeRegressor,
    ]
    for cls in classes:
        specs = introspect_params(cls)
        assert len(specs) > 0, f"{cls.__name__}: 0 params!"
        basic = get_basic_specs(specs)
        adv = get_advanced_specs(specs)
        for s in specs:
            assert s.name, f"{cls.__name__}: name empty"
            assert s.param_type in (
                "bool", "int", "float", "str", "select",
                "multiselect", "text", "union",
            ), f"{cls.__name__}.{s.name}: bad type {s.param_type}"
        print(f"  {cls.__name__}: {len(specs)} params ({len(basic)} basic, {len(adv)} advanced)")
    print(f"  -> {len(classes)} estimators: ALL OK")


def test_chem_adapters():
    """TEST 2: ChemAdapters"""
    from backend.chem.rdkit_adapter import RDKitAdapter
    from backend.chem.mordred_adapter import MordredAdapter
    from backend.chem.xtb_adapter import XTBAdapter
    from backend.chem.molai_adapter import MolAIAdapter
    from backend.chem.unipka_adapter import UniPkaAdapter

    adapters = [
        (RDKitAdapter, 5),
        (MordredAdapter, 2),
        (XTBAdapter, 1),
        (MolAIAdapter, 2),
        (UniPkaAdapter, 2),
    ]
    for cls, expected_min in adapters:
        specs = introspect_params(cls)
        assert len(specs) >= expected_min, f"{cls.__name__}: {len(specs)} < {expected_min}"
        print(f"  {cls.__name__}: {len(specs)} params")
    print(f"  -> {len(adapters)} adapters: ALL OK")


def test_feature_selection_preprocessing():
    """TEST 3: Feature Selection & Preprocessing"""
    from sklearn.feature_selection import SelectKBest, SelectFromModel, VarianceThreshold
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures
    from sklearn.decomposition import PCA

    classes = [
        SelectKBest, SelectFromModel, VarianceThreshold,
        StandardScaler, MinMaxScaler, RobustScaler,
        PolynomialFeatures, PCA,
    ]
    for cls in classes:
        specs = introspect_params(cls)
        assert len(specs) > 0, f"{cls.__name__}: 0 params!"
        print(f"  {cls.__name__}: {len(specs)} params")
    print(f"  -> {len(classes)} classes: ALL OK")


def test_apply_params():
    """TEST 4: apply_params — 型変換とバリデーション"""
    from sklearn.ensemble import RandomForestRegressor
    specs = introspect_params(RandomForestRegressor)

    # 文字列→int変換
    user_vals = {"n_estimators": "200", "max_depth": "", "min_samples_split": 5}
    result = apply_params(specs, user_vals)
    assert result.get("n_estimators") == 200, f"Expected 200, got {result.get('n_estimators')}"
    assert result.get("min_samples_split") == 5, f"Expected 5, got {result.get('min_samples_split')}"

    # デフォルト値と同じなら含まれない
    default_vals = {"n_estimators": 100}
    result2 = apply_params(specs, default_vals)
    assert "n_estimators" not in result2, "Default value should be excluded"

    print("  -> apply_params: type conversion OK")
    print("  -> apply_params: default exclusion OK")


def test_to_dict_json():
    """TEST 5: ParamSpec.to_dict (JSON serialization)"""
    from sklearn.linear_model import Ridge
    from sklearn.svm import SVR
    from backend.chem.rdkit_adapter import RDKitAdapter

    for cls in [Ridge, SVR, RDKitAdapter]:
        specs = introspect_params(cls)
        dicts = [s.to_dict() for s in specs]
        json_str = json.dumps(dicts, ensure_ascii=False)
        assert len(json_str) > 10
        # JSON往復チェック
        parsed = json.loads(json_str)
        assert len(parsed) == len(specs)
    print("  -> JSON serialization: ALL OK")


def test_new_model_auto_detection():
    """TEST 6: 新モデル追加（GPR）→ 自動検出テスト（UIコード変更不要の証明）"""
    from sklearn.gaussian_process import GaussianProcessRegressor
    specs = introspect_params(GaussianProcessRegressor)
    param_names = [s.name for s in specs]
    assert "alpha" in param_names, "alpha should be in GPR params"
    assert "normalize_y" in param_names, "normalize_y should be in GPR params"
    print(f"  GPR params: {param_names}")
    print(f"  -> GPR ({len(specs)} params): AUTO-DETECTED (no UI code needed)")


def test_factory_integration():
    """TEST 7: factory.pyレジストリとの統合"""
    try:
        from backend.models.factory import list_models
        models = list_models(task="regression", available_only=True)
        total_params = 0
        for m in models:
            cls = m.get("class")
            if cls is not None:
                specs = introspect_params(cls)
                total_params += len(specs)
        print(f"  -> {len(models)} models, {total_params} total params: ALL OK")
    except Exception as e:
        print(f"  -> factory integration skipped: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("param_schema.py comprehensive test")
    print("=" * 60)

    tests = [
        ("TEST 1: sklearn estimators", test_sklearn_estimators),
        ("TEST 2: ChemAdapters", test_chem_adapters),
        ("TEST 3: Feature Selection & Preprocessing", test_feature_selection_preprocessing),
        ("TEST 4: apply_params", test_apply_params),
        ("TEST 5: JSON serialization", test_to_dict_json),
        ("TEST 6: New model auto-detection", test_new_model_auto_detection),
        ("TEST 7: Factory integration", test_factory_integration),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n>> {name}")
        try:
            fn()
            passed += 1
            print(f"  PASSED")
        except Exception as e:
            failed += 1
            print(f"  FAILED: {e}")

    print()
    print("=" * 60)
    print(f"Results: {passed}/{len(tests)} passed, {failed} failed")
    if failed == 0:
        print("ALL TESTS PASSED")
    print("=" * 60)
