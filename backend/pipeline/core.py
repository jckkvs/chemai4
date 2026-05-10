"""
backend/pipeline/core.py

化学ML用の5段階統合パイプライン機能を提供するコアモジュール。
各ステージ（入力列制御、前処理、特徴量生成、特徴量選択、Estimator）を組み立て、
Scikit-Learn互換のPipelineを生成する。
"""
from typing import List, Dict, Optional, Union
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from .stages.stage_01_column_control import ColumnSelector
from .stages.stage_02_preprocessing import AutoTypeDetector
from .stages.stage_03_feature_generation import FeatureGenerator
from .stages.stage_04_feature_selection import SelectorFactory
from .stages.stage_05_estimator import EstimatorWrapper
from .constraints.constraints import MonotonicConstraint, LinearityConstraint, GroupConstraint

class ChemPipeline:
    """
    化学ML用の統合パイプライン
    
    5段階アーキテクチャ:
    1. 入力列制御 → 2. 列別前処理 → 3. 特徴量生成 → 4. 特徴量選択 → 5. Estimator
    """
    
    def __init__(self, task: str = "auto"):
        self.task = task
        self._stages = {}
        self._constraints = {}
        self._pipeline: Optional[Pipeline] = None
        
    def set_column_selection(self, mode: str = "all", columns: List[str] = None, start: str = None, end: str = None) -> "ChemPipeline":
        self._stages["column_control"] = ColumnSelector(mode=mode, columns=columns, start=start, end=end)
        return self
    
    def set_preprocessing(self, auto_detect_types: bool = True, numeric_config: Dict = None, categorical_config: Dict = None, binary_config: Dict = None) -> "ChemPipeline":
        self._stages["preprocessing"] = {
            "auto_detect": auto_detect_types,
            "configs": {
                "numeric": numeric_config or {"imputer": "median", "scaler": "standard"},
                "categorical": categorical_config or {"imputer": "constant", "encoder": "onehot"},
                "binary": binary_config or {"imputer": "most_frequent"},
            }
        }
        return self
    
    def add_feature_generation(self, generator_type: str, **kwargs) -> "ChemPipeline":
        if "feature_generation" not in self._stages:
            self._stages["feature_generation"] = []
        self._stages["feature_generation"].append(FeatureGenerator(generator_type, **kwargs))
        return self
    
    def set_feature_selection(self, method: str, **kwargs) -> "ChemPipeline":
        self._stages["feature_selection"] = SelectorFactory.create(method=method, task=self.task, **kwargs)
        return self
    
    def set_estimator(self, estimator: Union[str, BaseEstimator], monotonic_constraints: Dict[str, str] = None, linearity_constraints: Dict[str, float] = None, group_info: Dict[str, List[str]] = None, **estimator_params) -> "ChemPipeline":
        self._stages["estimator"] = EstimatorWrapper(
            estimator=estimator,
            task=self.task,
            monotonic=monotonic_constraints,
            linearity=linearity_constraints,
            groups=group_info,
            **estimator_params
        )
        return self
    
    def add_constraint(self, constraint_type: str, **config) -> "ChemPipeline":
        if constraint_type == "monotonic":
            self._constraints["monotonic"] = MonotonicConstraint(**config)
        elif constraint_type == "linearity":
            self._constraints["linearity"] = LinearityConstraint(**config)
        elif constraint_type == "group":
            self._constraints["group"] = GroupConstraint(**config)
        return self
    
    def build(self, X_columns: List[str]) -> Pipeline:
        steps = []
        
        if "column_control" in self._stages:
            steps.append(("col_select", self._stages["column_control"]))
        
        if "preprocessing" in self._stages:
            cfg = self._stages["preprocessing"]
            preprocessor = AutoTypeDetector.build_transformer(
                X_columns=X_columns,
                auto_detect=cfg["auto_detect"],
                configs=cfg["configs"]
            )
            steps.append(("preprocess", preprocessor))
        
        if "feature_generation" in self._stages:
            for i, gen in enumerate(self._stages["feature_generation"]):
                steps.append((f"feat_gen_{i}", gen))
        
        if "feature_selection" in self._stages:
            steps.append(("feat_select", self._stages["feature_selection"]))
        
        if "estimator" in self._stages:
            est = self._stages["estimator"]
            
            # 制約の互換性検証
            from .constraints.validator import ConstraintValidator
            constraints_list = []
            if "monotonic" in self._constraints:
                constraints_list.append(self._constraints["monotonic"])
            if "linearity" in self._constraints:
                constraints_list.append(self._constraints["linearity"])
            if "group" in self._constraints:
                constraints_list.append(self._constraints["group"])
                
            if constraints_list:
                valid_constraints, warnings = ConstraintValidator.check_compatibility(est.estimator, constraints_list)
                
                # 有効な制約のみを注入
                for c in valid_constraints:
                    c_type = type(c).__name__
                    if c_type == "MonotonicConstraint":
                        est.inject_monotonic(c)
                    elif c_type == "LinearityConstraint":
                        est.inject_linearity(c)
                    elif c_type == "GroupConstraint":
                        est.inject_groups(c)
                        
            steps.append(("estimator", est))
        
        self._pipeline = Pipeline(steps)
        return self._pipeline
    
    def get_ui_schema(self) -> dict:
        """UI生成用のJSON Schemaを出力"""
        return {
            "pipeline_config": {
                "column_control": self._stages.get("column_control").get_ui_schema() if "column_control" in self._stages else {},
                "preprocessing": {}, # 複雑なためプレースホルダ
                "feature_generation": [fg.get_ui_schema() for fg in self._stages.get("feature_generation", [])],
                "feature_selection": SelectorFactory.get_ui_schema(),
                "estimator": self._stages.get("estimator").get_ui_schema() if "estimator" in self._stages else {},
            },
            "constraints": {
                k: v.get_ui_schema() for k, v in self._constraints.items()
            },
            "dynamic_hints": {
                "estimator_params_auto_detect": True,
                "descriptor_params_auto_detect": True,
            }
        }
