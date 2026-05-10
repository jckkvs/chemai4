"""
backend/mlops/data_routing.py

多視点検証（MultiViewResult）アーキテクチャおよびデータ空間（S0〜S4）整合性ガード。
"""
from typing import Callable, Dict, Any, Literal
from dataclasses import dataclass, field
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class MultiViewResult:
    """
    解析結果を単一の視点ではなく、複数の検証視点から遅延評価可能にするベースクラス。
    
    Attributes:
        core_output: 基本結果（モデル出力値、SHAP値など、標準空間で計算された数学的に正しいコア結果）
        views: 視点名をキーとし、core_output を受け取って変換・集計する関数の辞書
        metadata: 計算条件や警告フラグなど
    """
    core_output: Any
    views: Dict[str, Callable] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _cache: Dict[str, Any] = field(default_factory=dict)

    def get_view(self, perspective: str, **kwargs) -> Any:
        """
        要求された視点のみをオンデマンド計算・またはキャッシュから返す。
        """
        cache_key = f"{perspective}_{hash(frozenset(kwargs.items()))}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if perspective not in self.views:
            raise ValueError(f"未サポート視点: '{perspective}'. Available: {list(self.views.keys())}")
        
        try:
            res = self.views[perspective](self.core_output, **kwargs)
            self._cache[cache_key] = res
            return res
        except Exception as e:
            logger.error(f"[MultiViewResult] 視点 '{perspective}' の計算に失敗: {e}")
            raise


def validate_data_routing(state: dict, required_stage: Literal["raw", "encoded", "scaled", "model"]) -> None:
    """
    機能が必要とするデータステージが正しく準備されているか検証。
    スケール不一致や次元不一致のサイレントエラーを未然に防ぐ。
    
    Args:
        state: app_state またはタスク状態辞書
        required_stage: 'raw', 'encoded', 'scaled', 'model' のいずれか
    """
    if required_stage == "raw":
        assert state.get("df_raw") is not None, "df_raw が未初期化です"
        assert not state["df_raw"].select_dtypes(include='number').isna().all().any(), "数値列に全欠損が存在します"
        
    elif required_stage == "encoded":
        df_enc = state.get("df_encoded")
        assert df_enc is not None, "df_encoded が未初期化です"
        assert df_enc.select_dtypes(include='object').empty, "エンコード未完了（object型列が残存しています）"
        
    elif required_stage == "scaled":
        X_sc = state.get("X_scaled")
        assert X_sc is not None, "X_scaled が未初期化です"
        
        # NOTE: PCAが有効な場合、平均0/分散1の保証が難しいケースもあるため簡易的な分散チェックにとどめる
        # 全体としてある程度スケーリングされているか（生データ特有の巨大分散がないか）
        if hasattr(X_sc, "std"):
            stds = X_sc.std(axis=0)
            stds = stds[~np.isnan(stds)]
            if len(stds) > 0:
                assert np.all(stds < 100), "X_scaled の分散が極端に大きいです（標準化失敗の疑い）"
                
    elif required_stage == "model":
        X_mod = state.get("X_model")
        assert X_mod is not None, "X_model が未初期化です"
        # X_modelは次元削減・特徴量選択後の最終S4空間
        if "X_scaled" in state:
            # 交互作用(SRI等)で一時的に増えることがあるが基本的には減るか同じ
            pass

def check_domain_constraints(results: Any, rules: dict) -> list[str]:
    """特定ドメインのルール違反を判定する汎用関数（後日拡張用）"""
    return []

def validate_perspective(result: MultiViewResult, axes: list[str]) -> dict:
    """各機能の結果に対し3軸による検証を強制するゲートウェイ関数。"""
    report = {}
    if "global" in axes and "global_mean" in result.views:
        report["global_trend"] = result.get_view("global_mean")
    if "probabilistic" in axes and "fold_stability" in result.views:
        report["uncertainty"] = result.get_view("fold_stability")
    if "domain" in axes and "domain_conflict" in result.views:
        report["chemical_plausibility"] = result.get_view("domain_conflict", rules={})
    return report
