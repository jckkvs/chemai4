"""
backend/utils/compute_budget.py

xTB等の量子化学計算のリソース管理ユーティリティ。

計算時間の事前見積もり、大規模データセットでの近似モード推奨、
タイムアウト設定の自動調整などを提供する。

既存の xtb_adapter.py のタイムアウト機構を補完する上位レイヤー。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ComputeBudget:
    """
    量子化学計算のリソース管理。

    xTB計算は O(N²〜N³) のスケーリングを持つため（Nは原子数）、
    大規模データセットや大きな分子では計算コストが急増する。
    このクラスは事前に計算量を見積もり、適切な設定を推奨する。

    Attributes:
        max_molecules: パイプラインで処理する最大分子数。
        timeout_per_mol: 1分子あたりのタイムアウト秒数。
        max_atoms_for_opt: 構造最適化(opt)を許可する最大原子数。
            これを超える分子は自動的に単点計算(sp)にフォールバック。
        approx_threshold: この分子数を超えると近似モード推奨。
    """
    max_molecules: int = 500
    timeout_per_mol: int = 300
    max_atoms_for_opt: int = 200
    approx_threshold: int = 50

    def estimate_xtb_time_seconds(
        self,
        n_atoms: int,
        calc_type: str = "opt",
    ) -> float:
        """
        1分子のxTB計算時間を簡易見積もりする（経験則ベース）。

        GFN2-xTBのスケーリング:
        - 単点計算(sp): O(N²), N=原子数、概算 ~0.3秒/原子
        - 構造最適化(opt): sp × 平均反復数(~10)、概算 ~3秒/原子
        - 振動計算(freq): opt + 3N個のsp → 概算 ~1秒/原子/次元

        Args:
            n_atoms: 原子数。
            calc_type: 計算タイプ ("sp", "opt", "freq")。

        Returns:
            推定計算時間（秒）。
        """
        if calc_type == "sp":
            return 0.3 * n_atoms
        elif calc_type == "freq":
            return 1.0 * n_atoms * 3  # 3N 個の変位
        else:  # opt（デフォルト）
            return 3.0 * n_atoms

    def estimate_total_time_minutes(
        self,
        n_molecules: int,
        avg_atoms: int = 30,
        calc_type: str = "opt",
    ) -> float:
        """
        データセット全体の推定計算時間（分）を返す。

        Args:
            n_molecules: 分子の数。
            avg_atoms: 平均原子数（デフォルト30: 中型有機分子）。
            calc_type: 計算タイプ。

        Returns:
            推定合計時間（分）。
        """
        per_mol = self.estimate_xtb_time_seconds(avg_atoms, calc_type)
        return (per_mol * n_molecules) / 60.0

    def should_use_approx(self, n_molecules: int) -> bool:
        """大規模データセットで近似モード（sp計算）を推奨すべきかを返す。"""
        return n_molecules > self.approx_threshold

    def recommend_calc_type(
        self,
        n_molecules: int,
        avg_atoms: int = 30,
    ) -> str:
        """
        データセットの規模に応じて推奨計算タイプを返す。

        ルール:
        - n_molecules <= 50 かつ avg_atoms <= 200: "opt"（構造最適化）
        - n_molecules <= 200 または avg_atoms > 200: "sp"（単点計算）
        - n_molecules > 200: "sp" + 警告

        Returns:
            "opt" or "sp"
        """
        if n_molecules > self.max_molecules:
            logger.warning(
                "分子数 %d は上限 %d を超えています。"
                "先頭 %d 件のみ処理されます。",
                n_molecules, self.max_molecules, self.max_molecules,
            )

        if avg_atoms > self.max_atoms_for_opt:
            logger.info(
                "平均原子数 %d が閾値 %d を超えるため、"
                "単点計算(sp)を推奨します。",
                avg_atoms, self.max_atoms_for_opt,
            )
            return "sp"

        if self.should_use_approx(n_molecules):
            est_opt = self.estimate_total_time_minutes(n_molecules, avg_atoms, "opt")
            est_sp = self.estimate_total_time_minutes(n_molecules, avg_atoms, "sp")
            logger.info(
                "大規模データセット (%d分子): "
                "opt推定=%d分, sp推定=%d分 → spを推奨",
                n_molecules, int(est_opt), int(est_sp),
            )
            return "sp"

        return "opt"

    def get_summary(
        self,
        n_molecules: int,
        avg_atoms: int = 30,
    ) -> dict[str, str | float | int]:
        """
        計算量見積もりのサマリー辞書を返す（UI表示用）。

        Returns:
            {"recommended_calc_type": str, "estimated_minutes": float, ...}
        """
        rec_type = self.recommend_calc_type(n_molecules, avg_atoms)
        est_min = self.estimate_total_time_minutes(n_molecules, avg_atoms, rec_type)

        return {
            "n_molecules": min(n_molecules, self.max_molecules),
            "avg_atoms": avg_atoms,
            "recommended_calc_type": rec_type,
            "estimated_minutes": round(est_min, 1),
            "timeout_per_mol": self.timeout_per_mol,
            "note": (
                "構造最適化(opt): 高精度だが低速 / "
                "単点計算(sp): 高速だが初期構造依存"
            ),
        }
