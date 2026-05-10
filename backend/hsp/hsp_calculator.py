# -*- coding: utf-8 -*-
"""
backend/hsp/hsp_calculator.py

Hansen Solubility Parameters (HSP) 計算モジュール。
RED値計算 + Hansen距離 + SMILES→HSP計算。

HSPiPy非依存。SMILES→HSPはvan Krevelen/Hoftyzer Group Contribution法
(hsp_predictor.py) を使用。

Implements: RED値・Hansen距離・HSP球体フィッティング
引用: Hansen, C.M. "Hansen Solubility Parameters: A User's Handbook",
      2nd Ed., CRC Press, 2007

API:
    calculate_red_value()  → RED値計算
    hansen_distance()      → Hansen距離
    predict_from_smiles()  → SMILES→HSP (Group Contribution法)

前提:
    pip install rdkit  # SMILES→HSP 計算用
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class HSPCalculator:
    """Hansen Solubility Parameters の計算・評価ツール。

    Implements: Hansen 2007, Ch.2
    引用: Hansen, C.M. "Hansen Solubility Parameters", 2nd Ed., 2007

    RED < 1 → 溶解  /  RED > 1 → 不溶解
    Ra = √(4·ΔδD² + ΔδP² + ΔδH²)   ... 式(1)
    RED = Ra / R₀                      ... 式(2)
    """

    @staticmethod
    def calculate_red_value(
        solute_hsp: tuple[float, float, float],
        solvent_hsp: tuple[float, float, float],
        radius: float,
    ) -> float:
        """
        RED 値 (Relative Energy Difference) 計算。

        Implements: Hansen 2007, §2.2 式(1)(2)
        引用: Ra = √(4·ΔδD² + ΔδP² + ΔδH²)
              RED = Ra / R₀

        RED < 1 → 溶解 / RED > 1 → 不溶解

        Args:
            solute_hsp: 溶質の (δD, δP, δH)
            solvent_hsp: 溶媒の (δD, δP, δH)
            radius: 溶解性球体の半径 R₀
        """
        ra = HSPCalculator.hansen_distance(solute_hsp, solvent_hsp)
        if radius <= 0:
            return float("inf")
        return ra / radius

    @staticmethod
    def hansen_distance(
        hsp_a: tuple[float, float, float],
        hsp_b: tuple[float, float, float],
    ) -> float:
        """
        Hansen 距離 Ra 計算。

        Implements: Hansen 2007, §2.2 式(1)
        引用: Ra = √(4·ΔδD² + ΔδP² + ΔδH²)

        分散力は4倍の重み付け（van Krevelen 2009も同様の重み付け）。

        Args:
            hsp_a: (δD, δP, δH) of compound A
            hsp_b: (δD, δP, δH) of compound B

        Returns:
            Hansen距離 Ra (MPa^0.5)
        """
        d_diff = hsp_a[0] - hsp_b[0]
        p_diff = hsp_a[1] - hsp_b[1]
        h_diff = hsp_a[2] - hsp_b[2]

        return float(np.sqrt(4 * d_diff**2 + p_diff**2 + h_diff**2))

    @staticmethod
    def predict_from_smiles(smiles: str) -> dict[str, Any]:
        """
        SMILES → HSP 予測 (van Krevelen/Hoftyzer Group Contribution法)。

        Implements: van Krevelen 2009, Table 4.3 + 式(4.6)-(4.8)

        Args:
            smiles: SMILES文字列

        Returns:
            {"delta_d", "delta_p", "delta_h", "delta_total",
             "molar_volume", "method", "confidence"}
        """
        from backend.hsp.hsp_predictor import HSPPredictor
        predictor = HSPPredictor()
        return predictor.predict(smiles)

    @staticmethod
    def predict_batch(smiles_list: list[str]) -> Any:
        """
        バッチ SMILES → HSP 予測。

        Args:
            smiles_list: SMILES文字列のリスト

        Returns:
            pd.DataFrame
        """
        from backend.hsp.hsp_predictor import HSPPredictor
        predictor = HSPPredictor()
        return predictor.predict_batch(smiles_list)

    @staticmethod
    def fit_sphere(
        hsp_data: list[tuple[float, float, float]],
        labels: list[bool],
    ) -> dict[str, Any]:
        """
        HSP球体フィッティング（scipy.optimize使用）。

        実験溶解性データから最適なHSP中心と半径を求める。

        Implements: Hansen 2007, §2.3 球体最適化
        引用: 「溶解性球体の中心が溶質のHSP、半径がR₀」

        Args:
            hsp_data: 溶媒の[(δD, δP, δH), ...]
            labels:   [True=溶解, False=不溶解, ...]

        Returns:
            {"center": [δD, δP, δH], "radius": R₀,
             "accuracy": float, "n_correct": int}
        """
        if len(hsp_data) < 3:
            raise ValueError("球体フィッティングには最低3点のデータが必要です")

        from scipy.optimize import minimize

        pts = np.array(hsp_data)
        labs = np.array(labels, dtype=bool)

        def _objective(params: np.ndarray) -> float:
            center = params[:3]
            r = abs(params[3])
            # Hansen距離（4倍重み付け）
            diffs = pts - center
            dists = np.sqrt(4 * diffs[:, 0]**2 + diffs[:, 1]**2 + diffs[:, 2]**2)
            # RED = dist / r
            predicted_in = dists <= r
            # 不一致数をコスト関数に
            n_wrong = int(np.sum(predicted_in != labs))
            # 正則化: 半径が極端に大きい/小さいのを防止
            reg = 0.01 * (r - 5.0)**2
            return float(n_wrong + reg)

        # 初期値: 溶解点の重心を中心、距離の中央値を半径
        if np.any(labs):
            center0 = pts[labs].mean(axis=0)
        else:
            center0 = pts.mean(axis=0)

        diffs0 = pts - center0
        dists0 = np.sqrt(4 * diffs0[:, 0]**2 + diffs0[:, 1]**2 + diffs0[:, 2]**2)
        r0 = float(np.median(dists0))

        result = minimize(
            _objective,
            x0=np.array([*center0, r0]),
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 0.01, "fatol": 0.5},
        )

        center_opt = result.x[:3]
        r_opt = abs(result.x[3])

        # 精度計算
        diffs_opt = pts - center_opt
        dists_opt = np.sqrt(4 * diffs_opt[:, 0]**2 + diffs_opt[:, 1]**2 + diffs_opt[:, 2]**2)
        predicted_in = dists_opt <= r_opt
        n_correct = int(np.sum(predicted_in == labs))
        accuracy = n_correct / len(labs)

        return {
            "center": center_opt.tolist(),
            "radius": float(r_opt),
            "accuracy": float(accuracy),
            "n_correct": n_correct,
            "n_total": len(labs),
            "n_in": int(np.sum(predicted_in)),
            "n_out": int(np.sum(~predicted_in)),
        }
