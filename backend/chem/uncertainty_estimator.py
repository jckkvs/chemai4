"""
backend/chem/uncertainty_estimator.py

xTB計算結果の信頼度を多角的に評価するモジュール。

計算収束性、電子状態の安定性、立体配座の代表性、
記述子の外挿リスクを定量化し、機械学習モデルへの入力として
信頼度スコアも特徴量化可能にする。

既存モジュールへの影響: なし（完全新規）
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceReport:
    """
    1分子の計算信頼度レポート。

    Attributes:
        convergence_score: xTB計算の収束性スコア (0-1)。
            1.0 = opt正常完了, 0.7 = sp fallback, 0.0 = 全失敗。
        electronic_stability: 電子状態の安定性スコア (0-1)。
            HOMO-LUMOギャップが十分に開いていれば高い。
        charge_consistency: 電荷保存の一貫性スコア (0-1)。
            Mulliken電荷合計が形式電荷に近ければ高い。
        descriptor_completeness: 記述子の完全性スコア (0-1)。
            NaN/欠損がなければ 1.0。
        conformer_representativeness: 立体配座の代表性 (0-1)。
            複数conformerのエネルギー分散が小さければ高い。
        overall_confidence: 総合信頼度スコア (0-1)。
        warnings: 警告メッセージのリスト。
    """
    convergence_score: float = 1.0
    electronic_stability: float = 1.0
    charge_consistency: float = 1.0
    descriptor_completeness: float = 1.0
    conformer_representativeness: float = 1.0
    overall_confidence: float = 1.0
    warnings: list[str] = field(default_factory=list)

    def to_features(self, prefix: str = "conf_") -> dict[str, float]:
        """信頼度スコアをML特徴量として返す。"""
        return {
            f"{prefix}convergence": self.convergence_score,
            f"{prefix}electronic_stability": self.electronic_stability,
            f"{prefix}charge_consistency": self.charge_consistency,
            f"{prefix}descriptor_completeness": self.descriptor_completeness,
            f"{prefix}conformer_repr": self.conformer_representativeness,
            f"{prefix}overall": self.overall_confidence,
        }


class UncertaintyEstimator:
    """
    xTB計算結果の信頼度を多角的に評価するクラス。

    使い方::

        estimator = UncertaintyEstimator()
        report = estimator.evaluate(xtb_result_dict, formal_charge=0)
        print(report.overall_confidence)  # 0.0 ~ 1.0
        features = report.to_features()   # ML特徴量として利用可能
    """

    # HOMO-LUMOギャップの安定性閾値 (eV)
    # 参考: 有機分子の典型的HLG: 3-10 eV, 不安定系: <1 eV
    MIN_SAFE_HL_GAP: float = 0.5   # eV — これ未満は電子状態不安定の警告
    IDEAL_HL_GAP: float = 2.0      # eV — これ以上ならフル信頼

    # 電荷合計の許容誤差
    CHARGE_TOLERANCE: float = 0.1  # 電子

    def evaluate(
        self,
        xtb_result: dict[str, Any],
        formal_charge: int = 0,
        calc_type_used: str | None = None,
        convergence_fallback: bool = False,
        conformer_energies: list[float] | None = None,
    ) -> ConfidenceReport:
        """
        xTB計算結果の信頼度を総合評価する。

        Args:
            xtb_result: xtb_adapter._parse_xtb_output() の出力辞書。
            formal_charge: 分子の形式電荷。
            calc_type_used: 実際に使用された計算タイプ ("opt" or "sp")。
                None の場合は判定をスキップ。
            convergence_fallback: 収束基準の緩和やsp fallbackが発生したか。
            conformer_energies: 複数conformerのエネルギーリスト (Hartree)。
                None の場合は conformer_representativeness を 1.0 とする。

        Returns:
            ConfidenceReport インスタンス。
        """
        report = ConfidenceReport()

        # 1. 収束性
        report.convergence_score = self._eval_convergence(
            xtb_result, calc_type_used, convergence_fallback,
        )
        if report.convergence_score < 1.0:
            report.warnings.append(
                f"収束性スコア低下: {report.convergence_score:.2f}"
            )

        # 2. 電子状態の安定性 (HOMO-LUMOギャップ)
        report.electronic_stability = self._eval_electronic_stability(xtb_result)
        if report.electronic_stability < 0.5:
            report.warnings.append(
                "HOMO-LUMOギャップが小さい: 電子状態が不安定な可能性"
            )

        # 3. 電荷の一貫性
        report.charge_consistency = self._eval_charge_consistency(
            xtb_result, formal_charge,
        )
        if report.charge_consistency < 0.8:
            report.warnings.append(
                "Mulliken電荷合計と形式電荷に不整合"
            )

        # 4. 記述子の完全性
        report.descriptor_completeness = self._eval_descriptor_completeness(xtb_result)
        if report.descriptor_completeness < 1.0:
            n_missing = sum(
                1 for v in xtb_result.values()
                if v is None or (isinstance(v, float) and math.isnan(v))
            )
            report.warnings.append(f"記述子に{n_missing}件の欠損あり")

        # 5. 立体配座の代表性
        report.conformer_representativeness = self._eval_conformer_representativeness(
            conformer_energies,
        )

        # 6. 総合スコア（重み付き幾何平均）
        weights = {
            "convergence": 3.0,
            "electronic": 2.0,
            "charge": 1.0,
            "completeness": 2.0,
            "conformer": 1.5,
        }
        scores = [
            report.convergence_score,
            report.electronic_stability,
            report.charge_consistency,
            report.descriptor_completeness,
            report.conformer_representativeness,
        ]
        w_list = list(weights.values())
        # 重み付き幾何平均: exp(Σ w_i * ln(s_i) / Σ w_i)
        total_w = sum(w_list)
        log_sum = sum(
            w * math.log(max(s, 1e-10))
            for w, s in zip(w_list, scores)
        )
        report.overall_confidence = math.exp(log_sum / total_w)

        return report

    def batch_evaluate(
        self,
        xtb_results: list[dict[str, Any]],
        formal_charges: list[int] | None = None,
        **kwargs: Any,
    ) -> list[ConfidenceReport]:
        """
        複数分子のxTB結果をバッチ評価する。

        Returns:
            ConfidenceReport のリスト。
        """
        reports = []
        for i, result in enumerate(xtb_results):
            charge = formal_charges[i] if formal_charges else 0
            try:
                report = self.evaluate(result, formal_charge=charge, **kwargs)
            except Exception as e:
                logger.warning("信頼度評価エラー (idx=%d): %s", i, e)
                report = ConfidenceReport(
                    overall_confidence=0.0,
                    warnings=[f"評価エラー: {e}"],
                )
            reports.append(report)
        return reports

    # ────────────────────────────────────────────────────────
    # 個別評価ヘルパー
    # ────────────────────────────────────────────────────────

    def _eval_convergence(
        self,
        xtb_result: dict,
        calc_type_used: str | None,
        convergence_fallback: bool,
    ) -> float:
        """計算収束性を 0-1 で評価する。"""
        score = 1.0

        # 計算結果が空ならゼロ
        if not xtb_result:
            return 0.0

        # 必須項目の存在チェック
        essential = ["xtb_TotalEnergy", "xtb_HomoEnergy", "xtb_LumoEnergy"]
        present = sum(1 for k in essential if k in xtb_result and xtb_result[k] is not None)
        if present < len(essential):
            score *= present / len(essential)

        # sp へのフォールバックがあった場合
        if calc_type_used == "sp" and convergence_fallback:
            score *= 0.7  # opt失敗→sp fallback

        # 収束基準の緩和があった場合
        if convergence_fallback:
            score *= 0.85

        return min(max(score, 0.0), 1.0)

    def _eval_electronic_stability(self, xtb_result: dict) -> float:
        """
        HOMO-LUMOギャップに基づく電子状態安定性を評価する。

        化学的根拠:
        - HLG < 0.5 eV: 多配置性が強く、GFN2-xTBの精度が低下する可能性
        - HLG > 2.0 eV: 通常の閉殻分子、xTBの信頼度が高い
        - 0.5 < HLG < 2.0: 線形補間

        参考: Bannwarth et al., JCTC 2019 — GFN2-xTBの適用範囲
        """
        gap = xtb_result.get("xtb_HomoLumoGap")
        if gap is None:
            # ギャップ不明 → 中間スコア
            return 0.5

        try:
            gap = float(gap)
        except (TypeError, ValueError):
            return 0.5

        if gap < 0:
            return 0.1  # 負のギャップ → 非常に不安定

        if gap >= self.IDEAL_HL_GAP:
            return 1.0
        elif gap <= self.MIN_SAFE_HL_GAP:
            return max(0.1, gap / self.MIN_SAFE_HL_GAP * 0.3)
        else:
            # 線形補間
            return 0.3 + 0.7 * (gap - self.MIN_SAFE_HL_GAP) / (
                self.IDEAL_HL_GAP - self.MIN_SAFE_HL_GAP
            )

    def _eval_charge_consistency(
        self,
        xtb_result: dict,
        formal_charge: int,
    ) -> float:
        """
        Mulliken電荷の合計値が形式電荷に一致するか評価する。

        xTBのMulliken電荷は基底関数の割り当てに依存するため、
        合計が形式電荷から大きくずれる場合は計算に問題がある可能性。
        """
        q_mean = xtb_result.get("xtb_MullikenChargeMean")
        if q_mean is None:
            return 1.0  # 電荷情報なし → 評価不能（ペナルティなし）

        # meanは1原子あたりの平均 → 合計は概算
        # ここでは mean が 0 に近いこと（中性分子の場合）を確認
        try:
            deviation = abs(float(q_mean) - (formal_charge * 0.01))
        except (TypeError, ValueError):
            return 1.0

        if deviation < self.CHARGE_TOLERANCE:
            return 1.0
        elif deviation < self.CHARGE_TOLERANCE * 5:
            return 1.0 - (deviation - self.CHARGE_TOLERANCE) / (
                self.CHARGE_TOLERANCE * 4
            )
        else:
            return 0.3

    @staticmethod
    def _eval_descriptor_completeness(xtb_result: dict) -> float:
        """記述子の完全性（NaN/None の割合）を評価する。"""
        if not xtb_result:
            return 0.0

        total = len(xtb_result)
        valid = sum(
            1 for v in xtb_result.values()
            if v is not None and not (isinstance(v, float) and math.isnan(v))
        )
        return valid / total if total > 0 else 0.0

    @staticmethod
    def _eval_conformer_representativeness(
        energies: list[float] | None,
    ) -> float:
        """
        複数conformerのエネルギー分散に基づく代表性を評価する。

        化学的根拠:
        - エネルギー分散が小さい → 分子が剛直 → 単一構造で代表性が高い
        - エネルギー分散が大きい → 柔軟な分子 → 単一構造の代表性が低い

        閾値: kT at 298K ≈ 0.6 kcal/mol ≈ 0.001 Hartree
        """
        if energies is None or len(energies) < 2:
            return 1.0  # 情報不足 → ペナルティなし

        try:
            arr = np.array([float(e) for e in energies])
        except (TypeError, ValueError):
            return 1.0

        std_hartree = float(np.std(arr))

        # kT at 298K ≈ 0.001 Hartree
        kT = 0.001
        if std_hartree < kT:
            return 1.0  # 全conformerが熱エネルギー以内 → 高代表性
        elif std_hartree < kT * 10:
            return 1.0 - 0.5 * (std_hartree - kT) / (kT * 9)
        else:
            return max(0.2, 0.5 - 0.3 * (std_hartree - kT * 10) / (kT * 50))
