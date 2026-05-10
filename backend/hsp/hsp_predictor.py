# -*- coding: utf-8 -*-
"""
backend/hsp/hsp_predictor.py

SMILES → HSP (δD, δP, δH) 予測モジュール。

van Krevelen/Hoftyzer Group Contribution 法を使用して
任意のSMILES分子のHansen Solubility Parametersを計算する。

Implements: Hoftyzer-van Krevelen Group Contribution 法
引用:
  - van Krevelen, D.W. & Te Nijenhuis, K.
    "Properties of Polymers", 4th Ed., Elsevier, 2009, Table 4.3
  - Hansen, C.M. "Hansen Solubility Parameters: A User's Handbook",
    2nd Ed., CRC Press, 2007

計算式:
  δD = ΣFdi / V
  δP = √(ΣFpi²) / V
  δH = √(ΣEhi / V)
  V  = ΣVi

  Fdi: 分散力の基団寄与 (J^½·cm^(3/2)/mol)
  Fpi: 極性力の基団寄与 (J^½·cm^(3/2)/mol)
  Ehi: 水素結合エネルギー寄与 (J/mol)
  Vi:  モル体積寄与 (cm³/mol)

API:
    predict(smiles) → {"delta_d", "delta_p", "delta_h", "method", "confidence"}
    predict_batch(smiles_list) → pd.DataFrame
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Hoftyzer-van Krevelen Group Contribution テーブル
# van Krevelen & Te Nijenhuis, 2009, Table 4.3
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class GroupContribution:
    """1つの官能基のHSP寄与パラメータ。

    Implements: van Krevelen 2009, Table 4.3
    """
    name: str       # 基団名（日本語）
    smarts: str     # RDKit SMARTSパターン
    fdi: float      # 分散力寄与 Fdi (J^½·cm^(3/2)/mol)
    fpi: float      # 極性力寄与 Fpi (J^½·cm^(3/2)/mol)
    ehi: float      # 水素結合エネルギー Ehi (J/mol)
    vi: float       # モル体積寄与 Vi (cm³/mol)


# ── 基団テーブル（SMARTSの特異度が高い順に配置）──
# 注意: マッチ順序が重要。より特異的なパターンを先に配置。
# 引用: van Krevelen & Te Nijenhuis, "Properties of Polymers", 4th Ed., 2009, Table 4.3
# 原文: "Group contributions to the solubility parameters according to Hoftyzer-Van Krevelen"
GROUP_TABLE: list[GroupContribution] = [
    # ═══ 多原子基（先にマッチさせる）═══
    GroupContribution(
        name="カルボキシル基 -COOH",
        smarts="[CX3](=O)[OX2H1]",
        fdi=530, fpi=420, ehi=10000, vi=28.5,
    ),
    GroupContribution(
        name="エステル -COO-",
        smarts="[CX3](=O)[OX2H0]",
        fdi=390, fpi=490, ehi=7000, vi=18.0,
    ),
    GroupContribution(
        name="アルデヒド -CHO",
        smarts="[CH1X3]=O",
        fdi=470, fpi=800, ehi=4500, vi=21.4,
    ),
    GroupContribution(
        name="アミド -CONH-",
        smarts="[CX3](=O)[NX3H1]",
        fdi=490, fpi=1030, ehi=10000, vi=9.5,
    ),
    GroupContribution(
        name="シアノ基 -C≡N",
        smarts="[CX2]#[NX1]",
        fdi=430, fpi=1100, ehi=2500, vi=24.0,
    ),
    GroupContribution(
        name="ニトロ基 -NO₂",
        smarts="[NX3+](=O)[O-]",
        fdi=500, fpi=1070, ehi=1500, vi=24.0,
    ),

    # ═══ 水素結合基 ═══
    GroupContribution(
        name="水 H2O",
        smarts="[OX2H2]",
        fdi=210, fpi=500, ehi=20000, vi=14.0,
    ),
    GroupContribution(
        name="ヒドロキシル基 -OH",
        smarts="[OX2H1;!$([OX2H1][CX3]=O)]",
        fdi=210, fpi=500, ehi=20000, vi=10.0,
    ),
    GroupContribution(
        name="一級アミン -NH₂",
        smarts="[NX3H2;!$([NX3H2][CX3]=O)]",
        fdi=280, fpi=350, ehi=8400, vi=19.2,
    ),
    GroupContribution(
        name="二級アミン -NH-",
        smarts="[NX3H1;!$(N=*);!$([NX3H1][CX3]=O)]",
        fdi=160, fpi=210, ehi=3100, vi=4.5,
    ),
    GroupContribution(
        name="三級アミン -N<",
        smarts="[NX3H0;!$(N=*);!$(N#*)]",
        fdi=20, fpi=800, ehi=5000, vi=-9.0,
    ),
    GroupContribution(
        name="チオール -SH",
        smarts="[SX2H1]",
        fdi=315, fpi=0, ehi=3000, vi=28.0,
    ),

    # ═══ ヘテロ原子基 ═══
    GroupContribution(
        name="ケトン -C(=O)-",
        smarts="[CX3;!$([CX3][OX2H1]);!$([CX3][OX2H0][#6])](=O)[#6]",
        fdi=290, fpi=770, ehi=2000, vi=10.8,
    ),
    GroupContribution(
        name="エーテル -O-",
        smarts="[OX2H0;!$([OX2](=*));!$([OX2][CX3]=O)]",
        fdi=100, fpi=400, ehi=3000, vi=3.8,
    ),
    GroupContribution(
        name="チオエーテル -S-",
        smarts="[SX2H0]",
        fdi=225, fpi=0, ehi=0, vi=12.0,
    ),
    GroupContribution(
        name="リン -P-",
        smarts="[PX3]",
        fdi=225, fpi=0, ehi=0, vi=15.0,
    ),

    # ═══ ハロゲン ═══
    GroupContribution(
        name="フッ素 -F",
        smarts="[FX1]",
        fdi=220, fpi=0, ehi=0, vi=18.0,
    ),
    GroupContribution(
        name="塩素 -Cl",
        smarts="[ClX1]",
        fdi=450, fpi=550, ehi=400, vi=24.0,
    ),
    GroupContribution(
        name="臭素 -Br",
        smarts="[BrX1]",
        fdi=550, fpi=0, ehi=0, vi=30.0,
    ),
    GroupContribution(
        name="ヨウ素 -I",
        smarts="[IX1]",
        fdi=550, fpi=0, ehi=0, vi=31.5,
    ),

    # ═══ 不飽和炭素（非芳香族）═══
    GroupContribution(
        name="末端ビニル =CH₂",
        smarts="[CH2X3]=[*]",
        fdi=400, fpi=0, ehi=0, vi=28.5,
    ),
    GroupContribution(
        name="ビニル =CH-",
        smarts="[CH1X3]=[*]",
        fdi=200, fpi=0, ehi=0, vi=13.5,
    ),
    GroupContribution(
        name="ビニル =C<",
        smarts="[CH0X3](=[*])([#6])[#6]",
        fdi=70, fpi=0, ehi=0, vi=-5.5,
    ),

    # ═══ 芳香族炭素 ═══
    GroupContribution(
        name="芳香族CH",
        smarts="[cH1]",
        fdi=270, fpi=0, ehi=0, vi=16.1,
    ),
    GroupContribution(
        name="芳香族C（置換）",
        smarts="[c;H0;!$([c]=[!#6])]",
        fdi=70, fpi=0, ehi=0, vi=-5.5,
    ),

    # ═══ 飽和炭素（最も一般的 → 最後にマッチ）═══
    GroupContribution(
        name="メチル -CH₃",
        smarts="[CH3]",
        fdi=420, fpi=0, ehi=0, vi=33.5,
    ),
    GroupContribution(
        name="メチレン -CH₂-",
        smarts="[CH2X4]",
        fdi=270, fpi=0, ehi=0, vi=16.1,
    ),
    GroupContribution(
        name="メチン >CH-",
        smarts="[CH1X4]",
        fdi=80, fpi=0, ehi=0, vi=-1.0,
    ),
    GroupContribution(
        name="四級炭素 >C<",
        smarts="[CH0X4]",
        fdi=-70, fpi=0, ehi=0, vi=-19.2,
    ),
]


# ═══════════════════════════════════════════════════════════════
# SMILES → HSP 計算エンジン
# ═══════════════════════════════════════════════════════════════

def _compute_hsp_group_contribution(smiles: str) -> dict[str, float | str] | None:
    """
    Hoftyzer-van Krevelen Group Contribution法によるHSP計算。

    Implements: van Krevelen 2009, Table 4.3 + 式(4.6)-(4.8)
    引用: van Krevelen, D.W. & Te Nijenhuis, K.
          "Properties of Polymers", 4th Ed., Elsevier, 2009

    計算式:
      δD = ΣFdi / V          ... 式(4.6)
      δP = √(ΣFpi²) / V     ... 式(4.7)
      δH = √(ΣEhi / V)      ... 式(4.8)
      V  = ΣVi               (モル体積)

    精度: δD ±1.0, δP ±2.0, δH ±2.0 MPa^0.5 (文献値)

    Args:
        smiles: SMILES文字列

    Returns:
        {"delta_d", "delta_p", "delta_h", "molar_volume",
         "method", "confidence", "matched_groups", "unmatched_atoms"}
        計算失敗時はNone
    """
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.debug("無効なSMILES: %s", smiles[:50])
            return None

        # -- 既知化合物テーブル (Hansen 2007, Appendix A) --
        # GC法が不得意な小分子/特殊分子は文献値を直接返す
        canonical = Chem.MolToSmiles(mol)
        _KNOWN_HSP = {
            # canonical SMILES: (delta_d, delta_p, delta_h)
            "O":             (15.5, 16.0, 42.3),    # Water
            "ClC(Cl)Cl":     (17.8,  3.1,  5.7),    # Chloroform
            "Cl":            (15.8,  2.0,  0.2),    # HCl
            "ClCCl":         (16.4,  6.3,  3.0),    # DCM
            "ClC(Cl)(Cl)Cl": (15.8,  0.0,  0.0),    # CCl4
            "CS(C)=O":       (18.4, 16.4, 10.2),    # DMSO
            "CN(C)C=O":      (17.4, 13.7, 11.3),    # DMF
            "O=CO":          (15.6,  5.1,  8.4),    # Formic acid
            "N":             (15.5, 13.0,  5.2),    # NH3
            "CC(C)=O":       (15.5, 10.4,  7.0),    # Acetone
            "CO":            (15.1, 12.3, 22.3),    # Methanol
            "CCO":           (15.8,  8.8, 19.4),    # Ethanol
            "CC(C)O":        (15.1,  6.1, 16.4),    # IPA
            "CCCCCC":        (14.9,  0.0,  0.0),    # Hexane
            "c1ccccc1":      (18.4,  0.0,  2.0),    # Benzene
            "Cc1ccccc1":     (18.0,  1.4,  2.0),    # Toluene
        }
        if canonical in _KNOWN_HSP:
            d, p, h = _KNOWN_HSP[canonical]
            return {
                "delta_d": d, "delta_p": p, "delta_h": h,
                "delta_total": float(np.sqrt(d**2 + p**2 + h**2)),
                "molar_volume": None,
                "method": "known_compound",
                "confidence": "reference",
                "matched_groups": [],
                "unmatched_atoms": 0,
                "coverage": 1.0,
            }

        # 水素を付加して正確な原子数を得る
        mol = Chem.AddHs(mol)

        # 各原子のマッチ済みフラグ
        n_atoms = mol.GetNumAtoms()
        matched = [False] * n_atoms

        # グループ寄与の累積
        sum_fdi = 0.0
        sum_fpi2 = 0.0  # Fpi² の累積
        sum_ehi = 0.0
        sum_vi = 0.0
        matched_groups: list[str] = []

        for gc in GROUP_TABLE:
            try:
                pattern = Chem.MolFromSmarts(gc.smarts)
                if pattern is None:
                    logger.warning("無効なSMARTS: %s (%s)", gc.smarts, gc.name)
                    continue

                matches = mol.GetSubstructMatches(pattern)
                for match in matches:
                    # アンカー原子(match[0])がまだ未マッチの場合のみカウント
                    anchor = match[0]
                    if not matched[anchor]:
                        # 全マッチ原子をマッチ済みに(多原子基の二重カウント防止)
                        for idx in match:
                            atom = mol.GetAtomWithIdx(idx)
                            if atom.GetAtomicNum() > 1:  # 水素以外
                                matched[idx] = True
                        sum_fdi += gc.fdi
                        sum_fpi2 += gc.fpi ** 2
                        sum_ehi += gc.ehi
                        sum_vi += gc.vi
                        matched_groups.append(gc.name)
            except Exception as e:
                logger.debug("SMARTS '%s' マッチエラー: %s", gc.smarts, e)

        # 未マッチの重原子をカウント（水素はスキップ）
        unmatched_heavy = 0
        for i in range(n_atoms):
            atom = mol.GetAtomWithIdx(i)
            if atom.GetAtomicNum() > 1 and not matched[i]:
                unmatched_heavy += 1
                # フォールバック: 未知の重原子にはCH2相当の寄与を仮定
                sum_fdi += 270
                sum_vi += 16.1

        # モル体積が小さすぎる場合のガード
        if sum_vi < 5.0:
            # 最小限のモル体積推定（重原子数×16 cm³/mol）
            heavy_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
            sum_vi = max(sum_vi, heavy_count * 16.0)
            if sum_vi < 5.0:
                logger.debug("モル体積が小さすぎます: V=%.1f, SMILES=%s", sum_vi, smiles[:50])
                return None

        # ── HSP計算: van Krevelen 式(4.6)-(4.8) ──
        delta_d = sum_fdi / sum_vi
        delta_p = np.sqrt(sum_fpi2) / sum_vi
        delta_h = np.sqrt(max(sum_ehi / sum_vi, 0.0))

        # 物理的に妥当な範囲にクリップ (MPa^0.5)
        delta_d = float(np.clip(delta_d, 8.0, 30.0))
        delta_p = float(np.clip(delta_p, 0.0, 25.0))
        delta_h = float(np.clip(delta_h, 0.0, 45.0))

        # 信頼度判定
        n_heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
        coverage = (n_heavy - unmatched_heavy) / max(n_heavy, 1)
        if coverage >= 0.9:
            confidence = "high"
        elif coverage >= 0.7:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "delta_d": delta_d,
            "delta_p": delta_p,
            "delta_h": delta_h,
            "delta_total": float(np.sqrt(delta_d**2 + delta_p**2 + delta_h**2)),
            "molar_volume": float(sum_vi),
            "method": "van_krevelen_hoftyzer",
            "confidence": confidence,
            "matched_groups": matched_groups,
            "unmatched_atoms": unmatched_heavy,
            "coverage": float(coverage),
        }

    except ImportError:
        logger.warning("RDKitが利用できません。HSP計算にはRDKitが必要です。")
        return None
    except Exception as e:
        logger.debug("HSP計算エラー: %s, err=%s", smiles[:50], e)
        return None


# ═══════════════════════════════════════════════════════════════
# HSP予測クラス（公開API）
# ═══════════════════════════════════════════════════════════════

class HSPPredictor:
    """SMILES から HSP (δD, δP, δH) を予測する。

    van Krevelen/Hoftyzer Group Contribution法で計算。
    事前学習モデルがある場合はML予測を優先する。

    Implements: van Krevelen 2009, Table 4.3
    引用: van Krevelen, D.W. & Te Nijenhuis, K.
          "Properties of Polymers", 4th Ed., Elsevier, 2009

    API:
        predict(smiles) → {"delta_d", "delta_p", "delta_h", "method"}
        predict_batch(smiles_list) → pd.DataFrame
    """

    def __init__(self, model_path: str | Path | None = None):
        self._model_d: Any = None
        self._model_p: Any = None
        self._model_h: Any = None
        self._has_model = False

        if model_path and Path(model_path).exists():
            self._load_model(model_path)

    def _load_model(self, path: str | Path) -> None:
        """事前学習モデル読み込み（オプション）。"""
        try:
            import joblib
            data = joblib.load(path)
            self._model_d = data["model_d"]
            self._model_p = data["model_p"]
            self._model_h = data["model_h"]
            self._has_model = True
            logger.info("HSP予測モデル読み込み: %s", path)
        except Exception as e:
            logger.warning("HSP予測モデル読み込み失敗: %s", e)
            self._has_model = False

    @property
    def is_available(self) -> bool:
        """予測機能が利用可能か（RDKit必須）。"""
        try:
            from rdkit import Chem  # noqa: F401
            return True
        except ImportError:
            return False

    def predict(self, smiles: str) -> dict[str, float | str]:
        """
        HSP 予測（ML → Group Contribution法の優先順位）。

        Implements: van Krevelen 2009, 式(4.6)-(4.8)

        Args:
            smiles: SMILES文字列

        Returns:
            {"delta_d", "delta_p", "delta_h", "method", "confidence"}
        """
        # ML予測（事前学習モデルがある場合）
        if self._has_model:
            features = self._extract_features(smiles)
            if features is not None:
                X = features.reshape(1, -1)
                return {
                    "delta_d": float(self._model_d.predict(X)[0]),
                    "delta_p": float(self._model_p.predict(X)[0]),
                    "delta_h": float(self._model_h.predict(X)[0]),
                    "method": "ml_prediction",
                    "confidence": "high",
                }

        # Group Contribution法
        result = _compute_hsp_group_contribution(smiles)
        if result is not None:
            return result

        raise ValueError(f"HSP予測失敗: {smiles[:50]}")

    def predict_batch(self, smiles_list: list[str]) -> pd.DataFrame:
        """バッチ予測。複数SMILES → DataFrame。"""
        results = []
        for smi in smiles_list:
            try:
                hsp = self.predict(smi)
                hsp["smiles"] = smi
                hsp["error"] = None
                results.append(hsp)
            except Exception as e:
                results.append({
                    "smiles": smi,
                    "delta_d": None,
                    "delta_p": None,
                    "delta_h": None,
                    "method": None,
                    "confidence": None,
                    "error": str(e),
                })

        return pd.DataFrame(results)

    def save_model(self, path: str | Path) -> None:
        """学習済みモデル保存。"""
        if not self._has_model:
            raise ValueError("モデル未学習")
        import joblib
        joblib.dump({
            "model_d": self._model_d,
            "model_p": self._model_p,
            "model_h": self._model_h,
        }, path)
        logger.info("HSP予測モデル保存: %s", path)

    @staticmethod
    def _extract_features(smiles: str) -> np.ndarray | None:
        """RDKit記述子をML用特徴量として抽出。"""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, rdMolDescriptors

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            features = np.array([
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                float(rdMolDescriptors.CalcNumHDonors(mol)),
                float(rdMolDescriptors.CalcNumHAcceptors(mol)),
                float(Descriptors.NumRotatableBonds(mol)),
                float(Descriptors.RingCount(mol)),
                float(rdMolDescriptors.CalcNumAromaticRings(mol)),
                float(rdMolDescriptors.CalcNumAliphaticRings(mol)),
                Descriptors.FractionCSP3(mol),
                float(rdMolDescriptors.CalcNumHeavyAtoms(mol)),
                float(mol.GetNumAtoms()),
                Descriptors.MolMR(mol),
                Descriptors.LabuteASA(mol),
                Descriptors.BalabanJ(mol) if mol.GetNumBonds() > 0 else 0.0,
            ])
            return features
        except Exception as e:
            logger.debug("RDKit特徴量抽出失敗: %s, err=%s", smiles[:30], e)
            return None
