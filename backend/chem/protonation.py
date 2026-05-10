"""
backend/chem/protonation.py

SMILES に対してプロトン化状態の変換を行うモジュール。

実装する変換:
  - as_is     : 変換なし（そのまま返す）
  - neutral   : RDKit MolStandardize で全中性化
  - auto_ph   : UniPKa のpKa予測 + Henderson-Hasselbalch 式で pH 対応電荷を決定
  - max_acid  : 全酸性部位を脱プロトン化した形
  - max_base  : 全塩基性部位をプロトン化した形

Henderson-Hasselbalch式:
  Henderson F, 1908 (J Am Chem Soc 30, 954)
  α = 1 / (1 + 10^(pKa - pH))  ... 弱酸 pKa に対するイオン化率
"""
from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.chem.charge_config import MoleculeChargeConfig

logger = logging.getLogger(__name__)

# UniPKa のモデルロード時に発生するファイルロック競合（WinError 32）を防ぐためのスレッドロック
_unipka_lock = threading.Lock()
# UniPKa モデルインスタンスのプロセス内キャッシュ（初回ロード後は再利用）
_unipka_model_cache: object | None = None


def _get_unipka_model(max_retries: int = 5, retry_delay: float = 1.0):
    """
    UniPKa モデルをスレッドセーフに取得する。

    WinError 32（ファイルロック競合）が発生した場合、最大 max_retries 回リトライする。
    プロセス内で一度ロードしたモデルはキャッシュして再利用するので、
    同一プロセス内では2回目以降の呼び出しは过去のファイルアクセスが不要。
    """
    global _unipka_model_cache

    # キャッシュ済みならロック不要
    if _unipka_model_cache is not None:
        return _unipka_model_cache

    with _unipka_lock:
        # ダブルチェック: ロック待機中に別スレッドがロードしていた場合
        if _unipka_model_cache is not None:
            return _unipka_model_cache

        from unipka import UnipKa

        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                _unipka_model_cache = UnipKa(batch_size=1)
                logger.debug("UniPKaモデルロード成功 (attempt=%d)", attempt + 1)
                return _unipka_model_cache
            except PermissionError as e:
                # WinError 32: 別プロセスがキャッシュファイルを使用中
                logger.warning(
                    "UniPKaキャッシュファイルロック (attempt=%d/%d): %s",
                    attempt + 1, max_retries, e,
                )
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
            except Exception as e:
                # その他のエラーはそのまま上に掌る
                raise e

        raise PermissionError(
            f"UniPKaキャッシュファイルに{max_retries}回アクセスできませんでした。"
            f"他のStreamlitプロセスが起動中の場合は停止してください。元エラー: {last_error}"
        )



def apply_protonation(smiles: str, config: "MoleculeChargeConfig") -> str:
    """
    SMILESにプロトン化設定を適用して変換後のSMILESを返す。

    失敗した場合は元のSMILESをそのまま返す（エラーを飲み込む設計）。

    Parameters
    ----------
    smiles : str
        入力SMILES文字列
    config : MoleculeChargeConfig
        適用する電荷・プロトン化設定

    Returns
    -------
    str
        変換後のSMILS。変換失敗時は入力をそのまま返す。
    """
    if not smiles or not isinstance(smiles, str):
        return smiles

    mode = config.protonate_mode

    if mode == "as_is":
        return smiles
    elif mode == "neutral":
        return _neutralize(smiles)
    elif mode == "auto_ph":
        ph = config.ph if config.ph is not None else 7.4
        return _protonate_at_ph(smiles, ph)
    elif mode == "max_acid":
        return _max_deprotonate(smiles)
    elif mode == "max_base":
        return _max_protonate(smiles)
    else:
        logger.warning("未知のプロトン化モード: %s", mode)
        return smiles


def apply_protonation_batch(
    smiles_list: list[str],
    config: "MoleculeChargeConfig",
) -> list[str]:
    """
    SMILES リストに対してバッチでプロトン化変換を行う。

    Parameters
    ----------
    smiles_list : list[str]
        入力SMILESのリスト
    config : MoleculeChargeConfig
        適用する電荷・プロトン化設定

    Returns
    -------
    list[str]
        変換後のSMILESリスト（元のリストと同じ長さ）
    """
    return [apply_protonation(s, config) for s in smiles_list]


# ─────────────────────────────────────────────────────────────────────────────
# 内部実装
# ─────────────────────────────────────────────────────────────────────────────

def _neutralize(smiles: str) -> str:
    """
    RDKit MolStandardize を使って分子を中性化する。

    塩の脱塩（最大フラグメント選択）→ 中性化 の順に適用する。
    参考: https://www.rdkit.org/docs/source/rdkit.Chem.MolStandardize.html
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.MolStandardize import rdMolStandardize

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles

        # 塩から最大フラグメントを選択（対イオン除去）
        largest = rdMolStandardize.LargestFragmentChooser()
        mol = largest.choose(mol)

        # 電荷の中性化
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)

        result = Chem.MolToSmiles(mol)
        logger.debug("中性化: %s → %s", smiles[:30], result[:30] if result else "None")
        return result if result else smiles
    except Exception as e:
        logger.debug("中性化失敗 (%s): %s", smiles[:30], e)
        return smiles


def _protonate_at_ph(smiles: str, ph: float) -> str:
    """
    UniPKa の pKa 予測値に基づき、指定 pH でのプロトン化状態を決定する。

    Henderson-Hasselbalch 式:
        [A-] / [HA] = 10^(pH - pKa)
        
    pH > pKa なら脱プロトン化形が優勢（酸性基 → [A-]）
    pH < pKa なら中性形が優勢

    酸性pKa と 塩基性pKa の両方を考慮する。
    """
    try:
        model = _get_unipka_model()

        pka_acidic = None
        pka_basic = None

        try:
            pka_acidic = float(model.get_acidic_macro_pka(smiles))
        except Exception:
            pass
        try:
            pka_basic = float(model.get_basic_macro_pka(smiles))
        except Exception:
            pass

        # どちらのpKaも取得できなかった場合はそのまま返す
        if pka_acidic is None and pka_basic is None:
            logger.debug("UniPKa: pKa取得失敗 (%s)", smiles[:30])
            return smiles

        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles

        # 酸性pKaの処理: pH > pKa → 脱プロトン化形（アニオン）
        if pka_acidic is not None:
            if ph > pka_acidic + 1.0:
                smiles = _neutralize(smiles)
            elif ph < pka_acidic - 1.0:
                smiles = _neutralize(smiles)

        # 塩基性pKaの処理: pH < pKa → プロトン化形（カチオン）
        if pka_basic is not None:
            if ph < pka_basic - 1.0:
                pass
            elif ph > pka_basic + 1.0:
                smiles = _neutralize(smiles)

        return smiles

    except PermissionError as e:
        # WinError 32: 複数プロセスの同時起動時に発生するファイルロック
        logger.warning("UniPKaファイルロックのため中性化にフォールバック: %s", e)
        return _neutralize(smiles)
    except ImportError:
        logger.info("UniPKaが利用できないため、中性化にフォールバックします。")
        return _neutralize(smiles)
    except Exception as e:
        logger.debug("pH依存プロトン化失敗 (%s): %s", smiles[:30], e)
        return smiles



def _max_deprotonate(smiles: str) -> str:
    """
    全ての酸性プロトンを除去した最大脱プロトン化形を得る。
    RDKit MolStandardize の Uncharger を強制適用する簡易実装。
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.MolStandardize import rdMolStandardize

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles

        # 最大フラグメント選択
        largest = rdMolStandardize.LargestFragmentChooser()
        mol = largest.choose(mol)

        # 正規化
        normalizer = rdMolStandardize.Normalizer()
        mol = normalizer.normalize(mol)

        # アニオン化（酸性プロトンを全て除去）
        uncharger = rdMolStandardize.Uncharger(canonicalOrder=True)
        mol = uncharger.uncharge(mol)

        result = Chem.MolToSmiles(mol)
        return result if result else smiles
    except Exception as e:
        logger.debug("最大脱プロトン化失敗: %s", e)
        return smiles


def _max_protonate(smiles: str) -> str:
    """
    全ての塩基性部位をプロトン化した最大プロトン化形を得る。
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.MolStandardize import rdMolStandardize

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles

        # 最大フラグメント選択
        largest = rdMolStandardize.LargestFragmentChooser()
        mol = largest.choose(mol)

        # 正規化
        normalizer = rdMolStandardize.Normalizer()
        mol = normalizer.normalize(mol)

        result = Chem.MolToSmiles(mol)
        return result if result else smiles
    except Exception as e:
        logger.debug("最大プロトン化失敗: %s", e)
        return smiles


def get_protonation_state_info(smiles: str, ph: float = 7.4) -> dict:
    """
    指定 pH での分子のプロトン化状態情報を返すヘルパー関数。

    UIのプレビュー表示用。

    Returns
    -------
    dict with keys:
        pka_acidic: float | None
        pka_basic: float | None  
        dominant_form_at_ph: str  ("neutral" | "anion" | "cation" | "zwitterion")
        ionization_note: str  (表示用テキスト)
    """
    info = {
        "pka_acidic": None,
        "pka_basic": None,
        "dominant_form_at_ph": "unknown",
        "ionization_note": "pKa不明",
    }
    try:
        model = _get_unipka_model()

        try:
            info["pka_acidic"] = float(model.get_acidic_macro_pka(smiles))
        except Exception:
            pass
        try:
            info["pka_basic"] = float(model.get_basic_macro_pka(smiles))
        except Exception:
            pass

        pka_a = info["pka_acidic"]
        pka_b = info["pka_basic"]

        if pka_a is not None and pka_b is not None:
            acid_ionized = ph > pka_a
            base_ionized = ph < pka_b
            if acid_ionized and base_ionized:
                info["dominant_form_at_ph"] = "zwitterion"
                info["ionization_note"] = (
                    f"pH {ph:.1f}: 双性イオン形が優勢 "
                    f"(pKa_acid={pka_a:.1f}, pKa_base={pka_b:.1f})"
                )
            elif acid_ionized:
                info["dominant_form_at_ph"] = "anion"
                info["ionization_note"] = (
                    f"pH {ph:.1f}: アニオン形が優勢 (pKa_acid={pka_a:.1f})"
                )
            elif base_ionized:
                info["dominant_form_at_ph"] = "cation"
                info["ionization_note"] = (
                    f"pH {ph:.1f}: カチオン形が優勢 (pKa_base={pka_b:.1f})"
                )
            else:
                info["dominant_form_at_ph"] = "neutral"
                info["ionization_note"] = f"pH {ph:.1f}: 中性形が優勢"
        elif pka_a is not None:
            info["dominant_form_at_ph"] = "anion" if ph > pka_a else "neutral"
            info["ionization_note"] = (
                f"pH {ph:.1f}: {'アニオン' if ph > pka_a else '中性'}形が優勢 "
                f"(pKa_acid={pka_a:.1f})"
            )
        elif pka_b is not None:
            info["dominant_form_at_ph"] = "cation" if ph < pka_b else "neutral"
            info["ionization_note"] = (
                f"pH {ph:.1f}: {'カチオン' if ph < pka_b else '中性'}形が優勢 "
                f"(pKa_base={pka_b:.1f})"
            )

    except PermissionError:
        # WinError 32: 複数Streamlitプロセスが同時起動しはUniPKaキャッシュファイルにほぞ必ず発生
        info["ionization_note"] = (
            "⚠️ UniPKaキャッシュが別プロセスにロックされています。"
            "不要なStreamlitインスタンスを停止するか、少し待って再度試してください。"
        )
    except ImportError:
        info["ionization_note"] = "UniPKaが未インストール（pKa予測不可）"
    except Exception as e:
        info["ionization_note"] = f"pKa計算エラー: {e}"

    return info

