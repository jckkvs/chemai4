"""
backend/chem/psmiles_adapter.py

重合体表記（Polymer SMILES, PSMILES）を解析し、
モノマー単位からポリマー特有の特徴量やベースライン記述子を計算するモジュール。
"""
from __future__ import annotations

import logging
from typing import ClassVar

import pandas as pd
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    _RDKIT_AVAILABLE = True
except ImportError:
    Chem = None  # type: ignore[assignment]
    Descriptors = None  # type: ignore[assignment]
    rdMolDescriptors = None  # type: ignore[assignment]
    _RDKIT_AVAILABLE = False

from backend.chem.base import BaseChemAdapter, DescriptorResult

logger = logging.getLogger(__name__)


class PSmilesAdapter(BaseChemAdapter):
    """
    PSMILES（ポリマーSMILES）から特徴量を計算するアダプタ。
    
    基本戦略:
    1. 文字列に `*` または `[*]` が含まれるかを判定する（`is_psmiles`）。
    2. PSMILES の場合、結合点（*）をダミー原子またはメチル基に置換、
       あるいはループ化させるなどして RDKit で扱えるモノマー部分構造モデルを作る。
    3. ポリマー向けの基本記述子（分子量やHBA/HBDの近似）および
       モノマー単位のMorgan FPを計算し、重合体の特性予測特徴量とする。
    
    psmilesライブラリがインストールされている場合はネイティブ機能を、
    無い場合は上記のようなRDKitベースの簡易的な近似処理を提供する。
    """
    name: ClassVar[str] = "psmiles"
    _available: bool | None = None
    _has_psmiles_lib: bool | None = None

    @classmethod
    def is_available(cls) -> bool:
        if cls._available is None:
            # 常に有効（フォールバックでRDKitを使用するため）
            try:
                import rdkit
                cls._available = True
            except ImportError:
                cls._available = False
        return cls._available
        
    @classmethod
    def has_psmiles_lib(cls) -> bool:
        if cls._has_psmiles_lib is None:
            try:
                import psmiles
                cls._has_psmiles_lib = True
            except ImportError:
                cls._has_psmiles_lib = False
        return cls._has_psmiles_lib

    @staticmethod
    def is_psmiles(smiles: str) -> bool:
        """SMILES文字列がPSMILES（ポリマー）表記かを判定する。"""
        if not isinstance(smiles, str):
            return False
        return "*" in smiles

    @staticmethod
    def _fallback_process_psmiles(smiles: str) -> Chem.Mol | None:
        """
        psmilesライブラリがない場合、RDKitで無理やりモノマーの情報を抽出するため
        `*` をダミー原子([At]等)または特定の原子に置換してMolオブジェクトを作る。
        """
        # 単純に [*] をメチル基 [CH3] に置換して近似する（末端基として最も自然でパースエラーになりにくい）
        # RDKitのパースエラーを避けるため、枝分かれなどの影響を最小限にする
        import re
        
        # [*:1] などのアイソトープ付きダミーアトム表記も [CH3] に置換
        clean_smi = re.sub(r'\[\*\:\d+\]', '[CH3]', smiles)
        clean_smi = clean_smi.replace("[*]", "[CH3]").replace("*", "[CH3]")
        
        mol = Chem.MolFromSmiles(clean_smi)
        if mol is not None:
            return mol
            
        # [CH3]置換でもダメな場合は、[*] を単に削除する（水素化の近似）
        # ただし "()" のような空のブランチが残るとパースエラーになるため、空のブランチも削除する
        clean_smi = smiles.replace("[*]", "").replace("*", "")
        clean_smi = clean_smi.replace("()", "")
        
        # 連続結合等の不整合があれば警告を出してNoneを返す
        mol = Chem.MolFromSmiles(clean_smi)
        return mol

    @property
    def description(self) -> str:
        return "ポリマーSMILES (PSMILES) 用の特徴量抽出（RDKit近似フォールバック付き）。"

    @classmethod
    def get_info(cls) -> dict:
        return {
            "name": cls.name,
            "version": "1.0",
            "description": cls.description,
            "is_available": cls.is_available(),
            "has_psmiles_lib": cls.has_psmiles_lib(),
        }

    def compute(self, smiles_list: list[str], **kwargs) -> DescriptorResult:
        """
        PSMILESを受け取り、モノマー近似に基づく記述子群を返す。
        """
        self._require_available()
        logger.info(f"🧪 PSMILES（ポリマー）記述子計算を開始します（{len(smiles_list)}分子）...")

        results = []
        failed_indices = []
        
        for idx, smi in enumerate(smiles_list):
            if not isinstance(smi, str) or not smi.strip():
                failed_indices.append(idx)
                results.append({})
                continue

            try:
                # 1. PSMILES解析
                if self.has_psmiles_lib():
                    # psmiles特有の処理（現状はプレースホルダでフォールバックへ流す）
                    pass
                
                # フォールバック（RDKitによるモノマー近似処理）
                mol = self._fallback_process_psmiles(smi)
                if mol is None:
                    failed_indices.append(idx)
                    results.append({})
                    continue
                
                # 2. モノマーとしての基礎記述子算出
                desc = {
                    "PSMILES_MonomerWt": Descriptors.MolWt(mol),
                    "PSMILES_NumHDonors": rdMolDescriptors.CalcNumHBD(mol),
                    "PSMILES_NumHAcceptors": rdMolDescriptors.CalcNumHBA(mol),
                    "PSMILES_MolLogP": Descriptors.MolLogP(mol),
                    "PSMILES_FractionCSP3": Descriptors.FractionCSP3(mol),
                }
                
                # 3. モノマーとしてのMorgan Fingerprint (近似部分構造)
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
                # 高頻度のビットのみ
                for i, bit in enumerate(fp):
                    if bit:
                        desc[f"PSMILES_Morgan_{i}"] = 1.0

                results.append(desc)
                
            except Exception as e:
                logger.debug(f"[{self.name}] '{smi}' の計算に失敗: {e}")
                failed_indices.append(idx)
                results.append({})

        # DataFrame化してゼロ埋め
        df = pd.DataFrame(results).fillna(0.0)
        
        # 存在しないカラム対策（全行失敗などの場合）
        if df.empty:
            df["PSMILES_MonomerWt"] = 0.0

        return DescriptorResult(
            descriptors=df,
            smiles_list=smiles_list,
            failed_indices=failed_indices,
            adapter_name=self.name,
            metadata={"has_psmiles_lib": self.has_psmiles_lib()}
        )

