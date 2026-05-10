# -*- coding: utf-8 -*-
"""
backend/chem/mol2vec_adapter.py

Mol2Vec アダプタ。
Word2Vec インスパイアの分子埋め込み。Morgan サブ構造を "単語" として扱い、
分子を固定長ベクトル（デフォルト300次元）に変換する。

参考:
  Jaeger et al. (2018) "Mol2vec: Unsupervised Machine Learning Approach
  with Chemical Intuition" J. Chem. Inf. Model., 58, 27-35.
  pip install mol2vec
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from backend.chem.base import BaseChemAdapter, DescriptorMetadata, DescriptorResult

logger = logging.getLogger(__name__)


class Mol2VecAdapter(BaseChemAdapter):
    """
    Mol2Vec アダプタ。

    事前学習済みモデルを使用して SMILES を 300 次元ベクトルに変換。
    学習型記述子のため、化学的直感に基づく潜在空間を提供。
    """

    def __init__(self, model_path: str | None = None, radius: int = 1):
        """
        Args:
            model_path: 事前学習済み Mol2Vec モデルのパス（None でデフォルト）
            radius: Morgan fingerprint の半径
        """
        self._model_path = model_path
        self._radius = radius
        self._model = None

    @property
    def name(self) -> str:
        return "mol2vec"

    @property
    def description(self) -> str:
        return (
            "Mol2Vec: Word2Vecインスパイアの分子埋め込み (300次元)。"
            "pip install mol2vec"
        )

    def is_available(self) -> bool:
        try:
            import mol2vec  # noqa: F401
            from gensim.models import Word2Vec  # noqa: F401
            return True
        except ImportError:
            return False

    def _load_model(self):
        if self._model is not None:
            return
        from gensim.models import Word2Vec

        if self._model_path:
            self._model = Word2Vec.load(self._model_path)
        else:
            # デフォルトモデルを試す
            import importlib.resources
            try:
                model_path = str(importlib.resources.files("mol2vec") / "models" / "model_300dim.pkl")
                self._model = Word2Vec.load(model_path)
            except Exception:
                logger.warning("Mol2Vec: デフォルトモデルが見つかりません。ダミーベクトルを生成します。")
                self._model = None

    def compute(self, smiles_list: list[str], **kwargs) -> DescriptorResult:
        self._require_available()
        logger.info(f"📝 Mol2Vec 埋め込み計算を開始します（{len(smiles_list)}分子）...")
        from rdkit import Chem
        from mol2vec.features import mol2alt_sentence, MolSentence, sentences2vec

        self._load_model()

        n = len(smiles_list)
        dim = 300
        failed_indices = []

        if self._model is None:
            # モデルがない場合はダミー
            descriptors = pd.DataFrame(
                np.zeros((n, dim)),
                columns=[f"Mol2Vec_{i}" for i in range(dim)]
            )
            return DescriptorResult(
                descriptors=descriptors,
                smiles_list=smiles_list,
                failed_indices=list(range(n)),
                adapter_name=self.name,
                metadata={"warning": "事前学習モデルが見つかりません"},
            )

        sentences = []
        valid_mask = []
        for i, smi in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    raise ValueError("Invalid SMILES")
                sentence = mol2alt_sentence(mol, self._radius)
                sentences.append(MolSentence(sentence))
                valid_mask.append(True)
            except Exception:
                sentences.append(None)
                valid_mask.append(False)
                failed_indices.append(i)

        # 有効な文のみベクトル化
        valid_sentences = [s for s in sentences if s is not None]
        if valid_sentences:
            vectors = sentences2vec(valid_sentences, self._model, unseen="UNK")
        else:
            vectors = np.zeros((0, dim))

        # 全サンプルの結果を構築
        full_array = np.full((n, dim), np.nan)
        valid_idx = 0
        for i in range(n):
            if valid_mask[i]:
                full_array[i] = vectors[valid_idx]
                valid_idx += 1

        col_names = [f"Mol2Vec_{j}" for j in range(dim)]
        descriptors = pd.DataFrame(full_array, columns=col_names)

        return DescriptorResult(
            descriptors=descriptors,
            smiles_list=smiles_list,
            failed_indices=failed_indices,
            adapter_name=self.name,
        )

    def get_descriptors_metadata(self) -> list[DescriptorMetadata]:
        return [
            DescriptorMetadata(
                name=f"Mol2Vec_{i}",
                meaning=f"Mol2Vec 分子埋め込みベクトル 第{i}次元。分子部分構造のWord2Vecベース分散表現 (Jaeger et al., JCIM 2018)",
                is_count=False,
            )
            for i in range(300)  # Mol2Vecは300次元固定
        ]
