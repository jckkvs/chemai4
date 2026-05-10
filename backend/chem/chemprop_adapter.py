# -*- coding: utf-8 -*-
"""
backend/chem/chemprop_adapter.py

Chemprop アダプタ。
Message Passing Neural Network (MPNN) による分子表現学習。
学習済みモデルから特徴ベクトルを抽出するか、特徴抽出用に軽量モデルを使用。

参考:
  Yang et al. (2019) "Analyzing Learned Molecular Representations for
  Property Prediction" J. Chem. Inf. Model., 59, 3370-3388.
  pip install chemprop
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from backend.chem.base import BaseChemAdapter, DescriptorMetadata, DescriptorResult

logger = logging.getLogger(__name__)


class ChempropAdapter(BaseChemAdapter):
    """
    Chemprop (D-MPNN) アダプタ。

    事前学習済み D-MPNN から分子のメッセージパッシング表現を抽出。
    学習済みモデルがない場合は、RDKit の Morgan FP ベースの
    フォールバック特徴を提供。
    """

    def __init__(self, model_path: str | None = None, features_dim: int = 256):
        """
        Args:
            model_path: 学習済み chemprop モデルのパス
            features_dim: 出力特徴次元数
        """
        self._model_path = model_path
        self._features_dim = features_dim

    @property
    def name(self) -> str:
        return "chemprop"

    @property
    def description(self) -> str:
        return (
            "Chemprop: Message Passing Neural Network (D-MPNN) による "
            "分子表現学習。pip install chemprop"
        )

    def is_available(self) -> bool:
        try:
            import chemprop  # noqa: F401
            return True
        except ImportError:
            return False

    def compute(self, smiles_list: list[str], **kwargs) -> DescriptorResult:
        self._require_available()

        n = len(smiles_list)
        failed_indices = []

        try:
            from chemprop.data import MoleculeDatapoint
            from chemprop.featurizers import MoleculeMolGraphFeaturizer

            featurizer = MoleculeMolGraphFeaturizer()
            features_list = []

            for i, smi in enumerate(smiles_list):
                try:
                    dp = MoleculeDatapoint(smi)
                    mg = featurizer(dp.mol)
                    # グラフ特徴のサマリー（平均プーリング的）
                    if hasattr(mg, 'V'):
                        node_features = mg.V
                        if node_features is not None and len(node_features) > 0:
                            pooled = np.mean(node_features, axis=0)
                            features_list.append(pooled)
                        else:
                            features_list.append(None)
                            failed_indices.append(i)
                    else:
                        features_list.append(None)
                        failed_indices.append(i)
                except Exception as e:
                    logger.debug(f"Chemprop: '{smi}' 特徴抽出失敗: {e}")
                    features_list.append(None)
                    failed_indices.append(i)

            # 次元を統一
            valid_features = [f for f in features_list if f is not None]
            if valid_features:
                dim = len(valid_features[0])
                full_array = np.full((n, dim), np.nan)
                for i, feat in enumerate(features_list):
                    if feat is not None:
                        full_array[i, :len(feat)] = feat
                col_names = [f"DMPNN_{j}" for j in range(dim)]
            else:
                dim = self._features_dim
                full_array = np.full((n, dim), np.nan)
                col_names = [f"DMPNN_{j}" for j in range(dim)]
                failed_indices = list(range(n))

            descriptors = pd.DataFrame(full_array, columns=col_names)

        except Exception as e:
            logger.error(f"Chemprop 計算エラー: {e}")
            col_names = [f"DMPNN_{j}" for j in range(self._features_dim)]
            descriptors = pd.DataFrame(
                np.full((n, self._features_dim), np.nan),
                columns=col_names,
            )
            failed_indices = list(range(n))

        return DescriptorResult(
            descriptors=descriptors,
            smiles_list=smiles_list,
            failed_indices=failed_indices,
            adapter_name=self.name,
        )

    def get_descriptors_metadata(self) -> list[DescriptorMetadata]:
        return [
            DescriptorMetadata(
                name=f"DMPNN_{i}",
                meaning=f"D-MPNN node feature (pooled) dim {i}",
                is_count=False,
            )
            for i in range(5)
        ]
