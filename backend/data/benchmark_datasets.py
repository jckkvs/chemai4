"""
backend/data/benchmark_datasets.py

化学・材料系の基本的なオープンベンチマークデータセットを
動的にダウンロード・キャッシュしてDataFrameとして提供するモジュール。
"""
from __future__ import annotations

import io
import logging
import urllib.request
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

# MoleculeNet系の代表的なデータセット（CSVのRaw URL）
BENCHMARK_URLS = {
    "esol": "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv",
    "freesolv": "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/SAMPL.csv",
    "lipophilicity": "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/Lipophilicity.csv"
}

BenchmarkName = Literal["esol", "freesolv", "lipophilicity"]

# インメモリキャッシュ
_db_cache: dict[str, pd.DataFrame] = {}


def list_benchmark_datasets() -> list[dict[str, str]]:
    """利用可能なベンチマークデータセットの一覧を返す。"""
    return [
        {
            "id": "esol",
            "name": "ESOL (水溶解度)",
            "description": "Delaneyによる水分子への溶解度(logS)データセット (1,128化合物)",
            "target": "measured log solubility in mols per litre"
        },
        {
            "id": "freesolv",
            "name": "FreeSolv (水和自由エネルギー)",
            "description": "SAMPLブラインドチャレンジから提供される水和自由エネルギーデータ (642化合物)",
            "target": "expt"
        },
        {
            "id": "lipophilicity",
            "name": "Lipophilicity (脂溶性)",
            "description": "アストラゼネカ社提供のオクタノール/水分配係数(logD at pH 7.4)データ (4,200化合物)",
            "target": "exp"
        }
    ]


def load_benchmark(name: BenchmarkName) -> pd.DataFrame:
    """
    指定されたベンチマークデータセットをダウンロードしてDataFrameとして返す。
    一度ダウンロードしたものはインメモリ・キャッシュされる。

    Args:
        name: データセット名 (esol, freesolv, lipophilicity)
        
    Returns:
        pd.DataFrame
    """
    if name not in BENCHMARK_URLS:
        raise ValueError(f"未知のベンチマークデータセット: {name}")

    if name in _db_cache:
        return _db_cache[name].copy()

    url = BENCHMARK_URLS[name]
    logger.info(f"Downloading benchmark dataset '{name}' from {url}...")
    
    try:
        # User-Agentを指定しないと弾かれる場合があるためurllibでRequestを組む
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req) as response:
            csv_data = response.read()
            
        df = pd.read_csv(io.BytesIO(csv_data))
        _db_cache[name] = df
        return df.copy()
        
    except Exception as e:
        logger.error(f"データセット '{name}' のダウンロード中にエラー: {e}")
        raise RuntimeError(f"ベンチマークデータの取得に失敗しました: {e}")
