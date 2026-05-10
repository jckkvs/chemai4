"""
backend/data/loader.py

各種ファイル形式のデータ読み込みモジュール。
CSV, Excel, Parquet, JSON, SQLite, SDF/MOL, SMILES列含むCSV に対応。
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# 対応拡張子の定義
_SUPPORTED_EXTENSIONS = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".txt": "tsv",
    ".xlsx": "excel",
    ".xls": "excel",
    ".parquet": "parquet",
    ".json": "json",
    ".db": "sqlite",
    ".sqlite": "sqlite",
    ".sqlite3": "sqlite",
    ".sdf": "sdf",
    ".mol": "mol",
}


def load_file(
    path: str | Path,
    smiles_col: str | None = None,
    target_col: str | None = None,
    sqlite_query: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    ファイルパスから DataFrame を読み込む。

    Args:
        path: 読み込むファイルのパス
        smiles_col: SMILES列名（SDF以外でSMILESが含まれる場合に指定）
        target_col: 目的変数の列名（読み込み後に確認用）
        sqlite_query: SQLiteの場合のクエリ（省略時は最初のテーブル）
        **kwargs: 各フォーマット固有の引数

    Returns:
        読み込んだ DataFrame

    Raises:
        ValueError: 未対応形式・読み込みエラー
        FileNotFoundError: ファイルが存在しない
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {path}")

    ext = path.suffix.lower()
    fmt = _SUPPORTED_EXTENSIONS.get(ext)

    if fmt is None:
        raise ValueError(
            f"未対応の拡張子 '{ext}'。対応: {list(_SUPPORTED_EXTENSIONS.keys())}"
        )

    loader_map = {
        "csv": _load_csv,
        "tsv": _load_tsv,
        "excel": _load_excel,
        "parquet": _load_parquet,
        "json": _load_json,
        "sqlite": _load_sqlite,
        "sdf": _load_sdf,
        "mol": _load_mol,
    }

    loader = loader_map[fmt]
    if fmt == "sqlite":
        df = loader(path, query=sqlite_query, **kwargs)  # type: ignore
    else:
        df = loader(path, **kwargs)  # type: ignore

    logger.info(f"読み込み完了: {path.name} -> shape={df.shape}")
    return df


def load_from_bytes(
    content: bytes,
    filename: str,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    アップロードされたファイルのバイト列からDataFrameを読み込む（Streamlit用）。

    Args:
        content: ファイルのバイト列
        filename: 元のファイル名（拡張子判定に使用）
        **kwargs: load_file に渡される追加引数

    Returns:
        読み込んだ DataFrame
    """
    import io

    ext = Path(filename).suffix.lower()
    fmt = _SUPPORTED_EXTENSIONS.get(ext)

    if fmt is None:
        raise ValueError(f"未対応の拡張子: {ext}")

    buf = io.BytesIO(content)

    if fmt == "csv":
        return pd.read_csv(buf, **kwargs)
    elif fmt == "tsv":
        return pd.read_csv(buf, sep="\t", **kwargs)
    elif fmt == "excel":
        return pd.read_excel(buf, **kwargs)
    elif fmt == "parquet":
        return pd.read_parquet(buf, **kwargs)
    elif fmt == "json":
        return pd.read_json(buf, **kwargs)
    elif fmt == "sdf":
        return _load_sdf_from_buf(buf, filename)
    else:
        raise ValueError(f"バイトストリームからの読み込みは '{fmt}' 未対応です")


# ---- プライベートローダー ----

def _load_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    encoding = kwargs.pop("encoding", None)
    if encoding is None:
        # UTF-8 を試みて失敗したら Shift-JIS（Windows日本語環境対応）
        try:
            return pd.read_csv(path, encoding="utf-8", **kwargs)
        except UnicodeDecodeError:
            logger.warning("UTF-8 失敗 → Shift-JIS で再試行")
            return pd.read_csv(path, encoding="shift-jis", **kwargs)
    return pd.read_csv(path, encoding=encoding, **kwargs)


def _load_tsv(path: Path, **kwargs: Any) -> pd.DataFrame:
    kwargs.setdefault("sep", "\t")
    return _load_csv(path, **kwargs)


def _load_excel(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_excel(path, **kwargs)


def _load_parquet(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_parquet(path, **kwargs)


def _load_json(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_json(path, **kwargs)


def _load_sqlite(path: Path, query: str | None = None, **kwargs: Any) -> pd.DataFrame:
    con = sqlite3.connect(path)
    try:
        if query:
            return pd.read_sql_query(query, con, **kwargs)
        # クエリ未指定時は最初のテーブルを読み込む
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'", con
        )
        if tables.empty:
            raise ValueError(f"SQLiteにテーブルが存在しません: {path}")
        table_name = tables["name"].iloc[0]
        logger.info(f"SQLite: テーブル '{table_name}' を読み込み")
        return pd.read_sql_query(f"SELECT * FROM {table_name}", con, **kwargs)
    finally:
        con.close()


def _load_sdf(path: Path, **kwargs: Any) -> pd.DataFrame:
    """RDKit を使って SDF ファイルを DataFrame に変換する。"""
    from backend.utils.optional_import import require
    require("rdkit.Chem", feature="SDF読み込み")

    from rdkit.Chem import PandasTools  # type: ignore

    df = PandasTools.LoadSDF(str(path), molColName="Molecule", includeFingerprints=False)
    logger.info(f"SDF読み込み: {len(df)} 件")
    return df


def _load_mol(path: Path, **kwargs: Any) -> pd.DataFrame:
    """単一 MOL ファイルを1行のDataFrameで返す。"""
    from backend.utils.optional_import import require
    require("rdkit.Chem", feature="MOL読み込み")

    from rdkit import Chem  # type: ignore

    mol = Chem.MolFromMolFile(str(path))
    if mol is None:
        raise ValueError(f"MOLファイルの解析に失敗: {path}")
    smiles = Chem.MolToSmiles(mol)
    return pd.DataFrame([{"SMILES": smiles, "Molecule": mol}])


def _load_sdf_from_buf(buf: Any, filename: str) -> pd.DataFrame:
    """バイトバッファからSDFを読み込む（Streamlit/stlite用）。"""
    import tempfile

    from backend.utils.optional_import import require
    require("rdkit.Chem", feature="SDF読み込み（バッファ）")

    from rdkit.Chem import PandasTools  # type: ignore

    with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as tmp:
        tmp.write(buf.read())
        tmp_path = tmp.name

    try:
        return PandasTools.LoadSDF(tmp_path, molColName="Molecule", includeFingerprints=False)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---- 書き出し ----

def save_dataframe(
    df: pd.DataFrame,
    path: str | Path,
    fmt: str | None = None,
    **kwargs: Any,
) -> Path:
    """
    DataFrame をファイルに保存する。

    Args:
        df: 保存するDataFrame
        path: 保存先パス
        fmt: 形式（省略時は拡張子から判定）。 "csv" | "excel" | "parquet" | "json"
        **kwargs: 各フォーマット固有の引数

    Returns:
        保存されたファイルパス

    Raises:
        ValueError: 未対応形式
    """
    path = Path(path)
    ext = path.suffix.lower()
    fmt = fmt or _SUPPORTED_EXTENSIONS.get(ext, "csv")

    if fmt in ("csv", "tsv"):
        sep = "\t" if fmt == "tsv" else ","
        df.to_csv(path, index=False, sep=sep, **kwargs)
    elif fmt == "excel":
        df.to_excel(path, index=False, **kwargs)
    elif fmt == "parquet":
        df.to_parquet(path, index=False, **kwargs)
    elif fmt == "json":
        df.to_json(path, orient="records", force_ascii=False, **kwargs)
    else:
        raise ValueError(f"未対応の保存フォーマット: {fmt}")

    logger.info(f"保存完了: {path} ({df.shape})")
    return path


def get_supported_extensions() -> list[str]:
    """対応している拡張子の一覧を返す（GUI表示用）。"""
    return list(_SUPPORTED_EXTENSIONS.keys())
