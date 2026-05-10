"""
backend/descriptors/base.py

記述子エンジンの関数を抽象化した基底クラス DescriptorFunction を提供する。
このクラスを継承して新しい記述子計算モジュールを作成することで、
統一されたシグネチャによる呼び出しと、UIやパイプラインからの自動認識が可能になる。
"""
from typing import List, Any
from abc import ABC, abstractmethod
import inspect

class DescriptorFunction(ABC):
    """
    記述子関数の抽象基底クラス
    
    必須シグネチャ:
        - 引数: smiles: List[str], **kwargs
        - 戻り値: List[Union[float, int, str]] （長さ=入力SMILES数）
    """
    
    @property
    @abstractmethod
    def metadata(self) -> dict:
        """
        docstringから自動抽出、または手動で定義されるメタデータ情報。
        UIで表示する際の「表示名」「カテゴリ」「説明」、また各パラメータの「型」「説明」を含む。
        """
        pass
    
    @abstractmethod
    def __call__(self, smiles: List[str], **kwargs) -> List[Any]:
        """
        記述量計算の実装本体。
        
        Parameters
        ----------
        smiles : List[str]
            SMILES文字列のリスト
        **kwargs : dict
            設定パラメータ（metadata['params'] に定義されたものに合致する）
        
        Returns
        -------
        List[Any]
            計算結果のリスト（入力の `smiles` と同じ長さであること）
        """
        pass
    
    def validate_signature(self) -> tuple[bool, str]:
        """
        関数シグネチャが実装要件を満たしているかを検証する。
        要件を満たさない場合は False と警告メッセージを返す（エラー扱いではなく、システムはロードをスキップするか警告を出す）。
        
        Returns:
            (is_valid: bool, message: str)
        """
        sig = inspect.signature(self.__call__)
        params = list(sig.parameters.keys())
        
        # 必須: 第1引数が 'smiles'
        if not params or params[0] != 'smiles':
            return False, "第1引数は 'smiles' である必要があります (smiles: List[str])"
        
        # 戻り値のアノテーションチェック（警告レベルの検証）
        if sig.return_annotation != inspect.Signature.empty:
            try:
                from typing import get_origin
                origin = get_origin(sig.return_annotation)
                if origin is not list and origin is not List:
                    return False, "戻り値のアノテーションは List[...] であることが推奨されます"
            except Exception:
                pass
                
        return True, "OK"
