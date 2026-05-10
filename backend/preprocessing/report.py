from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class PreprocessingReport:
    """
    データの前処理中に何が起きたかを記録する透明性レポート。
    - imputations: どのように欠損値が補完されたか
    - dropped_cols: 分散ゼロ等で削除された列
    - transformed_dims: 変換前後の次元数
    """
    original_features_count: int = 0
    final_features_count: int = 0
    missing_value_handled: bool = False
    imputations: Dict[str, str] = field(default_factory=dict)
    dropped_cols: List[str] = field(default_factory=list)
    scaler_used: str = "None"
    
    def generate_summary(self) -> str:
        summary = f"🔄 特徴量数: {self.original_features_count} ➡️ {self.final_features_count}次元\n"
        summary += f"⚖️ スケーリング: {self.scaler_used}\n"
        if self.missing_value_handled:
            summary += f"🩹 欠損値補完: {len(self.imputations)}列に適用\n"
            for col, method in list(self.imputations.items())[:5]:
                summary += f"  - {col}: {method}\n"
            if len(self.imputations) > 5:
                summary += f"  ...他 {len(self.imputations) - 5}列\n"
        if self.dropped_cols:
            summary += f"🗑️ 削除された列 (分散ゼロ等): {len(self.dropped_cols)}列\n"
            summary += f"  - {', '.join(self.dropped_cols[:5])}" + ("..." if len(self.dropped_cols) > 5 else "") + "\n"
        return summary
