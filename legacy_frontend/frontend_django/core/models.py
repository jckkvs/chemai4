"""
core/models.py
ChemAI ML Studio - Django データモデル
"""
import uuid
from django.db import models


class AnalysisSession(models.Model):
    """解析セッション。CSVアップロード〜結果表示までを管理。"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200, default="新しい解析")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # データ
    uploaded_file = models.FileField(upload_to="uploads/", blank=True, null=True)
    original_filename = models.CharField(max_length=500, blank=True, default="")
    n_rows = models.IntegerField(default=0)
    n_cols = models.IntegerField(default=0)

    # 列設定
    target_col = models.CharField(max_length=200, blank=True, default="")
    smiles_col = models.CharField(max_length=200, blank=True, default="")
    task_type = models.CharField(max_length=20, default="regression",
                                 choices=[("regression", "回帰"), ("classification", "分類")])

    # 記述子
    selected_descriptors = models.JSONField(default=list, blank=True)
    precalc_data_path = models.CharField(max_length=500, blank=True, default="")

    # 解析設定
    analysis_config = models.JSONField(default=dict, blank=True)

    # 結果
    status = models.CharField(max_length=30, default="created",
                              choices=[
                                  ("created", "作成済"),
                                  ("data_loaded", "データ読込済"),
                                  ("eda_completed", "EDA完了"),
                                  ("descriptors_calculated", "記述子計算済"),
                                  ("running", "解析実行中"),
                                  ("completed", "完了"),
                                  ("error", "エラー"),
                              ])
    result_data = models.JSONField(default=dict, blank=True)
    eda_results = models.JSONField(null=True, blank=True, default=dict)
    error_message = models.TextField(blank=True, default="")

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self):
        return f"{self.name} ({self.status})"
