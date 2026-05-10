"""
core/views.py
ChemAI ML Studio - Django ビュー
"""
from __future__ import annotations

import io
import json
import traceback
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

# backendへのパスを追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from .models import AnalysisSession


# ─────────────────────────────────────────────
# ページビュー
# ─────────────────────────────────────────────
def index(request):
    """ダッシュボード：セッション一覧 + 新規作成"""
    sessions = AnalysisSession.objects.all()[:20]
    return render(request, "core/index.html", {"sessions": sessions})


def session_detail(request, session_id):
    """セッション詳細（ステップ別タブ構成）"""
    session = get_object_or_404(AnalysisSession, id=session_id)
    step = request.GET.get("step", "data")
    return render(request, "core/session.html", {
        "session": session,
        "step": step,
    })


def new_session(request):
    """新規セッション作成"""
    session = AnalysisSession.objects.create(name="新しい解析")
    return redirect("session_detail", session_id=session.id)


# ─────────────────────────────────────────────
# API: データアップロード
# ─────────────────────────────────────────────
@csrf_exempt
@require_POST
def upload_data(request, session_id):
    """CSVファイルをアップロードしてセッションに紐付け"""
    session = get_object_or_404(AnalysisSession, id=session_id)

    uploaded_file = request.FILES.get("file")
    if not uploaded_file:
        return JsonResponse({"error": "ファイルが選択されていません"}, status=400)

    try:
        # ファイルを保存
        session.uploaded_file = uploaded_file
        session.original_filename = uploaded_file.name

        # DataFrameとして読み込み
        content = uploaded_file.read()
        uploaded_file.seek(0)

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif uploaded_file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            return JsonResponse({"error": "CSV/Excelファイルのみ対応"}, status=400)

        session.n_rows = len(df)
        session.n_cols = len(df.columns)
        session.status = "data_loaded"
        session.save()

        # プレビューデータ
        preview = df.head(10).to_dict(orient="records")
        columns = list(df.columns)
        dtypes = {col: str(df[col].dtype) for col in columns}

        return JsonResponse({
            "success": True,
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "columns": columns,
            "dtypes": dtypes,
            "preview": preview,
        })

    except Exception as e:
        return JsonResponse({"error": str(e), "trace": traceback.format_exc()}, status=500)


@csrf_exempt
@require_POST
def set_columns(request, session_id):
    """目的変数・SMILES列を設定"""
    session = get_object_or_404(AnalysisSession, id=session_id)
    data = json.loads(request.body)
    session.target_col = data.get("target_col", "")
    session.smiles_col = data.get("smiles_col", "")
    session.task_type = data.get("task_type", "regression")
    session.save()
    return JsonResponse({"success": True})


@csrf_exempt
@require_POST
def load_sample(request, session_id):
    """デバッグ用サンプルデータを読み込み"""
    session = get_object_or_404(AnalysisSession, id=session_id)
    data = json.loads(request.body)
    sample_type = data.get("type", "regression")
    include_smiles = data.get("include_smiles", True)

    np.random.seed(42)
    n = 25

    if sample_type == "regression":
        if include_smiles:
            smiles_list = [
                "CCO", "CC(=O)O", "c1ccccc1", "CC(C)O", "CCCO", "CC=O",
                "c1ccc(O)cc1", "CC(=O)OC", "CCOC", "CCN",
                "CC(C)(C)O", "c1ccc(N)cc1", "OC(=O)c1ccccc1", "CCOCC",
                "CC(O)CC", "c1ccc(Cl)cc1", "CC(=O)N", "CCCCCO",
                "c1ccc(F)cc1", "CC(C)=O", "OCCO", "c1ccncc1",
                "CC(=O)CC", "CCCCO", "c1ccc(C)cc1"
            ]
            df = pd.DataFrame({
                "SMILES": smiles_list[:n],
                "target_value": np.random.randn(n) * 2 + 5,
            })
            session.smiles_col = "SMILES"
        else:
            df = pd.DataFrame({
                "feature1": np.random.randn(n),
                "feature2": np.random.randn(n) * 2,
                "target_value": np.random.randn(n) * 2 + 5,
            })
        session.target_col = "target_value"
        session.task_type = "regression"
    else:
        df = pd.DataFrame({
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n) * 2,
            "target_class": np.random.choice(["A", "B"], n),
        })
        session.target_col = "target_class"
        session.task_type = "classification"

    # CSVとして保存（MEDIA_ROOTベースの絶対パス）
    from django.conf import settings as django_settings
    media_root = Path(django_settings.MEDIA_ROOT)
    csv_path = media_root / "uploads" / f"sample_{session.id}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    session.uploaded_file = f"uploads/sample_{session.id}.csv"
    session.original_filename = f"sample_{sample_type}.csv"
    session.n_rows = len(df)
    session.n_cols = len(df.columns)
    session.status = "data_loaded"
    session.save()

    return JsonResponse({
        "success": True,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": list(df.columns),
        "preview": df.head(10).to_dict(orient="records"),
    })


# ─────────────────────────────────────────────
# API: 記述子計算
# ─────────────────────────────────────────────
@csrf_exempt
@require_POST
def calculate_descriptors(request, session_id):
    """全エンジンで記述子を自動計算"""
    session = get_object_or_404(AnalysisSession, id=session_id)

    if not session.smiles_col:
        return JsonResponse({"error": "SMILES列が設定されていません"}, status=400)

    try:
        # データ読み込み
        df = pd.read_csv(session.uploaded_file.path)
        smiles_list = df[session.smiles_col].tolist()

        from backend.chem import ADAPTER_REGISTRY, get_available_adapters
        available = get_available_adapters()

        results = {}
        all_dfs = []

        for name, adapter_cls in available.items():
            try:
                adapter = adapter_cls()
                desc_df = adapter.compute(smiles_list)
                if desc_df is not None and len(desc_df.columns) > 0:
                    all_dfs.append(desc_df)
                    results[name] = {
                        "n_descriptors": len(desc_df.columns),
                        "columns": list(desc_df.columns)[:10],
                    }
            except Exception as e:
                results[name] = {"error": str(e)}

        if all_dfs:
            combined = pd.concat(all_dfs, axis=1)
            # NaN列を除去
            combined = combined.dropna(axis=1, how="all")
            # 保存
            save_path = Path(f"media/descriptors/{session.id}.parquet")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            combined.to_parquet(save_path)
            session.precalc_data_path = str(save_path)
            session.status = "descriptors_calculated"
            session.save()

            return JsonResponse({
                "success": True,
                "total_descriptors": len(combined.columns),
                "engines": results,
                "columns": list(combined.columns),
            })
        else:
            return JsonResponse({"error": "計算可能な記述子がありませんでした"}, status=500)

    except Exception as e:
        return JsonResponse({"error": str(e), "trace": traceback.format_exc()}, status=500)


@csrf_exempt
@require_POST
def run_analysis(request, session_id):
    """AutoMLで解析を実行"""
    session = get_object_or_404(AnalysisSession, id=session_id)

    if not session.target_col:
        return JsonResponse({"error": "目的変数が設定されていません"}, status=400)
    if session.status == "running":
        return JsonResponse({"error": "既に解析が実行中です"}, status=400)

    try:
        # データ読み込み
        df = pd.read_csv(session.uploaded_file.path)

        # 記述子データがあれば結合
        if session.precalc_data_path:
            try:
                desc_df = pd.read_parquet(session.precalc_data_path)
                df = pd.concat([df, desc_df], axis=1)
            except Exception:
                pass

        session.status = "running"
        session.save()

        # AutoML実行
        from backend.models.automl import AutoMLEngine
        data = json.loads(request.body) if request.body else {}
        model_keys = data.get("model_keys", None)
        cv_folds = data.get("cv_folds", 5)

        engine = AutoMLEngine(
            task=session.task_type,
            cv_folds=cv_folds,
            model_keys=model_keys,
            selected_descriptors=session.selected_descriptors or None,
        )

        result = engine.run(
            df,
            target_col=session.target_col,
            smiles_col=session.smiles_col or None,
        )

        # 結果をJSONシリアライズ可能な形式に変換
        result_data = {
            "task": result.task,
            "best_model_key": result.best_model_key,
            "best_score": float(result.best_score),
            "scoring": result.scoring,
            "model_scores": {k: float(v) for k, v in result.model_scores.items()},
            "model_details": {
                k: {
                    "mean": float(v.get("mean", 0)),
                    "std": float(v.get("std", 0)),
                    "fit_time": float(v.get("fit_time", 0)),
                    "fold_scores": [float(s) for s in v.get("fold_scores", [])],
                }
                for k, v in result.model_details.items()
            },
            "elapsed_seconds": float(result.elapsed_seconds),
            "warnings": result.warnings,
            "n_features": (
                len(result.processed_X.columns)
                if result.processed_X is not None
                else 0
            ),
        }

        session.status = "completed"
        session.result_data = result_data
        session.save()

        return JsonResponse({
            "success": True,
            "result": result_data,
        })

    except Exception as e:
        session.status = "error"
        session.error_message = str(e)
        session.save()
        return JsonResponse({
            "error": str(e),
            "trace": traceback.format_exc(),
        }, status=500)


@csrf_exempt
@require_POST
def run_multi_analysis(request, session_id):
    """
    複数の特徴量セット × パイプライン（通常 / 高次元JL-RP）でAutoMLを実行。

    POST body (JSON):
    {
        "cv_folds": 5,
        "feature_sets": [
            {
                "id": "set1",
                "name": "RDKit基本記述子",
                "descriptors": ["MW", "LogP", ...],  // 空=全列
                "pipeline": "normal",                  // "normal" | "highdim"
                "rp_eps": 0.1                          // highdimのみ
            },
            ...
        ]
    }
    Returns:
    {
        "success": True,
        "results": [
            {
                "set_id": "set1",
                "set_name": "RDKit基本記述子",
                "pipeline_type": "normal",
                "rp_applied": false,
                "n_features_in": 200,
                "n_features_out": 200,
                ...AutoMLResult fields...
            }
        ]
    }
    """
    session = get_object_or_404(AnalysisSession, id=session_id)

    if not session.target_col:
        return JsonResponse({"error": "目的変数が設定されていません"}, status=400)

    try:
        data = json.loads(request.body) if request.body else {}
        cv_folds = int(data.get("cv_folds", 5))
        feature_sets = data.get("feature_sets", [])

        if not feature_sets:
            return JsonResponse({"error": "feature_sets が空です"}, status=400)

        # データ読み込み
        df = pd.read_csv(session.uploaded_file.path)

        # 記述子データがあれば結合
        if session.precalc_data_path:
            try:
                desc_df = pd.read_parquet(session.precalc_data_path)
                df = pd.concat([df, desc_df], axis=1)
            except Exception:
                pass

        session.status = "running"
        session.save()

        from backend.models.automl import AutoMLEngine

        engine = AutoMLEngine(
            task=session.task_type,
            cv_folds=cv_folds,
        )

        multi_results = engine.run_multi_feature_sets(
            df=df,
            target_col=session.target_col,
            smiles_col=session.smiles_col or None,
            feature_sets=feature_sets,
        )

        # シリアライズ
        serialized = []
        for i, (fs_def, result) in enumerate(zip(feature_sets, multi_results)):
            # warningsから __feature_set_pipeline__ を抽出
            pipeline_type = fs_def.get("pipeline", "normal")
            for w in result.warnings:
                if w.startswith("__feature_set_pipeline__:"):
                    pipeline_type = w.split(":", 1)[1]
                    break

            # JL-RP が実際に適用されたか確認
            rp_applied = False
            n_features_in = 0
            n_features_out = 0
            if result.processed_X is not None:
                n_features_out = len(result.processed_X.columns)
            if result.X_train is not None:
                n_features_in = result.X_train.shape[1]
            # best_pipelineにjl_rpステップがあり、projection_active_=Trueなら適用済み
            try:
                pipe = result.best_pipeline
                # smiles_varsラッパー考慮
                inner = pipe
                if hasattr(pipe, "named_steps") and "main_pipe" in pipe.named_steps:
                    inner = pipe.named_steps["main_pipe"]
                if hasattr(inner, "named_steps") and "jl_rp" in inner.named_steps:
                    jl_rp_step = inner.named_steps["jl_rp"]
                    if hasattr(jl_rp_step, "projection_active_"):
                        rp_applied = jl_rp_step.projection_active_
                        n_features_in = getattr(jl_rp_step, "n_features_in_", n_features_in)
                        n_features_out = getattr(jl_rp_step, "n_components_", n_features_out)
            except Exception:
                pass

            user_warnings = [
                w for w in result.warnings
                if not w.startswith("__feature_set_")
            ]

            serialized.append({
                "set_id": fs_def.get("id", f"set{i+1}"),
                "set_name": fs_def.get("name", f"セット{i+1}"),
                "pipeline_type": pipeline_type,
                "rp_applied": rp_applied,
                "n_features_in": n_features_in,
                "n_features_out": n_features_out,
                "task": result.task,
                "best_model_key": result.best_model_key,
                "best_score": float(result.best_score),
                "scoring": result.scoring,
                "model_scores": {k: float(v) for k, v in result.model_scores.items()},
                "model_details": {
                    k: {
                        "mean": float(v.get("mean", 0)),
                        "std": float(v.get("std", 0)),
                        "fit_time": float(v.get("fit_time", 0)),
                        "fold_scores": [float(s) for s in v.get("fold_scores", [])],
                    }
                    for k, v in result.model_details.items()
                },
                "elapsed_seconds": float(result.elapsed_seconds),
                "warnings": user_warnings,
            })

        # 最良セットを識別（スコアが最高）
        if serialized:
            best_set_idx = max(range(len(serialized)), key=lambda i: serialized[i]["best_score"])
            for i, s in enumerate(serialized):
                s["is_best_set"] = (i == best_set_idx)

        session.status = "completed"
        session.result_data = {"multi_results": serialized}
        session.save()

        return JsonResponse({"success": True, "results": serialized})

    except Exception as e:
        session.status = "error"
        session.error_message = str(e)
        session.save()
        return JsonResponse({"error": str(e), "trace": traceback.format_exc()}, status=500)


def get_results(request, session_id):
    """セッションの解析結果をJSON返却"""
    session = get_object_or_404(AnalysisSession, id=session_id)
    if session.status != "completed":
        return JsonResponse({
            "status": session.status,
            "error": session.error_message,
        })
    return JsonResponse({
        "status": "completed",
        "result": session.result_data,
    })


def check_status(request, session_id):
    """セッションの現在ステータスをJSON返却"""
    session = get_object_or_404(AnalysisSession, id=session_id)
    return JsonResponse({
        "status": session.status,
        "error": session.error_message if session.status == "error" else "",
    })


def help_page(request):
    """ヘルプページ"""
    return render(request, "core/help.html")


# ─────────────────────────────────────────────
# API: パラメータ自動UI用エンドポイント
# ─────────────────────────────────────────────

def get_model_params_schema(request, model_key: str):
    """
    モデルのパラメータスキーマをJSONで返す。

    フロントエンドJSが動的にUIフォームを構築するために使用。
    """
    try:
        from backend.models.factory import list_models
        from backend.ui.param_schema import introspect_params

        # レジストリからモデルクラスを取得
        for task in ("regression", "classification"):
            for m in list_models(task=task, available_only=False):
                if m["key"] == model_key:
                    model_cls = m.get("class")
                    if model_cls is None:
                        return JsonResponse({"error": f"モデル '{model_key}' のクラスが見つかりません"}, status=404)
                    specs = introspect_params(model_cls)
                    return JsonResponse({
                        "model_key": model_key,
                        "model_name": m["name"],
                        "class_name": model_cls.__name__,
                        "params": [s.to_dict() for s in specs],
                    })

        return JsonResponse({"error": f"モデル '{model_key}' が見つかりません"}, status=404)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def get_adapter_params_schema(request, adapter_name: str):
    """
    アダプタのパラメータスキーマをJSONで返す。

    フロントエンドJSが動的にUIフォームを構築するために使用。
    """
    try:
        import importlib
        from backend.ui.param_schema import introspect_params

        # アダプタ名→モジュール/クラスのマッピング
        ADAPTERS = {
            "rdkit":          ("backend.chem.rdkit_adapter",           "RDKitAdapter"),
            "mordred":        ("backend.chem.mordred_adapter",         "MordredAdapter"),
            "group_contrib":  ("backend.chem.group_contrib_adapter",   "GroupContribAdapter"),
            "descriptastorus": ("backend.chem.descriptastorus_adapter", "DescriptaStorusAdapter"),
            "molai":          ("backend.chem.molai_adapter",           "MolAIAdapter"),
            "skfp":           ("backend.chem.skfp_adapter",            "SkfpAdapter"),
            "xtb":            ("backend.chem.xtb_adapter",             "XTBAdapter"),
            "unipka":         ("backend.chem.unipka_adapter",          "UniPkaAdapter"),
        }

        adapter_key = adapter_name.lower()
        if adapter_key not in ADAPTERS:
            return JsonResponse({"error": f"アダプタ '{adapter_name}' が見つかりません"}, status=404)

        mod_path, cls_name = ADAPTERS[adapter_key]
        mod = importlib.import_module(mod_path)
        adapter_cls = getattr(mod, cls_name)
        specs = introspect_params(adapter_cls)

        return JsonResponse({
            "adapter_name": adapter_name,
            "class_name": cls_name,
            "params": [s.to_dict() for s in specs],
        })

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


# ═══════════════════════════════════════════════════════════
# 実験計画法（DoE）
# ═══════════════════════════════════════════════════════════

def doe_page(request):
    """実験計画法ページ"""
    return render(request, "core/doe.html", {"active_page": "doe"})


@csrf_exempt
@require_POST
def doe_upload_existing(request):
    """既存実験データCSVアップロード"""
    try:
        f = request.FILES.get("file")
        if not f:
            return JsonResponse({"error": "ファイルが選択されていません"}, status=400)
        import io as _io
        if f.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(_io.BytesIO(f.read()))
        else:
            content = f.read().decode("utf-8-sig", errors="replace")
            df = pd.read_csv(_io.StringIO(content))
        import uuid
        session_key = str(uuid.uuid4())
        request.session[f"doe_existing_{session_key}"] = df.to_json(orient="split")
        column_info = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                column_info.append({"name": col, "dtype": "numeric",
                                    "min": round(float(df[col].min()), 6),
                                    "max": round(float(df[col].max()), 6)})
            else:
                cats = df[col].dropna().unique().tolist()
                column_info.append({"name": col, "dtype": "category",
                                    "categories": [str(c) for c in cats[:50]]})
        return JsonResponse({"session_key": session_key, "n_rows": len(df), "n_cols": len(df.columns),
                             "columns": list(df.columns), "column_info": column_info,
                             "preview": df.head(5).to_dict(orient="records")})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_POST
def doe_run(request):
    """DoE最適化実行API"""
    try:
        body = json.loads(request.body)
        from backend.doe.factor import Factor
        from backend.doe.design import DoEOptimizer
        factors = []
        for f in body.get("factors", []):
            if f["type"] == "continuous":
                factors.append(Factor.continuous(f["name"], float(f.get("low", 0)),
                                                 float(f.get("high", 1)), int(f.get("n_levels", 5))))
            else:
                factors.append(Factor.categorical(f["name"], f.get("categories", [])))
        if not factors:
            return JsonResponse({"error": "因子が設定されていません"}, status=400)
        criterion = body.get("criterion", "D")
        if criterion == "orthogonal":
            from backend.doe.orthogonal import apply_orthogonal_array
            oa_type = body.get("oa_type", "L9(3⁴)")
            design_df, warning = apply_orthogonal_array(oa_type, factors)
            if design_df.empty:
                return JsonResponse({"error": warning or "直交表生成に失敗"}, status=400)
            return JsonResponse({"criterion_name": f"直交表 ({oa_type})", "criterion_value": 0,
                                 "d_efficiency": 0, "n_existing": 0, "n_new": len(design_df),
                                 "columns": list(design_df.columns),
                                 "rows": design_df.to_dict(orient="records"),
                                 "is_new": [True] * len(design_df), "warning": warning})
        existing_df = None
        if body.get("mode") == "augment":
            sk = body.get("existing_session_key")
            if sk:
                raw = request.session.get(f"doe_existing_{sk}")
                if raw:
                    existing_df = pd.read_json(raw, orient="split")
        optimizer = DoEOptimizer(factors=factors, n_new=body.get("n_new", 10), criterion=criterion,
                                 max_candidates=body.get("max_candidates", 5000),
                                 n_starts=body.get("n_starts", 5), existing_df=existing_df)
        result = optimizer.optimize()
        return JsonResponse({"criterion_name": result.criterion_name, "criterion_value": result.criterion_value,
                             "d_efficiency": result.d_efficiency,
                             "n_existing": result.info.get("n_existing", 0),
                             "n_new": result.info.get("n_new", body.get("n_new", 10)),
                             "columns": list(result.design_df.columns),
                             "rows": result.design_df.to_dict(orient="records"), "is_new": result.is_new})
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)


# ═══════════════════════════════════════════════════════════
# 逆解析
# ═══════════════════════════════════════════════════════════

@csrf_exempt
@require_POST
def session_doe_factors(request, session_id):
    """セッションデータから因子を自動検出"""
    try:
        session = get_object_or_404(AnalysisSession, id=session_id)
        if not session.data_json:
            return JsonResponse({"error": "データが読み込まれていません"}, status=400)
        df = pd.read_json(session.data_json, orient="split")
        target_col = session.target_col
        factors = []
        for col in df.columns:
            if col == target_col:
                continue
            if col == getattr(session, 'smiles_col', None):
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                factors.append({
                    "name": col, "type": "continuous",
                    "low": round(float(df[col].min()), 6),
                    "high": round(float(df[col].max()), 6),
                    "n_levels": 5,
                })
            else:
                cats = df[col].dropna().unique().tolist()
                factors.append({
                    "name": col, "type": "categorical",
                    "categories": [str(c) for c in cats[:50]],
                })
        return JsonResponse({"factors": factors})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_POST
def session_doe_run(request, session_id):
    """セッション内DoE実行（既存データ=セッションのデータ）"""
    try:
        session = get_object_or_404(AnalysisSession, id=session_id)
        body = json.loads(request.body)
        from backend.doe.factor import Factor
        from backend.doe.design import DoEOptimizer

        factors = []
        for f in body.get("factors", []):
            if f["type"] == "continuous":
                factors.append(Factor.continuous(f["name"], float(f.get("low", 0)),
                                                 float(f.get("high", 1)), int(f.get("n_levels", 5))))
            else:
                factors.append(Factor.categorical(f["name"], f.get("categories", [])))
        if not factors:
            return JsonResponse({"error": "因子が設定されていません"}, status=400)

        criterion = body.get("criterion", "D")

        # 直交表
        if criterion == "orthogonal":
            from backend.doe.orthogonal import apply_orthogonal_array
            oa_type = body.get("oa_type", "L9(3⁴)")
            design_df, warning = apply_orthogonal_array(oa_type, factors)
            if design_df.empty:
                return JsonResponse({"error": warning or "直交表生成に失敗"}, status=400)
            return JsonResponse({
                "criterion_name": f"直交表 ({oa_type})", "criterion_value": 0,
                "d_efficiency": 0, "n_existing": 0, "n_new": len(design_df),
                "columns": list(design_df.columns),
                "rows": design_df.to_dict(orient="records"),
                "is_new": [True] * len(design_df),
                "warning": warning,
            })

        # 既存データ（augmentモード時はセッションのデータを使用）
        existing_df = None
        if body.get("mode") == "augment" and session.data_json:
            df = pd.read_json(session.data_json, orient="split")
            target_col = session.target_col
            factor_names = [f.name for f in factors]
            available_cols = [c for c in factor_names if c in df.columns]
            if available_cols:
                existing_df = df[available_cols]

        optimizer = DoEOptimizer(
            factors=factors, n_new=body.get("n_new", 10), criterion=criterion,
            max_candidates=body.get("max_candidates", 5000),
            n_starts=body.get("n_starts", 5), existing_df=existing_df)
        result = optimizer.optimize()
        return JsonResponse({
            "criterion_name": result.criterion_name,
            "criterion_value": result.criterion_value,
            "d_efficiency": result.d_efficiency,
            "n_existing": result.info.get("n_existing", 0),
            "n_new": result.info.get("n_new", body.get("n_new", 10)),
            "columns": list(result.design_df.columns),
            "rows": result.design_df.to_dict(orient="records"),
            "is_new": result.is_new,
        })
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)


# ═══════════════════════════════════════════════════════════
# 逆解析
# ═══════════════════════════════════════════════════════════

@csrf_exempt
@require_POST
def run_inverse(request, session_id):
    """逆解析API"""
    try:
        session = get_object_or_404(AnalysisSession, id=session_id)
        body = json.loads(request.body)
        if not session.result_data:
            return JsonResponse({"error": "順解析の結果がありません"}, status=400)
        rd = session.result_data
        if "multi_results" in rd:
            rd = max(rd["multi_results"], key=lambda r: r.get("best_score", 0))
        import pickle, base64
        ppkl = rd.get("pipeline_pickle")
        if not ppkl:
            return JsonResponse({"error": "学習済みパイプラインが保存されていません"}, status=400)
        pipeline = pickle.loads(base64.b64decode(ppkl))
        feature_names = rd.get("feature_names", [])
        from backend.optim.inverse_optimizer import InverseConfig, run_inverse_optimization
        config = InverseConfig(method=body.get("method", "random"), target_mode=body.get("target_mode", "maximize"),
                               target_min=body.get("target_min"), target_max=body.get("target_max"),
                               constraints=body.get("constraints", {}), method_params=body.get("method_params", {}))
        result = run_inverse_optimization(predict_fn=lambda X_df: pipeline.predict(X_df),
                                          feature_names=feature_names, config=config)
        return JsonResponse({"candidates": result.candidates.to_dict(orient="records"),
                             "n_evaluated": result.n_evaluated, "best_predicted": result.best_predicted,
                             "method": result.method, "elapsed_seconds": round(result.elapsed_seconds, 2)})
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)


# ═══════════════════════════════════════════════════════════
# ベイズ最適化候補提案
# ═══════════════════════════════════════════════════════════

@csrf_exempt
@require_POST
def run_bayesian_suggest(request, session_id):
    """ベイズ最適化 候補提案API"""
    try:
        session = get_object_or_404(AnalysisSession, id=session_id)
        body = json.loads(request.body)
        if not session.data_json:
            return JsonResponse({"error": "データが読み込まれていません"}, status=400)
        df = pd.read_json(session.data_json, orient="split")
        target_col = session.target_col
        feature_cols = body.get("feature_cols", [])
        if not feature_cols:
            feature_cols = [c for c in df.columns if c != target_col and pd.api.types.is_numeric_dtype(df[c])]
        X = df[feature_cols].values
        y = df[target_col].values
        from backend.optim.bayesian_optimizer import BayesianOptimizer, BOConfig
        from backend.optim.search_space import SearchSpace
        bo_config = BOConfig(objective=body.get("objective", "minimize"), acquisition=body.get("acquisition", "ei"),
                             xi=body.get("xi", 0.01), kappa=body.get("kappa", 2.0),
                             target_lo=body.get("target_lo"), target_hi=body.get("target_hi"),
                             kernel_type=body.get("kernel_type", "default"),
                             batch_strategy=body.get("batch_strategy", "kriging_believer"),
                             n_candidates=body.get("n_candidates", 5))
        # 探索空間を既存データから自動推定し候補を生成
        space = SearchSpace.from_dataframe(df[feature_cols], margin=0.1)
        search_df = space.generate_candidates(method="auto", n_max=body.get("n_random", 5000))
        bo = BayesianOptimizer(config=bo_config)
        bo.fit(X, y)
        suggestions = bo.suggest(search_df, n=bo_config.n_candidates)
        pca_data = _compute_pca_viz(X, suggestions[feature_cols].values if hasattr(suggestions, 'columns') else suggestions, feature_cols)
        return JsonResponse({"suggestions": suggestions.to_dict(orient="records") if hasattr(suggestions, 'to_dict') else [],
                             "gp_info": bo.get_gp_info(), "pca": pca_data, "n_train": len(X)})
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)


def _compute_pca_viz(X_train, X_suggest, feature_names):
    """PCA 2D + Biplot + 累積寄与率"""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    X_all = np.vstack([X_train, X_suggest])
    X_scaled = StandardScaler().fit_transform(X_all)
    n_comp = min(3, X_scaled.shape[1])
    pca = PCA(n_components=n_comp)
    Z = pca.fit_transform(X_scaled)
    n_tr = len(X_train)
    loadings = pca.components_.T
    biplot = [{"name": fn, "pc1": round(float(loadings[i, 0]), 4), "pc2": round(float(loadings[i, 1]), 4) if n_comp > 1 else 0}
              for i, fn in enumerate(feature_names[:20])]
    return {"train_2d": [{"x": round(float(z[0]), 4), "y": round(float(z[1]), 4) if n_comp > 1 else 0} for z in Z[:n_tr]],
            "suggest_2d": [{"x": round(float(z[0]), 4), "y": round(float(z[1]), 4) if n_comp > 1 else 0} for z in Z[n_tr:]],
            "cumulative_variance": [round(float(v), 4) for v in np.cumsum(pca.explained_variance_ratio_)],
            "biplot": biplot}


# ═══════════════════════════════════════════════════════════
# リーケージ検出
# ═══════════════════════════════════════════════════════════

@csrf_exempt
@require_POST
def check_leakage(request, session_id):
    """リーケージ事前チェックAPI"""
    try:
        session = get_object_or_404(AnalysisSession, id=session_id)
        if not session.data_json:
            return JsonResponse({"error": "データが読み込まれていません"}, status=400)
        df = pd.read_json(session.data_json, orient="split")
        target_col = session.target_col
        body = json.loads(request.body) if request.body else {}
        from backend.data.leakage_detector import detect_leakage, check_feature_leakage
        feat_report = check_feature_leakage(df, target_col)
        X_num = df.drop(columns=[target_col], errors="ignore").select_dtypes(include=[np.number])
        y = df[target_col] if target_col in df.columns else None
        sample_resp = None
        if len(X_num.columns) > 0 and len(X_num) >= 5:
            sr = detect_leakage(X_num, y, method=body.get("method", "auto"))
            sample_resp = {"risk_level": sr.risk_level, "risk_score": round(sr.risk_score, 3),
                           "n_suspicious_pairs": sr.n_suspicious_pairs,
                           "recommended_cv": sr.recommended_cv, "cv_reason": sr.cv_reason,
                           "method_used": sr.method_used, "n_groups": sr.n_groups,
                           "group_labels": sr.group_labels.tolist() if sr.group_labels is not None else None}
        return JsonResponse({"feature_check": {"has_risk": feat_report.has_risk, "summary": feat_report.summary,
                                               "warnings": [{"feature": w.feature, "risk": w.risk,
                                                             "reason": w.reason, "score": w.score}
                                                            for w in feat_report.warnings]},
                             "sample_check": sample_resp})
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)


# ═══════════════════════════════════════════════════════════
# EDA
# ═══════════════════════════════════════════════════════════

@csrf_exempt
@require_POST
def run_eda(request, session_id):
    """EDA計算を実行し結果をセッションに保存"""
    session = get_object_or_404(AnalysisSession, id=session_id)
    
    try:
        # データ読み込み
        df = pd.read_csv(session.uploaded_file.path)
        
        # EDA関数呼び出し
        from backend.data.eda import (
            summarize_dataframe,
            compute_column_stats,
            compute_correlation,
            detect_outliers,
            analyze_target,
            compute_vif,
            convert_numpy_to_list,
        )
        
        # 計算実行
        eda_results = {
            "summary": summarize_dataframe(df),
            "column_stats": [cs.__dict__ for cs in compute_column_stats(df)],
            "correlation": compute_correlation(df).to_dict(),
            "outliers": [
                {k: v for k, v in o.__dict__.items() if k != "outlier_indices"}
                for o in detect_outliers(df)
            ],
            "vif": compute_vif(df),
        }
        
        if session.target_col and session.target_col in df.columns:
            eda_results["target_analysis"] = analyze_target(df, session.target_col)
        
        # JSONシリアライズ対応
        eda_results = convert_numpy_to_list(eda_results)
        
        # 結果をセッションに保存
        session.eda_results = eda_results
        session.status = "eda_completed"
        session.save()
        
        return JsonResponse({"success": True, "message": "EDA計算完了"})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({"error": str(e), "trace": traceback.format_exc()}, status=500)


def get_eda_results(request, session_id):
    """EDA結果を取得"""
    session = get_object_or_404(AnalysisSession, id=session_id)
    
    if not session.eda_results:
        return JsonResponse({"error": "EDA結果がありません"}, status=404)
    
    return JsonResponse({"success": True, "results": session.eda_results})
