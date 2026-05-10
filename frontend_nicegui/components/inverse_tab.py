"""
逆解析タブ（SMILES + MolAI + PCAモード）
"""

from nicegui import ui
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import io, base64
from backend.inverse.molai_pca_optimizer import MolAIPCAInverseOptimizer

def render_inverse_panel(state: dict):
    """逆解析パネルの描画"""
    ui.label("🧪 逆解析（SMILES + MolAI + PCA）").classes("text-h6 mt-4")
    
    # 必要なコンポーネントの存在確認
    # best_pipeline -> pipeline_result.best_pipeline or state.get("automl_result").best_pipeline
    best_pipeline = None
    if "best_pipeline" in state:
        best_pipeline = state["best_pipeline"]
    elif "automl_result" in state and state["automl_result"] is not None:
        best_pipeline = getattr(state["automl_result"], "best_pipeline", None)
        
    has_components = all([
        best_pipeline is not None,
        state.get("pca_model"),
        state.get("df_encoded") is not None,
        state.get("smiles_col")
    ])
    
    if not has_components:
        ui.label("⚠️ 逆解析には解析完了とPCAモデルが必要です。").classes("text-warning bg-warning/10 p-2 rounded")
        # デバッグ用に不足モジュールを表示
        missing = []
        if not best_pipeline: missing.append("最適パイプライン")
        if not state.get("pca_model"): missing.append("PCAモデル(EDAタブで次元削減を実行してください)")
        if state.get("df_encoded") is None: missing.append("エンコード済みデータ")
        if not state.get("smiles_col"): missing.append("SMILES列指定")
        if missing:
            ui.label(f"不足: {', '.join(missing)}").classes("text-xs text-grey")
        return

    # stateにbest_pipelineを設定しておく（あとで関数内で使うため）
    if "best_pipeline" not in state:
        state["best_pipeline"] = best_pipeline

    # ── 入力フォーム ──
    with ui.row().classes("w-full items-end gap-4 mb-4"):
        target_input = ui.number(label="目標値 (Target)", format="%.2f").props("dense outlined")
        top_k_input = ui.slider(min=1, max=20, step=1, value=5, label="提案数").props("dense")
        
        ui.button(
            "🔍 探索実行",
            on_click=lambda: _run_inverse_search(state, target_input.value, top_k_input.value),
            color="primary"
        ).props("dense unelevated")

    # ── 結果表示 ──
    if "inverse_results" in state:
        _render_results_table(state["inverse_results"])

def _run_inverse_search(state: dict, target_value: float, top_k: int):
    """探索実行ハンドラ（非同期）"""
    if target_value is None:
        ui.notify("目標値を入力してください", type="warning")
        return
    
    ui.notify("逆解析中...", type="info", timeout=2000)
    
    try:
        # オプティマイザー初期化
        optimizer = MolAIPCAInverseOptimizer(
            predictor_model=state["best_pipeline"],
            pca_model=state["pca_model"],
            reference_descriptors=state["df_encoded"].values,
            reference_smiles=state["df"][state["smiles_col"]].tolist()
        )
        
        # 探索実行
        results = optimizer.search(target_value=target_value, top_k=top_k)
        state["inverse_results"] = results
        
        # 画面更新（再描画のためコンテナリフレッシュ相当の手続きが必要だが、今回はタブ自体の再構成に頼るかrefreshableを使う）
        # ただし、UI全体をrefreshableにするため、呼び出し元で工夫するか、このパネル自体をrefreshableにする必要がある。
        ui.notify(f"✅ 探索完了: {len(results)} 件", type="positive")
        # 簡易的に画面リロードを促す（実際はrefreshableにするのがよい）
        ui.open(ui.run_javascript('window.location.hash = "#inverse"'))
        
    except Exception as e:
        ui.notify(f"❌ エラー: {e}", type="negative", timeout=5000)
        import traceback
        traceback.print_exc()

def _render_results_table(results: list):
    """結果をテーブルと構造式で表示"""
    if not results:
        return

    # RDKit画像生成
    img_map = {}
    for r in results:
        smiles = r["smiles"]
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            img = Draw.MolToImage(mol, size=(150, 150))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode()
            img_map[smiles] = f"image/png;base64,{img_b64}"
        else:
            img_map[smiles] = None

    # テーブルデータ作成
    rows = []
    for r in results:
        rows.append({
            "structure": r["smiles"],
            "smiles": r["smiles"],
            "predicted": f"{r['predicted_value']:.3f}",
            "distance": f"{r['distance']:.4f}",
        })

    columns = [
        {"name": "structure", "label": "構造", "field": "structure"},
        {"name": "smiles", "label": "SMILES", "field": "smiles"},
        {"name": "predicted", "label": "予測値", "field": "predicted", "sortable": True},
        {"name": "distance", "label": "記述子距離", "field": "distance", "sortable": True},
    ]

    # テーブル描画
    with ui.table(columns=columns, rows=rows, row_key="smiles").props("dense flat bordered"):
        with ui.add_slot("body-cell-structure"):
            def cell_structure(props):
                smiles = props["row"]["smiles"]
                img_src = img_map.get(smiles)
                if img_src:
                    ui.image(img_src).classes("w-24 h-24 object-contain")
                else:
                    ui.label("Invalid").classes("text-red-500")
            cell_structure()

    # 補足
    ui.markdown(
        "> 💡 **探索ロジック**: PCA空間で目標値に近づくベクトルを探索し、"
        "それに最も近い**実在する分子**を検索しています。"
        "これにより、化学的に妥当な構造のみが提案されます。"
    ).classes("text-caption text-grey-6 mt-2")
