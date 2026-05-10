"""
frontend_streamlit/components/auto_form.py
"""
import streamlit as st
from typing import Dict, Any

def render_streamlit_auto_form(schema: Dict[str, Any], key: str = "auto") -> Dict[str, Any]:
    props = schema.get("properties", {})
    groups = schema.get("ui:groups", {})
    layout = schema.get("ui:layout", "default")

    # 初期化（初回のみ）
    for n, p in props.items():
        sk = f"{key}_{n}"
        if sk not in st.session_state:
            st.session_state[sk] = p.get("default")

    if schema.get("ui:searchable"):
        q = st.text_input("🔍 設定項目を検索", key=f"{key}_search").lower()
    else:
        q = ""

    def _render(n: str, p: Dict):
        if q and q not in n.lower() and q not in str(st.session_state.get(f"{key}_{n}", "")).lower():
            return
        w = p.get("ui:widget", p.get("type", "string"))
        sk = f"{key}_{n}"
        
        if w in ("integer", "slider"):
            st.session_state[sk] = st.slider(n, min_value=p.get("minimum",0), max_value=p.get("maximum",100),
                                             value=st.session_state[sk] or 0, step=1)
        elif w == "number":
            st.session_state[sk] = st.number_input(n, min_value=p.get("minimum"), max_value=p.get("maximum"),
                                                   value=float(st.session_state[sk] or 0.0), step=0.01)
        elif w in ("boolean", "toggle"):
            st.session_state[sk] = st.checkbox(n, value=bool(st.session_state[sk]))
        elif w == "select":
            opts = p.get("enum", [])
            idx = opts.index(st.session_state[sk]) if st.session_state[sk] in opts else 0
            st.session_state[sk] = st.selectbox(n, options=opts, index=idx)
        else:
            st.session_state[sk] = st.text_input(n, value=str(st.session_state[sk] or ""))

    if layout == "accordion" and groups:
        for g, fields in groups.items():
            with st.expander(f"🔹 {g.replace('_',' ').title()}", expanded=True):
                for f in fields: 
                    if f in props:
                        _render(f, props[f])
    else:
        for f, p in props.items(): 
            _render(f, p)

    return {n: st.session_state[f"{key}_{n}"] for n in props}

def render_progress_streamlit(task_id: str):
    from backend.pipeline.tasks.manager import get_task_manager
    mgr = get_task_manager()
    
    if task_id not in st.session_state:
        st.session_state[task_id] = {"status": "queued", "progress": 0.0}
    
    state = mgr.get_status(task_id)
    if not state:
        return
        
    s_state = st.session_state[task_id]
    s_state["status"] = state["status"]
    
    if state["status"] == "success":
        s_state["progress"] = 1.0
        st.progress(s_state["progress"])
        st.success("✅ 計算完了")
    elif state["status"] == "failed":
        s_state["progress"] = 1.0
        st.progress(s_state["progress"])
        st.error(f"❌ エラー発生: {state.get('error', 'Unknown')}")
    else:
        if s_state["progress"] < 0.9:
            s_state["progress"] += 0.1
        st.progress(s_state["progress"])
        st.info(f"状態: {s_state['status']} ⏳")
