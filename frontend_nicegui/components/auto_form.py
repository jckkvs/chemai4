"""
frontend_nicegui/components/auto_form.py
"""
from nicegui import ui
from typing import Dict, Any, Callable, Optional

class NiceGUIAutoForm:
    def __init__(self, schema: Dict[str, Any], initial: Dict[str, Any] = None,
                 on_change: Optional[Callable[[Dict], None]] = None):
        self.schema = schema
        self.values = initial or {}
        self.on_change = on_change
        self._widgets: Dict[str, Any] = {}
        self._container: Optional[ui.column] = None
        self._build()

    def _build(self):
        self._container = ui.column().classes("w-full gap-4 p-4")
        with self._container:
            if self.schema.get("ui:searchable"):
                self._search = ui.input("🔍 設定項目を検索").classes("w-full")
                self._search.on("update:model-value", lambda e: self._filter(e.value))
            
            layout = self.schema.get("ui:layout")
            groups = self.schema.get("ui:groups", {})
            props = self.schema.get("properties", {})

            if layout == "accordion" and groups:
                for grp, fields in groups.items():
                    with ui.expansion(grp.replace("_", " ").title(), icon="tune").classes("w-full"):
                        for f in fields:
                            if f in props: self._field(f, props[f])
            else:
                for f, p in props.items():
                    self._field(f, p)

    def _field(self, name: str, prop: Dict):
        w_type = prop.get("ui:widget", prop.get("type", "string"))
        default = self.values.get(name, prop.get("default"))

        if w_type in ("integer", "slider"):
            w = ui.slider(min=prop.get("minimum",0), max=prop.get("maximum",100), step=1, value=default or 0)
        elif w_type == "number":
            w = ui.number(value=default or 0.0, step=prop.get("step", 0.01))
            if "minimum" in prop: w.props(f'min="{prop["minimum"]}"')
            if "maximum" in prop: w.props(f'max="{prop["maximum"]}"')
        elif w_type in ("boolean", "toggle"):
            w = ui.switch(value=bool(default))
        elif w_type == "select":
            opts = prop.get("enum", [])
            w = ui.select(opts, value=default if default in opts else (opts[0] if opts else None))
        else:
            w = ui.input(value=str(default or ""))

        w.on("update:model-value", lambda _: self._emit())
        w.bind_value(self.values, name)
        self._widgets[name] = w
        ui.label(prop.get("ui:help", "")).classes("text-xs text-gray-500 -mt-2")

    def _filter(self, q: str):
        q = q.lower()
        for n, w in self._widgets.items():
            vis = q == "" or q in n.lower() or q in str(self.values.get(n,"")).lower()
            w.set_visibility(vis)

    def _emit(self):
        if self.on_change: self.on_change(self.values.copy())

    def get_values(self) -> Dict[str, Any]: 
        return self.values.copy()

def render_progress_bar(task_id: str):
    # This expects a 'get_task_state' and 'get_task_progress' function
    from backend.pipeline.tasks.manager import get_task_manager
    mgr = get_task_manager()
    
    progress = ui.linear_progress(value=0.0).classes("w-full")
    status_label = ui.label("Queued").classes("text-sm text-gray-600 mt-2")
    
    def _update():
        state = mgr.get_status(task_id)
        if not state:
            return
        
        # Estimate progress manually if state is just queued/success since our manager isn't granular yet
        if state["status"] == "success":
            progress.set_value(1.0)
            status_label.set_text("Completed ✅")
        elif state["status"] == "failed":
            progress.set_value(1.0)
            status_label.set_text(f"Failed ❌: {state.get('error', 'unknown')}")
        else:
            # Fake running progress
            curr = progress.value
            if curr < 0.9:
                progress.set_value(curr + 0.05)
            status_label.set_text("Running ⏳")
            
    ui.timer(1.0, _update, active=True)
