/**
 * ChemAI ML Studio - Django Frontend JavaScript
 */

// ── タブ切り替え ──────────────────────────────
function switchTab(tabGroup, tabId) {
    // タブボタン
    document.querySelectorAll(`[data-tab-group="${tabGroup}"] .tab`).forEach(t => {
        t.classList.toggle('active', t.dataset.tab === tabId);
    });
    // タブコンテンツ
    document.querySelectorAll(`[data-tab-group="${tabGroup}"] .tab-content`).forEach(c => {
        c.classList.toggle('active', c.id === tabId);
    });
}

// ── ファイルアップロード（ドラッグ&ドロップ） ──
function initDropzone(dropzoneEl, sessionId) {
    if (!dropzoneEl) return;

    const fileInput = dropzoneEl.querySelector('input[type="file"]');

    dropzoneEl.addEventListener('click', () => fileInput.click());

    dropzoneEl.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzoneEl.classList.add('dragover');
    });

    dropzoneEl.addEventListener('dragleave', () => {
        dropzoneEl.classList.remove('dragover');
    });

    dropzoneEl.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzoneEl.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            uploadFile(files[0], sessionId);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            uploadFile(fileInput.files[0], sessionId);
        }
    });
}

async function uploadFile(file, sessionId) {
    const formData = new FormData();
    formData.append('file', file);

    const statusEl = document.getElementById('upload-status');
    if (statusEl) {
        statusEl.innerHTML = '<div class="spinner"></div> アップロード中...';
        statusEl.style.display = 'block';
    }

    try {
        const resp = await fetch(`/api/session/${sessionId}/upload/`, {
            method: 'POST',
            body: formData,
        });
        const data = await resp.json();

        if (data.success) {
            if (statusEl) {
                statusEl.innerHTML = `<div class="alert alert-success">✅ ${data.n_rows}行 × ${data.n_cols}列 読み込み完了</div>`;
            }
            renderPreview(data);
            // ページをリロードして次のステップへ
            setTimeout(() => location.reload(), 1000);
        } else {
            if (statusEl) {
                statusEl.innerHTML = `<div class="alert alert-error">❌ ${data.error}</div>`;
            }
        }
    } catch (err) {
        if (statusEl) {
            statusEl.innerHTML = `<div class="alert alert-error">❌ ネットワークエラー: ${err.message}</div>`;
        }
    }
}

// ── サンプルデータ読み込み ──────────────────────
async function loadSample(sessionId, type, includeSmiles = true) {
    const btn = event.target;
    btn.disabled = true;
    btn.textContent = '読み込み中...';

    try {
        const resp = await fetch(`/api/session/${sessionId}/sample/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ type, include_smiles: includeSmiles }),
        });
        const data = await resp.json();
        if (data.success) {
            location.reload();
        } else {
            alert(`エラー: ${data.error}`);
        }
    } catch (err) {
        alert(`ネットワークエラー: ${err.message}`);
    } finally {
        btn.disabled = false;
    }
}

// ── 列設定 ────────────────────────────────────
async function setColumns(sessionId) {
    const targetCol = document.getElementById('target-col').value;
    const smilesCol = document.getElementById('smiles-col').value;
    const taskType = document.getElementById('task-type').value;

    try {
        const resp = await fetch(`/api/session/${sessionId}/columns/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                target_col: targetCol,
                smiles_col: smilesCol,
                task_type: taskType,
            }),
        });
        const data = await resp.json();
        if (data.success) {
            location.reload();
        } else {
            alert(`エラー: ${data.error}`);
        }
    } catch (err) {
        alert(`ネットワークエラー: ${err.message}`);
    }
}

// ── プレビューテーブル描画 ──────────────────────
function renderPreview(data) {
    const container = document.getElementById('data-preview');
    if (!container || !data.preview) return;

    let html = '<table class="data-table"><thead><tr>';
    data.columns.forEach(col => { html += `<th>${col}</th>`; });
    html += '</tr></thead><tbody>';
    data.preview.forEach(row => {
        html += '<tr>';
        data.columns.forEach(col => {
            const val = row[col];
            html += `<td>${val !== null && val !== undefined ? val : '—'}</td>`;
        });
        html += '</tr>';
    });
    html += '</tbody></table>';
    container.innerHTML = html;
}

// ── 初期化 ────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    // ドロップゾーン初期化
    const dz = document.querySelector('.dropzone');
    const sessionId = document.body.dataset.sessionId;
    if (dz && sessionId) {
        initDropzone(dz, sessionId);
    }
});
