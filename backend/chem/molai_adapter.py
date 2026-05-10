"""
backend/chem/molai_adapter.py

MolAI: CNN Encoder + GRU Decoder SMILES オートエンコーダーによる
分子潜在ベクトル記述子の生成アダプター。

論文: Mahdizadeh & Eriksson, J. Chem. Inf. Model. 2025
     DOI: 10.1021/acs.jcim.5c00491

アーキテクチャ:
  - Encoder: 1D CNN (キャラクターレベルSMILES → 固定長潜在ベクトル)
  - Decoder: GRU (潜在ベクトル → SMILES 再構成)
  - 入力: SMILES 文字列 (最大 MAX_SMILES_LEN 文字)
  - 出力: n_components 次元 (PCA で圧縮した潜在ベクトル)

Implements: MolAI §2 Architecture, §3.1 Encoder
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from backend.chem.base import BaseChemAdapter, DescriptorMetadata, DescriptorResult

logger = logging.getLogger(__name__)

# ── SMILES トークナイザー定数 ────────────────────────────────────────────
# 論文 §2.1: キャラクターレベルトークナイズ, Br/Cl は一括トークン
MAX_SMILES_LEN = 111  # 論文 §2.1: "SMILES longer than 111 characters ... excluded"
PAD_CHAR = " "        # padding character

_SPECIAL_TOKENS = ["Br", "Cl"]  # 2文字トークン（先に置換）
_CHARSET: list[str] = (
    list("CNOSFPIBrCl")          # 元素記号
    + list("=#@\\/%()[].")       # 結合・括弧系
    + list("0123456789")         # 数字
    + list("+-hHcsnoSF")         # 電荷・芳香族等
    + [PAD_CHAR]
)
_CHARSET_UNIQUE = list(dict.fromkeys(_CHARSET))  # 重複除去・順序保持
_CHAR2IDX: dict[str, int] = {c: i for i, c in enumerate(_CHARSET_UNIQUE)}
VOCAB_SIZE = len(_CHARSET_UNIQUE)


def _tokenize_smiles(smiles: str) -> list[str]:
    """SMILES を文字トークンのリストに変換する。Br/Cl は単一トークンとして扱う。
    
    Implements: 論文 §2.1 Tokenization
    """
    tokens: list[str] = []
    i = 0
    while i < len(smiles):
        matched = False
        for tok in _SPECIAL_TOKENS:
            if smiles[i:i + len(tok)] == tok:
                tokens.append(tok)
                i += len(tok)
                matched = True
                break
        if not matched:
            tokens.append(smiles[i])
            i += 1
    return tokens


def _smiles_to_onehot(smiles: str) -> np.ndarray:
    """SMILES → one-hot エンコード行列 (MAX_SMILES_LEN x VOCAB_SIZE)。
    
    Implements: 論文 §2.1 Encoding
    """
    tokens = _tokenize_smiles(smiles)[:MAX_SMILES_LEN]
    # パディング
    tokens += [PAD_CHAR] * (MAX_SMILES_LEN - len(tokens))
    mat = np.zeros((MAX_SMILES_LEN, VOCAB_SIZE), dtype=np.float32)
    for j, tok in enumerate(tokens):
        idx = _CHAR2IDX.get(tok, _CHAR2IDX[PAD_CHAR])
        mat[j, idx] = 1.0
    return mat


# ── PyTorch モデル定義 ────────────────────────────────────────────────────
def _build_encoder(latent_dim: int = 256):
    """CNN Encoder: (batch, seq_len, vocab) → (batch, latent_dim)
    
    Implements: 論文 §2.2 Encoder Architecture (1D CNN layers)
    """
    try:
        import torch
        import torch.nn as nn

        class MolAIEncoder(nn.Module):
            def __init__(self, vocab_size: int, latent_dim: int):
                super().__init__()
                self.conv1 = nn.Conv1d(vocab_size, 9,  kernel_size=9)
                self.conv2 = nn.Conv1d(9,          9,  kernel_size=9)
                self.conv3 = nn.Conv1d(9,          10, kernel_size=11)
                self.flatten = nn.Flatten()
                # seq after 3 convs: MAX_SMILES_LEN - 8 - 8 - 10 = 85
                flat_dim = 10 * (MAX_SMILES_LEN - 8 - 8 - 10)  # = 10 * 85 = 850
                self.fc = nn.Linear(flat_dim, latent_dim)
                self.relu = nn.ReLU()

            def forward(self, x):  # x: (batch, seq_len, vocab_size)
                x = x.permute(0, 2, 1)          # → (batch, vocab_size, seq_len)
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                x = self.relu(self.conv3(x))
                x = self.flatten(x)
                return self.fc(x)

        return MolAIEncoder(VOCAB_SIZE, latent_dim)
    except ImportError:
        return None


def _build_decoder(latent_dim: int = 256):
    """GRU Decoder: (batch, latent_dim) → (batch, MAX_SMILES_LEN, VOCAB_SIZE)

    Implements: 論文 §2.2 Decoder Architecture (GRU seq2seq)
    潜在ベクトルを各タイムステップにrepeat → 2層GRU → FC → SMILES one-hot
    """
    try:
        import torch
        import torch.nn as nn

        class MolAIDecoder(nn.Module):
            def __init__(self, vocab_size: int, latent_dim: int,
                         gru_hidden: int = 488, gru_layers: int = 3):
                super().__init__()
                self.seq_len = MAX_SMILES_LEN
                self.latent_dim = latent_dim
                self.gru_hidden = gru_hidden
                # 潜在ベクトル → GRU初期隠れ状態
                self.latent2hidden = nn.Linear(latent_dim, gru_hidden * gru_layers)
                self.gru_layers = gru_layers
                # GRU: 入力は潜在ベクトル(各ステップにrepeat)
                self.gru = nn.GRU(
                    input_size=latent_dim,
                    hidden_size=gru_hidden,
                    num_layers=gru_layers,
                    batch_first=True,
                    dropout=0.0,
                )
                # GRU出力 → one-hot分布
                self.fc_out = nn.Linear(gru_hidden, vocab_size)

            def forward(self, z):
                """z: (batch, latent_dim) → logits: (batch, seq_len, vocab_size)"""
                batch_size = z.size(0)
                # 初期隠れ状態
                h0 = self.latent2hidden(z)
                h0 = h0.view(self.gru_layers, batch_size, self.gru_hidden)
                # 潜在ベクトルを全タイムステップにrepeat
                z_repeated = z.unsqueeze(1).repeat(1, self.seq_len, 1)
                # GRU
                gru_out, _ = self.gru(z_repeated, h0)
                # FC → logits
                logits = self.fc_out(gru_out)
                return logits

            def decode_to_smiles(self, z):
                """潜在ベクトル → SMILES文字列のリスト"""
                self.eval()
                with torch.no_grad():
                    logits = self.forward(z)
                    # Greedy argmax
                    indices = torch.argmax(logits, dim=-1)  # (batch, seq_len)
                idx2char = {i: c for c, i in _CHAR2IDX.items()}
                results = []
                for row in indices.cpu().numpy():
                    chars = []
                    for idx in row:
                        c = idx2char.get(int(idx), "")
                        if c == PAD_CHAR:
                            break
                        chars.append(c)
                    results.append("".join(chars))
                return results

        return MolAIDecoder(VOCAB_SIZE, latent_dim)
    except ImportError:
        return None


# ── アダプター本体 ─────────────────────────────────────────────────────────
class MolAIAdapter(BaseChemAdapter):
    """MolAI SMILES オートエンコーダー記述子アダプター。

    CNN Encoder で SMILES を高次元潜在ベクトルに変換し、
    PCA で n_components 次元に圧縮して記述子として返す。

    Implements: F-MolAI | 論文: §2 Architecture | 式 (latent = Encoder(SMILES))
    API:
        n_components (int): PCA 出力次元数（デフォルト 32）
        latent_dim (int): エンコーダー出力次元（デフォルト 256）
    前提: torch >= 2.0 がインストールされていること
    """

    def __init__(self, n_components: int | str = "auto", latent_dim: int = 256):
        """
        Args:
            n_components: PCA出力次元数。
                - int: 固定次元数
                - "auto": 累積寄与率95%超えの最小次元数を自動決定
            latent_dim: エンコーダー出力次元
        """
        self.n_components = n_components
        self.latent_dim = latent_dim
        self._encoder = None
        self._decoder = None
        self._pca = None
        self._scaler = None

    @property
    def name(self) -> str:
        return "molai"

    @property
    def description(self) -> str:
        return (
            "MolAI CNN+GRU Autoencoder (Mahdizadeh & Eriksson, JCIM 2025)。"
            f"SMILES を {self.latent_dim} 次元潜在ベクトルに変換後、"
            f"PCA で {self.n_components} 次元に圧縮します。"
        )

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
            import sklearn  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_encoder(self):
        """エンコーダーを遅延初期化して返す。"""
        if self._encoder is None:
            self._encoder = _build_encoder(self.latent_dim)
            if self._encoder is None:
                raise RuntimeError("torch がインストールされていません。")
            self._encoder.eval()
        return self._encoder

    def compute(
        self,
        smiles_list: list[str],
        selected_descriptors: list[str] | None = None,
        **kwargs: Any,
    ) -> DescriptorResult:
        """SMILES リストから MolAI 潜在ベクトル記述子を計算する。

        Implements: F-MolAI | 論文 §2.2 Encoder, §3.1 Latent Space
        Args:
            smiles_list: 入力 SMILES のリスト
            selected_descriptors: 使用する列名（None = 全件）
        Returns:
            DescriptorResult: molai_pc1, molai_pc2, ... の DataFrame
        """
        self._require_available()
        logger.info(f"🤖 MolAI CNN Encoder + PCA 計算を開始します（{len(smiles_list)}分子）...")
        import torch
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        encoder = self._get_encoder()
        failed_indices: list[int] = []
        latent_vecs: list[np.ndarray] = []

        # ── Step 1: SMILES → one-hot → latent vector ──
        for i, smi in enumerate(smiles_list):
            try:
                if not smi or len(_tokenize_smiles(smi)) > MAX_SMILES_LEN:
                    raise ValueError(f"SMILES が長すぎます (max {MAX_SMILES_LEN} chars): {smi[:30]}...")
                oh = _smiles_to_onehot(smi)
                t = torch.tensor(oh[np.newaxis, :, :], dtype=torch.float32)
                with torch.no_grad():
                    lv = encoder(t).numpy().flatten()
                latent_vecs.append(lv)
            except Exception as e:
                logger.warning("MolAI エンコードに失敗: idx=%d, smi=%s, err=%s", i, smi[:30], e)
                failed_indices.append(i)
                latent_vecs.append(np.full(self.latent_dim, np.nan))

        latent_mat = np.array(latent_vecs, dtype=np.float32)

        # ── Step 2: PCA で次元削減 ──
        valid_mask = ~np.isnan(latent_mat).any(axis=1)
        n_valid = int(valid_mask.sum())

        if n_valid < 2:
            n_comp = min(
                self.n_components if isinstance(self.n_components, int) else 32,
                latent_mat.shape[1],
            )
            pc_mat = np.full((len(smiles_list), n_comp), np.nan, dtype=np.float32)
            col_names = [f"molai_pc{k + 1}" for k in range(n_comp)]
            df = pd.DataFrame(pc_mat, columns=col_names)
        else:
            scaler = StandardScaler()
            X_valid = scaler.fit_transform(latent_mat[valid_mask])
            self._scaler = scaler  # 逆変換用に保持

            if self.n_components == "auto":
                max_comp = min(n_valid, latent_mat.shape[1])
                pca_full = PCA(n_components=max_comp, random_state=42)
                pca_full.fit(X_valid)
                cum_ratio = np.cumsum(pca_full.explained_variance_ratio_)
                n_95 = int(np.searchsorted(cum_ratio, 0.95) + 1)
                n_comp = min(n_95, max_comp)
                n_comp = max(n_comp, 2)
                logger.info(
                    f"MolAI PCA自動最適化: 累積寄与率95%%到達={n_95}次元, "
                    f"最大={max_comp}次元 → 採用={n_comp}次元 "
                    f"(累積寄与率={cum_ratio[n_comp - 1]:.1%})"
                )
            else:
                n_comp = min(self.n_components, n_valid, latent_mat.shape[1])

            pca = PCA(n_components=n_comp, random_state=42)
            pc_mat = np.full((len(smiles_list), n_comp), np.nan, dtype=np.float32)
            pc_mat[valid_mask] = pca.fit_transform(X_valid).astype(np.float32)
            self._pca = pca

            col_names = [f"molai_pc{k + 1}" for k in range(n_comp)]
            df = pd.DataFrame(pc_mat, columns=col_names)

        # selected_descriptors でフィルタリング
        if selected_descriptors:
            available = [c for c in selected_descriptors if c in df.columns]
            df = df[available] if available else df

        return DescriptorResult(
            descriptors=df,
            smiles_list=smiles_list,
            failed_indices=failed_indices,
            adapter_name=self.name,
            metadata={
                "latent_dim": self.latent_dim,
                "n_components": n_comp,
                "n_failed": len(failed_indices),
            },
        )

    # ─── エンコード (生の潜在ベクトル) ───────────────────────────────
    def encode_raw(self, smiles_list: list[str]) -> np.ndarray:
        """SMILES → 生の潜在ベクトル（PCA前）を返す。"""
        self._require_available()
        import torch

        encoder = self._get_encoder()
        vecs = []
        for smi in smiles_list:
            try:
                oh = _smiles_to_onehot(smi)
                t = torch.tensor(oh[np.newaxis, :, :], dtype=torch.float32)
                with torch.no_grad():
                    lv = encoder(t).numpy().flatten()
                vecs.append(lv)
            except Exception:
                vecs.append(np.full(self.latent_dim, np.nan))
        return np.array(vecs, dtype=np.float32)

    # ─── デコード (潜在ベクトル → SMILES) ────────────────────────────
    def decode(self, latent_vectors: np.ndarray) -> list[str]:
        """生の潜在ベクトル（PCA前）→ SMILES文字列のリスト。

        Args:
            latent_vectors: (N, latent_dim) の numpy配列
        Returns:
            SMILES文字列のリスト
        """
        self._require_available()
        import torch

        decoder = self._get_decoder()
        z = torch.tensor(latent_vectors, dtype=torch.float32)
        return decoder.decode_to_smiles(z)

    def _get_decoder(self):
        """デコーダーを遅延初期化して返す。"""
        if self._decoder is None:
            self._decoder = _build_decoder(self.latent_dim)
            if self._decoder is None:
                raise RuntimeError("torch がインストールされていません。")
            self._decoder.eval()
        return self._decoder

    # ─── PCA逆変換 + デコード ────────────────────────────────────────
    def pca_inverse_transform(self, pc_vectors: np.ndarray) -> np.ndarray:
        """PCA空間のベクトル → 生の潜在ベクトルに逆変換。

        Args:
            pc_vectors: (N, n_pca_components) の numpy配列
        Returns:
            (N, latent_dim) の numpy配列
        """
        if self._pca is None or self._scaler is None:
            raise RuntimeError("先に compute() を実行してPCAとScalerを学習させてください。")
        # PCA逆変換 → スケーラー逆変換
        X_scaled = self._pca.inverse_transform(pc_vectors)
        return self._scaler.inverse_transform(X_scaled)

    def decode_from_pca(self, pc_vectors: np.ndarray) -> list[str]:
        """PCA空間のベクトル → SMILES文字列。逆解析の核心メソッド。

        流れ: PCA空間 → PCA逆変換 → 潜在空間 → GRU Decoder → SMILES

        Args:
            pc_vectors: (N, n_pca_components) の numpy配列
        Returns:
            SMILES文字列のリスト
        """
        latent = self.pca_inverse_transform(pc_vectors)
        return self.decode(latent)

    # ─── オートエンコーダー学習 ──────────────────────────────────────
    def train_autoencoder(
        self,
        smiles_list: list[str],
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        progress_callback: Any | None = None,
    ) -> dict:
        """Encoder + Decoder を教師なし学習する。

        Args:
            smiles_list: 学習用SMILES
            epochs: エポック数
            batch_size: バッチサイズ
            lr: 学習率
            progress_callback: (epoch, loss) を受け取るコールバック関数
        Returns:
            {"final_loss": float, "reconstruction_accuracy": float}
        """
        self._require_available()
        import torch
        import torch.nn as nn

        # one-hot行列の準備
        valid_data = []
        for smi in smiles_list:
            try:
                if smi and len(_tokenize_smiles(smi)) <= MAX_SMILES_LEN:
                    oh = _smiles_to_onehot(smi)
                    valid_data.append(oh)
            except Exception:
                continue

        if len(valid_data) < 2:
            raise ValueError(f"有効なSMILESが不足しています ({len(valid_data)}件)")

        X = np.array(valid_data, dtype=np.float32)
        X_tensor = torch.tensor(X)

        encoder = self._get_encoder()
        decoder = self._get_decoder()

        # 学習モードに設定
        encoder.train()
        decoder.train()

        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=lr,
        )
        criterion = nn.CrossEntropyLoss()

        # ターゲット: one-hotのargmax（各位置のクラスインデックス）
        target_indices = torch.argmax(X_tensor, dim=-1)  # (N, seq_len)

        final_loss = 0.0
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            # シャッフル
            perm = torch.randperm(len(X_tensor))
            for start in range(0, len(X_tensor), batch_size):
                idx = perm[start:start + batch_size]
                batch_x = X_tensor[idx]
                batch_target = target_indices[idx]

                # Forward
                z = encoder(batch_x)
                logits = decoder(z)  # (batch, seq_len, vocab_size)

                # Loss: Cross-Entropy (logitsとtargetのクラスインデックス)
                loss = criterion(
                    logits.reshape(-1, VOCAB_SIZE),
                    batch_target.reshape(-1),
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            final_loss = avg_loss

            if progress_callback:
                progress_callback(epoch + 1, avg_loss)
            if (epoch + 1) % 10 == 0:
                logger.info(f"MolAI AE学習: epoch={epoch + 1}/{epochs}, loss={avg_loss:.4f}")

        # 学習完了 → eval mode
        encoder.eval()
        decoder.eval()

        # 再構成精度を計算
        with torch.no_grad():
            z_all = encoder(X_tensor)
            logits_all = decoder(z_all)
            pred_indices = torch.argmax(logits_all, dim=-1)
            accuracy = float((pred_indices == target_indices).float().mean())

        logger.info(f"MolAI AE学習完了: loss={final_loss:.4f}, 再構成精度={accuracy:.1%}")
        return {"final_loss": final_loss, "reconstruction_accuracy": accuracy}

    def get_descriptor_names(self) -> list[str]:
        """利用可能な記述子名（molai_pc1 〜 molai_pcN）を返す。"""
        n = self.n_components if isinstance(self.n_components, int) else 32
        return [f"molai_pc{k + 1}" for k in range(n)]

    def get_descriptors_metadata(self) -> list[DescriptorMetadata]:
        """各 PCA 成分のメタデータを返す。"""
        n = self.n_components if isinstance(self.n_components, int) else 32
        return [
            DescriptorMetadata(
                name=f"molai_pc{k + 1}",
                meaning=f"MolAI 潜在空間 第{k + 1}主成分 (PCA圧縮)",
                is_count=False,
                is_binary=False,
                description=(
                    "MolAI CNN Encoder で得た潜在ベクトルを PCA 圧縮した成分。"
                    "論文: Mahdizadeh & Eriksson, JCIM 2025, DOI: 10.1021/acs.jcim.5c00491"
                ),
            )
            for k in range(n)
        ]
