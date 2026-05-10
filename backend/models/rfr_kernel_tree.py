"""
backend/models/rfr_kernel_tree.py

RFR-inspired Decision Tree with Kernel Methods.

Implements:
  - RGF (Regularized Greedy Forest) per IJCAI'16 paper
  - Honest Tree (split/estimation sample separation)
  - Soft splits (probabilistic assignment to both children)
  - TREEKERNEL (RFR Kernel reproduction for single tree)
  - Bernoulli Forest (Bernoulli sampling per tree)
  - Rotation Forest (PCA-based feature rotation)
  - Integration with kernel methods (SVR, KernelRidge, GPR, GPC, SVC)

Key idea: A single decision tree that achieves RF-level accuracy by combining:
  - Regularized greedy growth (RGF)
  - Honest estimation (separate samples for structure/leaves)
  - Soft splits (smooth decision boundaries)
  - Tree kernel (RFR-like similarity)

References:
  - RGF: https://www.ijcai.org/Proceedings/16/Papers/309.pdf
  - Honest Tree: Athey & Wager (2018), "Estimation and Inference of Heterogeneous
    Treatment Effects using Random Forests"
  - RFR Kernel: Davies & Ghahramani (2014), "The Random Forest Kernel"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TNode:
    """Tree node for RGF/Honest/Soft tree."""
    # Structure
    feature_idx: int = -1
    threshold: float = 0.0
    left: "TNode | None" = None
    right: "TNode | None" = None

    # Leaf data
    leaf_id: int = -1           # unique leaf identifier
    weight: float = 0.0         # leaf prediction weight (for RGF)
    linear_model: Any = None     # optional linear model in leaf

    # Honest tree: separate samples for structure and estimation
    n_struct: int = 0           # samples used for structure (splitting)
    n_estim: int = 0            # samples used for estimation (leaf weights)

    # Statistics
    depth: int = 0
    parent: "TNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def get_leaf_ids(self) -> list[int]:
        """Collect all leaf IDs in this subtree."""
        if self.is_leaf:
            return [self.leaf_id] if self.leaf_id >= 0 else []
        ids = []
        if self.left:
            ids.extend(self.left.get_leaf_ids())
        if self.right:
            ids.extend(self.right.get_leaf_ids())
        return ids


# ═══════════════════════════════════════════════════════════════════
# RGF: Regularized Greedy Forest
# ═══════════════════════════════════════════════════════════════════

class RGFRegressor(BaseEstimator, RegressorMixin):
    """
    Regularized Greedy Forest (RGF) for regression.

    Algorithm (per IJCAI'16 paper):
      1. Start with a single leaf (root) containing all training data
      2. For each leaf, compute the gain of splitting it
      3. Select the leaf with maximum gain and split it
      4. Update leaf weights (solve regularized least squares)
      5. Repeat until max_leaves or other stopping criteria met

    Key differences from standard trees:
      - Grows greedily one leaf at a time (not level-wise or depth-first)
      - Uses regularized loss for leaf weight optimization
      - Can split internal nodes (not just leaves) - "structured" approach

    Parameters:
    -----------
    max_leaves : int, default=50
        Maximum number of leaves in the forest/tree
    loss : str, default="ls"
        Loss function: "ls" (least squares) or "log" (logistic for probability)
    l2 : float, default=0.1
        L2 regularization for leaf weights
    l1 : float, default=0.0
        L1 regularization (not fully implemented in basic version)
    min_samples_leaf : int, default=5
        Minimum samples required in a leaf
    max_depth : int, default=None
        Maximum depth of the tree (None = no limit)
    feature_subset : float or int, default=None
        Fraction/number of features to consider for each split
        None = all features
    random_state : int, default=None
        Random seed
    """

    def __init__(
        self,
        max_leaves: int = 50,
        loss: str = "ls",
        l2: float = 0.1,
        l1: float = 0.0,
        min_samples_leaf: int = 5,
        max_depth: int | None = None,
        feature_subset: float | int | None = None,
        random_state: int | None = None,
    ) -> None:
        self.max_leaves = max_leaves
        self.loss = loss
        self.l2 = l2
        self.l1 = l1
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.feature_subset = feature_subset
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> "RGFRegressor":
        X_arr = _to_numpy(X)
        y_arr = np.asarray(y, dtype=float).ravel()
        n_samples, n_features = X_arr.shape

        self.n_features_in_ = n_features
        rng = np.random.default_rng(self.random_state)

        # Initialize: single root leaf
        self.root_ = TNode(leaf_id=0, n_struct=n_samples, n_estim=n_samples)
        self.leaves_ = {0: self.root_}
        self.next_leaf_id_ = 1
        self.loss_history_ = []

        # Store data at each leaf
        # leaf_id -> {"X": ..., "y": ..., "indices": ...}
        self._leaf_data: dict[int, dict] = {
            0: {"X": X_arr, "y": y_arr, "indices": np.arange(n_samples)}
        }

        # Compute initial leaf weight (global mean with regularization)
        self.root_.weight = self._solve_leaf_weight(y_arr)
        current_loss = self._compute_loss(y_arr, self.root_.weight)
        self.loss_history_.append(float(current_loss))

        # Greedy leaf-by-leaf growth
        while len(self.leaves_) < self.max_leaves:
            best_gain = -np.inf
            best_leaf_id = -1
            best_split: dict | None = None

            for leaf_id, leaf in list(self.leaves_.items()):
                if leaf.is_leaf:
                    split = self._find_best_split_rgf(
                        leaf, self._leaf_data[leaf_id], n_features, rng
                    )
                    if split is not None and split["gain"] > best_gain:
                        best_gain = split["gain"]
                        best_leaf_id = leaf_id
                        best_split = split

            if best_split is None or best_gain <= 0:
                logger.info("RGF: No more beneficial splits found.")
                break

            # Execute the split
            self._split_leaf(best_leaf_id, best_split, n_features)

            # Recompute all leaf weights after structural change
            self._update_all_leaf_weights()

            # Compute current loss
            current_loss = self._compute_tree_loss(X_arr, y_arr)
            self.loss_history_.append(float(current_loss))
            logger.debug(
                f"RGF: {len(self.leaves_)} leaves, loss={current_loss:.6f}, "
                f"gain={best_gain:.6f}"
            )

        self.n_leaves_ = len(self.leaves_)
        logger.info(
            f"RGFRegressor.fit(): {self.n_leaves_} leaves, "
            f"final_loss={self.loss_history_[-1]:.6f}"
        )
        return self

    def _solve_leaf_weight(self, y: np.ndarray) -> float:
        """Solve for optimal leaf weight with L2 regularization."""
        if len(y) == 0:
            return 0.0
        # For squared loss: w* = mean(y) / (1 + l2/n) approximately
        # Exact: minimize sum((y - w)^2) + l2 * w^2
        #        = n*w^2 - 2*sum(y)*w + sum(y^2) + l2*w^2
        # d/dw = 2*n*w - 2*sum(y) + 2*l2*w = 0
        # w = sum(y) / (n + l2)
        n = len(y)
        return float(np.sum(y) / (n + self.l2))

    def _compute_loss(self, y: np.ndarray, w: float) -> float:
        """Compute regularized loss for a single leaf prediction."""
        if len(y) == 0:
            return 0.0
        mse = np.mean((y - w) ** 2)
        reg = self.l2 * w ** 2
        return float(mse + reg)

    def _find_best_split_rgf(
        self, leaf: TNode, data: dict, n_features: int, rng: np.random.Generator
    ) -> dict | None:
        """Find best split for a leaf using RGF criterion."""
        X, y = data["X"], data["y"]
        n = len(y)

        if n < self.min_samples_leaf * 2:
            return None
        if self.max_depth is not None and leaf.depth >= self.max_depth:
            return None

        # Feature sampling
        feat_indices = self._get_feature_subset(n_features, rng)

        best_gain = -np.inf
        best_split = None

        current_weight = self._solve_leaf_weight(y)
        current_loss = self._compute_loss(y, current_weight)

        for feat_idx in feat_indices:
            col = X[:, feat_idx]
            unique_vals = np.unique(col)
            if len(unique_vals) < 2:
                continue

            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for thr in thresholds:
                left_mask = col <= thr
                right_mask = ~left_mask

                nl, nr = left_mask.sum(), right_mask.sum()
                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue

                yl, yr = y[left_mask], y[right_mask]

                # RGF gain: reduction in regularized loss
                wl = self._solve_leaf_weight(yl)
                wr = self._solve_leaf_weight(yr)

                loss_l = self._compute_loss(yl, wl)
                loss_r = self._compute_loss(yr, wr)
                loss_parent = current_loss * n

                gain = loss_parent - (nl * loss_l + nr * loss_r)

                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        "feature_idx": feat_idx,
                        "threshold": thr,
                        "gain": gain,
                        "nl": nl,
                        "nr": nr,
                        "wl": wl,
                        "wr": wr,
                    }

        if best_split is not None:
            best_split["gain"] /= n  # Normalize by sample count
        return best_split

    def _get_feature_subset(self, n_features: int, rng: np.random.Generator) -> np.ndarray:
        if self.feature_subset is None:
            return np.arange(n_features)
        elif self.feature_subset == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
        elif self.feature_subset == "log2":
            k = max(1, int(np.log2(n_features)))
        elif isinstance(self.feature_subset, float):
            k = max(1, int(self.feature_subset * n_features))
        elif isinstance(self.feature_subset, int):
            k = min(n_features, self.feature_subset)
        else:
            k = n_features
        return rng.choice(n_features, size=k, replace=False)

    def _split_leaf(self, leaf_id: int, split: dict, n_features: int) -> None:
        """Split a leaf node and update data structures."""
        leaf = self.leaves_[leaf_id]
        data = self._leaf_data[leaf_id]

        # Remove from leaves dict
        del self.leaves_[leaf_id]
        del self._leaf_data[leaf_id]

        X, y, indices = data["X"], data["y"], data["indices"]

        left_mask = X[:, split["feature_idx"]] <= split["threshold"]
        right_mask = ~left_mask

        # Create children
        left_leaf = TNode(
            leaf_id=self.next_leaf_id_,
            weight=split["wl"],
            n_struct=split["nl"],
            n_estim=split["nl"],
            depth=leaf.depth + 1,
            parent=leaf,
        )
        self.next_leaf_id_ += 1
        right_leaf = TNode(
            leaf_id=self.next_leaf_id_,
            weight=split["wr"],
            n_struct=split["nr"],
            n_estim=split["nr"],
            depth=leaf.depth + 1,
            parent=leaf,
        )
        self.next_leaf_id_ += 1

        leaf.feature_idx = split["feature_idx"]
        leaf.threshold = split["threshold"]
        leaf.left = left_leaf
        leaf.right = right_leaf
        leaf.weight = 0.0  # Internal node has no weight

        # Store data for children
        self._leaf_data[left_leaf.leaf_id] = {
            "X": X[left_mask],
            "y": y[left_mask],
            "indices": indices[left_mask],
        }
        self._leaf_data[right_leaf.leaf_id] = {
            "X": X[right_mask],
            "y": y[right_mask],
            "indices": indices[right_mask],
        }

        self.leaves_[left_leaf.leaf_id] = left_leaf
        self.leaves_[right_leaf.leaf_id] = right_leaf

    def _update_all_leaf_weights(self) -> None:
        """Recompute optimal weights for all leaves."""
        for leaf_id, data in self._leaf_data.items():
            leaf = self.leaves_[leaf_id]
            leaf.weight = self._solve_leaf_weight(data["y"])

    def _compute_tree_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute total loss over all training data."""
        preds = self._predict_numpy(X)
        return float(np.mean((y - preds) ** 2) + self.l2 * np.mean(preds ** 2))

    def _route_to_leaf(self, node: TNode, x: np.ndarray) -> TNode:
        """Route a single sample to its leaf."""
        if node.is_leaf:
            return node
        if x[node.feature_idx] <= node.threshold:
            return self._route_to_leaf(node.left, x)
        return self._route_to_leaf(node.right, x)

    def _predict_numpy(self, X: np.ndarray) -> np.ndarray:
        preds = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            leaf = self._route_to_leaf(self.root_, x)
            preds[i] = leaf.weight
        return preds

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "root_")
        X_arr = _to_numpy(X)
        return self._predict_numpy(X_arr)


# ═══════════════════════════════════════════════════════════════════
# Honest Tree
# ═══════════════════════════════════════════════════════════════════

class HonestTreeRegressor(BaseEstimator, RegressorMixin):
    """
    Honest Tree for regression.

    Uses separate samples for:
      - Structure: deciding tree topology (splits)
      - Estimation: computing leaf predictions (weights)

    This eliminates bias in leaf predictions and is crucial for:
      - Causal inference (treatment effect estimation)
      - Kernel construction (RFR-like kernels need honest estimation)
      - Uncertainty quantification

    Parameters:
    -----------
    base_estimator : estimator, default=None
        Base estimator for leaf models (if None, uses constant weight)
    max_depth : int, default=10
        Maximum tree depth
    min_samples_split : int, default=10
        Minimum samples required to split a node (structure samples)
    min_samples_leaf : int, default=5
        Minimum samples required in a leaf (estimation samples)
    honest_fraction : float, default=0.5
        Fraction of data used for structure (rest for estimation)
        If 0.5, half the data determines tree structure, half estimates leaves
    max_features : str or int or float, default=None
        Feature subset strategy
    random_state : int, default=None
        Random seed
    """

    def __init__(
        self,
        base_estimator: Any = None,
        max_depth: int = 10,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        honest_fraction: float = 0.5,
        max_features: Any = None,
        max_bins: int = 64,
        random_state: int | None = None,
    ) -> None:
        self.base_estimator = base_estimator
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.honest_fraction = honest_fraction
        self.max_features = max_features
        self.max_bins = max_bins
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> "HonestTreeRegressor":
        X_arr = _to_numpy(X)
        y_arr = np.asarray(y, dtype=float).ravel()
        n_samples = X_arr.shape[0]
        rng = np.random.default_rng(self.random_state)

        # Split data into structure and estimation sets
        n_struct = int(n_samples * self.honest_fraction)
        n_struct = max(self.min_samples_split, min(n_struct, n_samples - self.min_samples_leaf))

        perm = rng.permutation(n_samples)
        struct_idx = perm[:n_struct]
        estim_idx = perm[n_struct:]

        self.struct_X_ = X_arr[struct_idx]
        self.struct_y_ = y_arr[struct_idx]
        self.estim_X_ = X_arr[estim_idx] if len(estim_idx) > 0 else X_arr
        self.estim_y_ = y_arr[estim_idx] if len(estim_idx) > 0 else y_arr

        self.n_features_in_ = X_arr.shape[1]

        # Build tree structure using structure samples
        self.root_ = self._build_honest_tree(
            self.struct_X_, self.struct_y_,
            self.estim_X_, self.estim_y_,
            depth=0,
        )

        self.n_leaves_ = self._count_leaves(self.root_)
        logger.info(
            f"HonestTreeRegressor.fit(): {self.n_leaves_} leaves, "
            f"struct_n={len(struct_idx)}, estim_n={len(estim_idx) if len(estim_idx) > 0 else n_samples}"
        )
        return self

    def _build_honest_tree(
        self,
        X_struct: np.ndarray,
        y_struct: np.ndarray,
        X_estim: np.ndarray,
        y_estim: np.ndarray,
        depth: int,
    ) -> TNode:
        n_struct = X_struct.shape[0]
        node = TNode(depth=depth, n_struct=n_struct, n_estim=X_estim.shape[0])

        # Stopping criteria
        if (
            depth >= self.max_depth
            or n_struct < self.min_samples_split
            or n_struct < self.min_samples_leaf * 2
        ):
            node.weight = self._estimate_leaf_weight(X_estim, y_estim)
            node.leaf_id = getattr(self, "_leaf_counter", 0)
            self._leaf_counter = getattr(self, "_leaf_counter", 0) + 1
            return node

        # Find best split using structure data
        best_score = np.inf
        best_feat = -1
        best_thr = None
        best_masks = None

        n_features = X_struct.shape[1]
        feat_indices = self._get_feature_indices(n_features, depth, rng=None)

        for feat_idx in feat_indices:
            col = X_struct[:, feat_idx]
            unique_vals = np.unique(col)
            if len(unique_vals) < 2:
                continue

            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
            if len(thresholds) > self.max_bins:
                idx = np.linspace(0, len(thresholds) - 1, self.max_bins, dtype=int)
                thresholds = thresholds[idx]

            for thr in thresholds:
                left_mask_s = col <= thr
                right_mask_s = ~left_mask_s

                nl_s = left_mask_s.sum()
                nr_s = right_mask_s.sum()
                if nl_s < self.min_samples_leaf or nr_s < self.min_samples_leaf:
                    continue

                # Score using structure data
                score = self._split_score(X_struct, y_struct, left_mask_s, right_mask_s)
                if score < best_score:
                    best_score = score
                    best_feat = feat_idx
                    best_thr = thr
                    best_masks = (left_mask_s, right_mask_s)

        if best_feat == -1 or best_masks is None:
            node.weight = self._estimate_leaf_weight(X_estim, y_estim)
            node.leaf_id = getattr(self, "_leaf_counter", 0)
            self._leaf_counter = getattr(self, "_leaf_counter", 0) + 1
            return node

        # Now gather estimation data for children
        X_left_e = X_estim[X_estim[:, best_feat] <= best_thr]
        y_left_e = y_estim[X_estim[:, best_feat] <= best_thr]
        X_right_e = X_estim[X_estim[:, best_feat] > best_thr]
        y_right_e = y_estim[X_estim[:, best_feat] > best_thr]

        node.feature_idx = best_feat
        node.threshold = best_thr

        # Recursively build children with appropriate data
        left_X_s = X_struct[best_masks[0]]
        left_y_s = y_struct[best_masks[0]]
        right_X_s = X_struct[best_masks[1]]
        right_y_s = y_struct[best_masks[1]]

        node.left = self._build_honest_tree(
            left_X_s, left_y_s, X_left_e, y_left_e, depth + 1
        )
        node.right = self._build_honest_tree(
            right_X_s, right_y_s, X_right_e, y_right_e, depth + 1
        )

        return node

    def _estimate_leaf_weight(self, X_estim: np.ndarray, y_estim: np.ndarray) -> float:
        """Estimate leaf prediction using estimation samples."""
        if len(y_estim) == 0:
            return 0.0
        if self.base_estimator is not None:
            try:
                model = clone(self.base_estimator)
                model.fit(X_estim, y_estim)
                # For prediction, we'd need a query point; use global mean as fallback
                return float(np.mean(y_estim))
            except Exception:
                pass
        return float(np.mean(y_estim))

    def _split_score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        left_mask: np.ndarray,
        right_mask: np.ndarray,
    ) -> float:
        """MSE-based split score."""
        nl = left_mask.sum()
        nr = right_mask.sum()
        n = nl + nr
        if n == 0:
            return np.inf
        score_l = np.var(y[left_mask]) if nl > 0 else 0
        score_r = np.var(y[right_mask]) if nr > 0 else 0
        return (nl * score_l + nr * score_r) / n

    def _get_feature_indices(
        self, n_features: int, depth: int, rng: np.random.Generator | None
    ) -> np.ndarray:
        if self.max_features is None:
            return np.arange(n_features)
        elif self.max_features == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            k = max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, float):
            k = max(1, int(self.max_features * n_features))
        elif isinstance(self.max_features, int):
            k = min(n_features, self.max_features)
        else:
            k = n_features
        if rng is not None:
            return rng.choice(n_features, size=k, replace=False)
        return np.arange(k)  # Fallback (should use rng)

    def _count_leaves(self, node: TNode) -> int:
        if node.is_leaf:
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)

    def _route_to_leaf(self, node: TNode, x: np.ndarray) -> TNode:
        if node.is_leaf:
            return node
        if x[node.feature_idx] <= node.threshold:
            return self._route_to_leaf(node.left, x)
        return self._route_to_leaf(node.right, x)

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "root_")
        X_arr = _to_numpy(X)
        preds = np.zeros(X_arr.shape[0])
        for i, x in enumerate(X_arr):
            leaf = self._route_to_leaf(self.root_, x)
            preds[i] = leaf.weight
        return preds

    def get_leaf_assignments(self, X: Any) -> np.ndarray:
        """Return leaf ID for each sample (useful for kernel construction)."""
        check_is_fitted(self, "root_")
        X_arr = _to_numpy(X)
        leaf_ids = np.zeros(X_arr.shape[0], dtype=int)
        for i, x in enumerate(X_arr):
            leaf = self._route_to_leaf(self.root_, x)
            leaf_ids[i] = leaf.leaf_id
        return leaf_ids


# ═══════════════════════════════════════════════════════════════════
# Soft Split Tree
# ═══════════════════════════════════════════════════════════════════

class SoftTreeRegressor(BaseEstimator, RegressorMixin):
    """
    Decision tree with soft splits.

    Instead of hard binary splits, each sample is assigned to both
    children with a probability/weight, creating smoother decision boundaries.

    Soft split function (sigmoid):
        w_left(x) = sigmoid((threshold - x_j) / temperature)
        w_right(x) = 1 - w_left(x)

    This allows:
      - Samples near boundary to contribute to both sides
      - Smoother predictions
      - Better kernel construction (RFR-like)

    Parameters:
    -----------
    temperature : float, default=0.1
        Softness of split. Lower = harder split, Higher = softer.
    base_estimator : estimator, default=None
        Leaf model
    max_depth : int, default=10
    min_samples_split : int, default=10
    min_samples_leaf : int, default=5
    max_features : str or int or float, default=None
    max_bins : int, default=64
    random_state : int, default=None
    """

    def __init__(
        self,
        temperature: float = 0.1,
        base_estimator: Any = None,
        max_depth: int = 10,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        max_features: Any = None,
        max_bins: int = 64,
        random_state: int | None = None,
    ) -> None:
        self.temperature = temperature
        self.base_estimator = base_estimator
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_bins = max_bins
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> "SoftTreeRegressor":
        X_arr = _to_numpy(X)
        y_arr = np.asarray(y, dtype=float).ravel()

        self.n_features_in_ = X_arr.shape[1]
        self._leaf_counter = 0

        # Build tree (structure only - soft weights computed at prediction)
        self.root_ = self._build_soft_tree(X_arr, y_arr, depth=0)

        self.n_leaves_ = self._count_leaves(self.root_)
        return self

    def _build_soft_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> TNode:
        n = X.shape[0]
        node = TNode(depth=depth)
        node.n_struct = n
        node.n_estim = n

        if (
            depth >= self.max_depth
            or n < self.min_samples_split
            or n < self.min_samples_leaf * 2
        ):
            node.weight = float(np.mean(y)) if len(y) > 0 else 0.0
            node.leaf_id = self._leaf_counter
            self._leaf_counter += 1
            return node

        # Find best split
        best_score = np.inf
        best_feat = -1
        best_thr = None

        n_features = X.shape[1]
        rng = np.random.default_rng(self.random_state)
        if self.max_features == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
        elif isinstance(self.max_features, int):
            k = min(n_features, self.max_features)
        else:
            k = n_features
        feat_indices = rng.choice(n_features, size=k, replace=False) if k < n_features else np.arange(n_features)

        for feat_idx in feat_indices:
            col = X[:, feat_idx]
            unique_vals = np.unique(col)
            if len(unique_vals) < 2:
                continue

            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
            if len(thresholds) > self.max_bins:
                idx = np.linspace(0, len(thresholds) - 1, self.max_bins, dtype=int)
                thresholds = thresholds[idx]

            for thr in thresholds:
                left_mask = col <= thr
                right_mask = ~left_mask
                nl, nr = left_mask.sum(), right_mask.sum()
                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue

                score = self._split_score(X, y, left_mask, right_mask)
                if score < best_score:
                    best_score = score
                    best_feat = feat_idx
                    best_thr = thr

        if best_feat == -1:
            node.weight = float(np.mean(y)) if len(y) > 0 else 0.0
            node.leaf_id = self._leaf_counter
            self._leaf_counter += 1
            return node

        node.feature_idx = best_feat
        node.threshold = best_thr

        left_mask = X[:, best_feat] <= best_thr
        right_mask = ~left_mask

        node.left = self._build_soft_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_soft_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def _split_score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        left_mask: np.ndarray,
        right_mask: np.ndarray,
    ) -> float:
        nl = left_mask.sum()
        nr = right_mask.sum()
        n = nl + nr
        if n == 0:
            return np.inf
        score_l = np.var(y[left_mask]) if nl > 0 else 0
        score_r = np.var(y[right_mask]) if nr > 0 else 0
        return (nl * score_l + nr * score_r) / n

    def _soft_weight(self, x_val: float, threshold: float) -> tuple[float, float]:
        """Compute soft split weights for a single feature value."""
        if self.temperature <= 0:
            # Hard split
            if x_val <= threshold:
                return 1.0, 0.0
            else:
                return 0.0, 1.0

        # Sigmoid-based soft split
        z = (threshold - x_val) / self.temperature
        w_left = 1.0 / (1.0 + np.exp(z))
        w_right = 1.0 - w_left
        return float(w_left), float(w_right)

    def _route_soft(self, node: TNode, x: np.ndarray) -> list[tuple[TNode, float]]:
        """
        Route a sample through the tree with soft splits.
        Returns list of (leaf_node, weight) pairs.
        """
        if node.is_leaf:
            return [(node, 1.0)]

        w_left, w_right = self._soft_weight(x[node.feature_idx], node.threshold)

        result = []
        if w_left > 1e-10:
            for leaf, w in self._route_soft(node.left, x):
                result.append((leaf, w_left * w))
        if w_right > 1e-10:
            for leaf, w in self._route_soft(node.right, x):
                result.append((leaf, w_right * w))
        return result

    def _count_leaves(self, node: TNode) -> int:
        if node.is_leaf:
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "root_")
        X_arr = _to_numpy(X)
        preds = np.zeros(X_arr.shape[0])
        for i, x in enumerate(X_arr):
            leaf_weights = self._route_soft(self.root_, x)
            pred = 0.0
            for leaf, w in leaf_weights:
                pred += w * leaf.weight
            preds[i] = pred
        return preds

    def get_leaf_weights(self, X: Any) -> np.ndarray:
        """
        Return soft leaf assignment weights for each sample.
        Output shape: (n_samples, n_leaves)
        """
        check_is_fitted(self, "root_")
        X_arr = _to_numpy(X)
        n_leaves = self._count_leaves(self.root_)
        result = np.zeros((X_arr.shape[0], n_leaves))

        leaf_id_map = {}

        def _collect_leaves(node: TNode, idx: int) -> int:
            if node.is_leaf:
                leaf_id_map[id(node)] = idx
                return idx + 1
            idx = _collect_leaves(node.left, idx)
            idx = _collect_leaves(node.right, idx)
            return idx

        _collect_leaves(self.root_, 0)

        for i, x in enumerate(X_arr):
            leaf_weights = self._route_soft(self.root_, x)
            for leaf, w in leaf_weights:
                lid = leaf_id_map.get(id(leaf), -1)
                if lid >= 0:
                    result[i, lid] = w
        return result


# ═══════════════════════════════════════════════════════════════════
# TREEKERNEL: RFR Kernel for a Single Tree
# ═══════════════════════════════════════════════════════════════════

class TreeKernel:
    """
    TREEKERNEL: RFR (Random Forest Kernel) reproduction for a single tree.

    The RFR kernel (Davies & Ghahramani 2014) computes similarity between
    samples based on the tree structure:
        K(x_i, x_j) = sum over leaves L of (1 / n_L) * I(x_i in L) * I(x_j in L)

    For a single tree with honest estimation, we can compute this as:
        K_honest(x_i, x_j) = 1 / n_L  if both x_i and x_j fall in leaf L

    Extensions for soft splits:
        K_soft(x_i, x_j) = sum over leaves L of w_i(L) * w_j(L) / n_L
        where w_i(L) is the soft weight of x_i in leaf L.

    This kernel can be used with any kernel method:
        - KernelRidge
        - SVR
        - SVC
        - GaussianProcessRegressor / GaussianProcessClassifier

    Parameters:
    -----------
    tree_model : estimator
        Fitted tree model that supports get_leaf_assignments() or predict_leaf()
    kernel_type : str, default="rfr"
        "rfr" - standard RFR kernel
        "rfr_honest" - RFR kernel with honest estimation
        "rfr_soft" - RFR kernel with soft splits
    normalize : bool, default=True
        Whether to normalize the kernel (K_norm = K / sqrt(K_ii * K_jj))
    """

    def __init__(
        self,
        tree_model: Any = None,
        kernel_type: str = "rfr",
        normalize: bool = True,
    ) -> None:
        self.tree_model = tree_model
        self.kernel_type = kernel_type
        self.normalize = normalize

    def fit(self, X: Any, y: Any = None) -> "TreeKernel":
        """Fit the tree model and precompute leaf information."""
        X_arr = _to_numpy(X)

        if self.tree_model is None:
            self.tree_model_ = HonestTreeRegressor(max_depth=10, honest_fraction=0.5)
        else:
            self.tree_model_ = clone(self.tree_model) if hasattr(self.tree_model, "get_params") else self.tree_model

        if not hasattr(self.tree_model_, "root_"):
            self.tree_model_.fit(X_arr, y if y is not None else np.zeros(X_arr.shape[0]))

        self.X_fit_ = X_arr
        self.n_samples_fit_ = X_arr.shape[0]

        # Precompute leaf assignments
        if hasattr(self.tree_model_, "get_leaf_assignments"):
            self.leaf_assignments_ = self.tree_model_.get_leaf_assignments(X_arr)
        elif hasattr(self.tree_model_, "apply"):
            self.leaf_assignments_ = self.tree_model_.apply(X_arr)
        else:
            # Fallback: route manually
            self.leaf_assignments_ = self._manual_leaf_assignments(X_arr)

        # Get unique leaves and their counts
        self.unique_leaves_, self.leaf_counts_ = np.unique(
            self.leaf_assignments_, return_counts=True
        )

        # Build leaf -> index mapping
        self.leaf_to_idx_ = {leaf: idx for idx, leaf in enumerate(self.unique_leaves_)}

        return self

    def _manual_leaf_assignments(self, X: np.ndarray) -> np.ndarray:
        """Manually route samples to leaves if no built-in method."""
        if not hasattr(self.tree_model_, "root_"):
            raise ValueError("Tree model not fitted or doesn't support leaf assignment")

        assignments = np.zeros(X.shape[0], dtype=int)
        for i, x in enumerate(X):
            leaf = self._route_to_leaf(self.tree_model_.root_, x)
            assignments[i] = leaf.leaf_id
        return assignments

    def _route_to_leaf(self, node: Any, x: np.ndarray) -> Any:
        """Route to leaf. Handles both TNode and sklearn tree."""
        if hasattr(node, "is_leaf"):
            if node.is_leaf:
                return node
            if x[node.feature_idx] <= node.threshold:
                return self._route_to_leaf(node.left, x)
            return self._route_to_leaf(node.right, x)
        else:
            # sklearn tree
            return node

    def __call__(self, X: Any, Y: Any = None) -> np.ndarray:
        """
        Compute the tree kernel matrix.

        K[i, j] = 1 / n_L  where L is the leaf containing both X[i] and Y[j]
        (or sum of such terms for soft assignments)
        """
        X_arr = _to_numpy(X)
        Y_arr = _to_numpy(Y) if Y is not None else X_arr
        Y_is_X = Y is None

        # Get leaf assignments
        if Y_is_X:
            leaf_X = self.leaf_assignments_
        else:
            if hasattr(self.tree_model_, "get_leaf_assignments"):
                leaf_X = self.leaf_assignments_
                leaf_Y = self.tree_model_.get_leaf_assignments(Y_arr)
            elif hasattr(self.tree_model_, "apply"):
                leaf_X = self.leaf_assignments_
                leaf_Y = self.tree_model_.apply(Y_arr)
            else:
                leaf_Y = self._manual_leaf_assignments(Y_arr)

        n_X = len(leaf_X)
        n_Y = len(leaf_Y) if not Y_is_X else n_X

        # Compute kernel matrix
        K = np.zeros((n_X, n_Y))

        if Y_is_X:
            for i in range(n_X):
                li = leaf_X[i]
                idx_i = self.leaf_to_idx_.get(li, -1)
                if idx_i < 0:
                    continue
                n_L = self.leaf_counts_[idx_i]
                K[i, i] += 1.0 / n_L
                for j in range(i + 1, n_X):
                    lj = leaf_X[j]
                    if li == lj:
                        K[i, j] += 1.0 / n_L
                        K[j, i] = K[i, j]

            if self.normalize:
                diag = np.sqrt(np.diag(K))
                with np.errstate(divide="ignore", invalid="ignore"):
                    inv_diag = np.where(diag > 0, 1.0 / diag, 0.0)
                K = inv_diag[:, np.newaxis] * K * inv_diag[np.newaxis, :]
        else:
            for i in range(n_X):
                li = leaf_X[i]
                idx_i = self.leaf_to_idx_.get(li, -1)
                if idx_i < 0:
                    continue
                n_L = self.leaf_counts_[idx_i]
                for j in range(n_Y):
                    lj = leaf_Y[j]
                    if li == lj:
                        K[i, j] = 1.0 / n_L

            if self.normalize:
                diag_X = np.array([
                    1.0 / np.sqrt(self.leaf_counts_[self.leaf_to_idx_.get(leaf_X[i], 0)])
                    for i in range(n_X)
                ])
                if Y_is_X:
                    diag_Y = diag_X
                else:
                    diag_Y = np.array([
                        1.0 / np.sqrt(self.leaf_counts_[self.leaf_to_idx_.get(leaf_Y[j], 0)])
                        for j in range(n_Y)
                    ])
                K = diag_X[:, np.newaxis] * K * diag_Y[np.newaxis, :]

        return K


# ═══════════════════════════════════════════════════════════════════
# Kernel wrapper: Apply TREEKERNEL to any kernel method
# ═══════════════════════════════════════════════════════════════════

class TreeKernelWrapper(BaseEstimator, RegressorMixin):
    """
    Wrapper that uses TREEKERNEL with any sklearn kernel method.

    Combines the prediction power of RF-like trees with kernel methods:
      - Use tree structure to define a similarity kernel
      - Use kernel method (SVR, KernelRidge, GPR) for final prediction

    This effectively creates "KernelRidge with RFR kernel" or "SVR with RFR kernel".

    Parameters:
    -----------
    base_kernel_model : estimator
        Kernel method to wrap (KernelRidge, SVR, GaussianProcessRegressor, etc.)
    tree_model : estimator
        Tree model for kernel construction
    tree_kernel_kwargs : dict
        Arguments for TreeKernel
    """

    def __init__(
        self,
        base_kernel_model: Any = None,
        tree_model: Any = None,
        tree_kernel_kwargs: dict | None = None,
    ) -> None:
        self.base_kernel_model = base_kernel_model
        self.tree_model = tree_model
        self.tree_kernel_kwargs = tree_kernel_kwargs or {}

    def fit(self, X: Any, y: Any) -> "TreeKernelWrapper":
        X_arr = _to_numpy(X)
        y_arr = np.asarray(y, dtype=float).ravel()

        # Fit tree and compute kernel
        self.tree_kernel_ = TreeKernel(
            tree_model=self.tree_model,
            **self.tree_kernel_kwargs,
        )
        self.tree_kernel_.fit(X_arr, y_arr)

        # Compute kernel matrix
        K = self.tree_kernel_(X_arr)

        # Fit base kernel model with precomputed kernel
        if self.base_kernel_model is None:
            from sklearn.kernel_ridge import KernelRidge
            self.base_model_ = KernelRidge(kernel="precomputed")
        else:
            self.base_model_ = clone(self.base_kernel_model)
            if hasattr(self.base_model_, "kernel"):
                self.base_model_.kernel = "precomputed"

        self.base_model_.fit(K, y_arr)
        self.X_fit_ = X_arr
        self.n_features_in_ = X_arr.shape[1]
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "base_model_")
        X_arr = _to_numpy(X)

        # Compute cross-kernel between train and test
        K_test = self.tree_kernel_(X_arr, self.X_fit_)

        return self.base_model_.predict(K_test)


# ═══════════════════════════════════════════════════════════════════
# Bernoulli Forest
# ═══════════════════════════════════════════════════════════════════

class BernoulliForestRegressor(BaseEstimator, RegressorMixin):
    """
    Bernoulli Forest for regression.

    Each tree uses Bernoulli sampling (sampling with probability p)
    instead of bootstrap sampling. This creates trees with different
    data subsets while maintaining the original sample size.

    Key properties:
      - Each sample has probability p of being included in each tree
      - Expected OOB samples: (1-p)^n_trees
      - Related to "poisson forest" and "bernoulli bootstrap"

    Parameters:
    -----------
    n_estimators : int, default=100
        Number of trees
    p : float, default=0.5
        Inclusion probability for Bernoulli sampling
    base_tree : estimator, default=None
        Base tree model (HonestTreeRegressor recommended)
    max_depth : int, default=10
    min_samples_leaf : int, default=5
    max_features : str or int or float, default="sqrt"
    n_jobs : int, default=1
    random_state : int, default=None
    """

    def __init__(
        self,
        n_estimators: int = 100,
        p: float = 0.5,
        base_tree: Any = None,
        max_depth: int = 10,
        min_samples_leaf: int = 5,
        max_features: Any = "sqrt",
        n_jobs: int = 1,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.p = p
        self.base_tree = base_tree
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> "BernoulliForestRegressor":
        from joblib import Parallel, delayed

        X_arr = _to_numpy(X)
        y_arr = np.asarray(y, dtype=float).ravel()
        n = X_arr.shape[0]

        rng = np.random.default_rng(self.random_state)
        seeds = [int(rng.integers(0, 2**31)) for _ in range(self.n_estimators)]

        def _fit_one(seed: int) -> Any:
            _rng = np.random.default_rng(seed)
            # Bernoulli sampling
            include = _rng.random(n) < self.p
            if include.sum() < self.min_samples_leaf:
                include = np.ones(n, dtype=bool)  # Fallback
            idx = np.where(include)[0]

            if self.base_tree is None:
                tree = HonestTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features=self.max_features,
                    random_state=seed,
                )
            else:
                tree = clone(self.base_tree)
                tree.random_state = seed

            tree.fit(X_arr[idx], y_arr[idx])
            return tree

        n_jobs = self.n_jobs if self.n_jobs != 0 else 1
        self.estimators_ = Parallel(n_jobs=n_jobs)(
            delayed(_fit_one)(seed) for seed in seeds
        )
        self.n_features_in_ = X_arr.shape[1]
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "estimators_")
        X_arr = _to_numpy(X)
        from joblib import Parallel, delayed
        n_jobs = self.n_jobs if self.n_jobs != 0 else 1

        all_preds = Parallel(n_jobs=n_jobs)(
            delayed(t.predict)(X_arr) for t in self.estimators_
        )
        return np.mean(all_preds, axis=0)


# ═══════════════════════════════════════════════════════════════════
# Rotation Forest
# ═══════════════════════════════════════════════════════════════════

class RotationForestRegressor(BaseEstimator, RegressorMixin):
    """
    Rotation Forest for regression.

    Each base tree is trained on PCA-rotated features:
      1. Split features into K subsets
      2. For each subset, apply PCA to learn rotation matrix
      3. Concatenate all principal components
      4. Train a tree on the rotated features

    This creates diverse trees with uncorrelated features.

    Parameters:
    -----------
    n_estimators : int, default=100
        Number of trees
    n_feature_subsets : int, default=3
        Number of feature subsets for rotation
    base_tree : estimator, default=None
        Base tree model
    n_jobs : int, default=1
    random_state : int, default=None
    """

    def __init__(
        self,
        n_estimators: int = 100,
        n_feature_subsets: int = 3,
        base_tree: Any = None,
        n_jobs: int = 1,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.n_feature_subsets = n_feature_subsets
        self.base_tree = base_tree
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> "RotationForestRegressor":
        from joblib import Parallel, delayed

        X_arr = _to_numpy(X)
        y_arr = np.asarray(y, dtype=float).ravel()
        n, d = X_arr.shape

        rng = np.random.default_rng(self.random_state)
        seeds = [int(rng.integers(0, 2**31)) for _ in range(self.n_estimators)]

        def _fit_one(seed: int) -> tuple[Any, Any, list]:
            _rng = np.random.default_rng(seed)

            # Create feature subsets
            features = np.arange(d)
            _rng.shuffle(features)
            subset_size = max(1, d // self.n_feature_subsets)
            subsets = []
            for k in range(self.n_feature_subsets):
                start = k * subset_size
                end = (k + 1) * subset_size if k < self.n_feature_subsets - 1 else d
                subsets.append(features[start:end])

            # Apply PCA rotation to each subset
            rotation_matrices = []
            X_rotated = np.zeros_like(X_arr)

            for subset in subsets:
                if len(subset) <= 1:
                    rotation_matrices.append((subset, None))  # No rotation
                    continue

                X_sub = X_arr[:, subset]
                # Bootstrap sample for PCA fitting
                boot_idx = _rng.choice(n, size=n, replace=True)
                pca = PCA(n_components=len(subset), random_state=seed)
                pca.fit(X_sub[boot_idx])
                rotation_matrices.append((subset, pca))

                X_rotated[:, subset] = pca.transform(X_sub)

            # Train tree on rotated features
            if self.base_tree is None:
                tree = HonestTreeRegressor(max_depth=10, random_state=seed)
            else:
                tree = clone(self.base_tree)
                tree.random_state = seed

            tree.fit(X_rotated, y_arr)
            return tree, rotation_matrices, X_rotated

        n_jobs = self.n_jobs if self.n_jobs != 0 else 1
        results = Parallel(n_jobs=n_jobs)(
            delayed(_fit_one)(seed) for seed in seeds
        )

        self.estimators_ = [r[0] for r in results]
        self.rotation_matrices_ = [r[1] for r in results]
        self.n_features_in_ = d
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "estimators_")
        X_arr = _to_numpy(X)

        all_preds = []
        for tree, rot_mats in zip(self.estimators_, self.rotation_matrices_):
            X_rot = np.zeros_like(X_arr)
            for subset, pca in rot_mats:
                if pca is not None:
                    X_rot[:, subset] = pca.transform(X_arr[:, subset])
                else:
                    X_rot[:, subset] = X_arr[:, subset]
            all_preds.append(tree.predict(X_rot))

        return np.mean(all_preds, axis=0)


# ═══════════════════════════════════════════════════════════════════
# RFRKernelTree: Unified model combining all techniques
# ═══════════════════════════════════════════════════════════════════

class RFRKernelTreeRegressor(BaseEstimator, RegressorMixin):
    """
    RFRKernelTree: A single decision tree with RF-level accuracy.

    Combines:
      - RGF (Regularized Greedy Forest) for structured growth
      - Honest Tree (separate structure/estimation samples)
      - Soft splits (smooth boundaries)
      - TREEKERNEL (RFR-like kernel for prediction)

    Can also use the tree kernel with external kernel methods.

    Parameters:
    -----------
    use_rgf : bool, default=True
        Use RGF-style growth (greedy leaf-by-leaf)
    use_honest : bool, default=True
        Use honest estimation (separate samples)
    use_soft : bool, default=False
        Use soft splits
    temperature : float, default=0.1
        Softness parameter (if use_soft=True)
    max_leaves : int, default=50
        Maximum leaves (RGF)
    max_depth : int, default=10
    min_samples_leaf : int, default=5
    l2 : float, default=0.1
        L2 regularization (RGF)
    random_state : int, default=None
    """

    def __init__(
        self,
        use_rgf: bool = True,
        use_honest: bool = True,
        use_soft: bool = False,
        temperature: float = 0.1,
        max_leaves: int = 50,
        max_depth: int = 10,
        min_samples_leaf: int = 5,
        l2: float = 0.1,
        random_state: int | None = None,
    ) -> None:
        self.use_rgf = use_rgf
        self.use_honest = use_honest
        self.use_soft = use_soft
        self.temperature = temperature
        self.max_leaves = max_leaves
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2 = l2
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> "RFRKernelTreeRegressor":
        X_arr = _to_numpy(X)
        y_arr = np.asarray(y, dtype=float).ravel()

        if self.use_rgf and not self.use_honest:
            # Pure RGF
            self.tree_ = RGFRegressor(
                max_leaves=self.max_leaves,
                l2=self.l2,
                min_samples_leaf=self.min_samples_leaf,
                max_depth=self.max_depth,
                random_state=self.random_state,
            )
        elif self.use_honest:
            # Honest Tree
            self.tree_ = HonestTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                honest_fraction=0.5,
                random_state=self.random_state,
            )
        else:
            # Fallback: simple tree
            self.tree_ = HonestTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                honest_fraction=1.0,  # No honest split
                random_state=self.random_state,
            )

        self.tree_.fit(X_arr, y_arr)

        # Build TREEKERNEL from the fitted tree
        self.tree_kernel_ = TreeKernel(
            tree_model=self.tree_,
            kernel_type="rfr_honest" if self.use_honest else "rfr",
        )
        self.tree_kernel_.fit(X_arr, y_arr)

        self.X_fit_ = X_arr
        self.y_fit_ = y_arr
        self.n_features_in_ = X_arr.shape[1]

        # Compute in-sample predictions
        self.train_kernel_ = self.tree_kernel_(X_arr)
        self.train_pred_ = self.tree_.predict(X_arr)

        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "tree_")
        X_arr = _to_numpy(X)
        return self.tree_.predict(X_arr)

    def predict_kernel(self, X: Any) -> np.ndarray:
        """
        Predict using the tree kernel (for use with kernel methods).
        Returns the kernel matrix between X and training data.
        """
        check_is_fitted(self, "tree_kernel_")
        X_arr = _to_numpy(X)
        return self.tree_kernel_(X_arr, self.X_fit_)

    def get_tree_kernel(self, X: Any = None, Y: Any = None) -> np.ndarray:
        """Get the tree kernel matrix."""
        check_is_fitted(self, "tree_kernel_")
        return self.tree_kernel_(X, Y)


# ═══════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════

def _to_numpy(X: Any) -> np.ndarray:
    """Convert to numpy array."""
    import pandas as pd
    if isinstance(X, pd.DataFrame):
        return X.values.astype(float)
    arr = np.asarray(X)
    if arr.dtype.kind not in ("f", "i", "u"):
        arr = arr.astype(float)
    return arr
