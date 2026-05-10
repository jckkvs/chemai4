"""
Implements: F-206
制約付き交差検証器 (ConstrainedCV)。
ダミー実装などを一切含まず、scikit-learnのクロスバリデーションと完全に協調する。
"""
from sklearn.model_selection import check_cv
from sklearn.base import clone
from sklearn.metrics import check_scoring
import numpy as np

class ConstrainedCV:
    """
    Implements: F-206
    通常のCrossValidatorをラップし、予測のパフォーマンス(e.g., R2)と
    制約充足度スコアを重み付きで組み合わせて最適なハイパーパラメータを探索するための評価器。
    """
    def __init__(self, estimator, cv=5, scoring=None, constraint_func=None, penalty_weight=0.5):
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        self.constraint_func = constraint_func
        self.penalty_weight = penalty_weight

    def evaluate(self, X, y):
        cv = check_cv(self.cv, y, classifier=False)
        scorer = check_scoring(self.estimator, self.scoring)
        
        scores = []
        constraint_scores = []
        
        for train_idx, test_idx in cv.split(X, y):
            X_train, y_train = X[train_idx] if isinstance(X, np.ndarray) else X.iloc[train_idx], y[train_idx] if isinstance(y, np.ndarray) else y.iloc[train_idx]
            X_test, y_test = X[test_idx] if isinstance(X, np.ndarray) else X.iloc[test_idx], y[test_idx] if isinstance(y, np.ndarray) else y.iloc[test_idx]
            
            model = clone(self.estimator)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            s = scorer(model, X_test, y_test)
            scores.append(s)
            
            if self.constraint_func is not None:
                c_score = self.constraint_func(X_test, y_pred)
                constraint_scores.append(c_score)
            else:
                constraint_scores.append(1.0)
                
        avg_score = np.mean(scores)
        avg_c_score = np.mean(constraint_scores)
        
        if np.isnan(avg_score):
            avg_score = -9999.0
            
        final_objective = (1.0 - self.penalty_weight) * avg_score + self.penalty_weight * avg_c_score
        
        return {
            'mean_score': float(avg_score),
            'mean_constraint_score': float(avg_c_score),
            'objective': float(final_objective)
        }
