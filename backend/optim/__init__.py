"""backend.optim — ベイズ最適化・逆解析による実験計画モジュール."""
from backend.optim.search_space import SearchSpace, Variable
from backend.optim.constraints import (
    RangeConstraint, SumConstraint, InequalityConstraint,
    AtLeastNConstraint, AtLeastOneConstraint, CustomConstraint, apply_constraints,
)
from backend.optim.bayesian_optimizer import BayesianOptimizer
from backend.optim.inverse_optimizer import InverseConfig, InverseResult, run_inverse_optimization
from backend.optim.composition_sampler import DirichletSampler

__all__ = [
    "SearchSpace", "Variable", "BayesianOptimizer",
    "RangeConstraint", "SumConstraint", "InequalityConstraint",
    "AtLeastNConstraint", "AtLeastOneConstraint", "CustomConstraint", "apply_constraints",
    "InverseConfig", "InverseResult", "run_inverse_optimization",
    "DirichletSampler",
]
