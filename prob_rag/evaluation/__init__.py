"""
Evaluation module for Prob-RAG.

Contains metrics computation and evaluation utilities.
"""

from .metrics import (
    EvaluationResult,
    AggregateMetrics,
    Evaluator,
    LLMEvaluator,
    normalize_answer,
    compute_exact_match,
    compute_f1,
    compute_f1_multi,
    is_abstention_response,
    compute_baseline_comparison,
    save_evaluation_results,
)

__all__ = [
    "EvaluationResult",
    "AggregateMetrics",
    "Evaluator",
    "LLMEvaluator",
    "normalize_answer",
    "compute_exact_match",
    "compute_f1",
    "compute_f1_multi",
    "is_abstention_response",
    "compute_baseline_comparison",
    "save_evaluation_results",
]
