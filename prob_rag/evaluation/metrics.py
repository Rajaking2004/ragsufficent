"""
Evaluation Framework for Prob-RAG

Implements comprehensive evaluation metrics:
- Accuracy (exact match, F1, semantic)
- Coverage (fraction of questions answered)
- Selective Accuracy (accuracy on answered questions)
- Hallucination Detection
- Calibration Metrics
- Routing Quality Metrics
"""

import logging
import re
import string
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from collections import Counter
import json

import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        roc_auc_score,
        brier_score_loss,
        calibration_curve
    )
except ImportError:
    accuracy_score = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from ..config import EvaluationConfig, RouterState
from ..modules.generator import GenerationResult
from ..data.datasets import RAGSample


logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    # Sample info
    sample_id: str
    question: str
    
    # Answers
    predicted_answer: str
    ground_truth: str
    ground_truth_aliases: List[str] = field(default_factory=list)
    
    # Scores
    exact_match: bool = False
    f1_score: float = 0.0
    semantic_match: Optional[bool] = None  # From LLM eval
    llm_eval_decision: Optional[str] = None  # "perfect", "acceptable", "incorrect", "missing"
    
    # Routing info
    routing_state: Optional[RouterState] = None
    sufficiency_score: Optional[float] = None
    
    # Flags
    is_abstention: bool = False
    is_hedged: bool = False
    is_correct: bool = False  # Overall correctness
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "question": self.question,
            "predicted_answer": self.predicted_answer,
            "ground_truth": self.ground_truth,
            "exact_match": self.exact_match,
            "f1_score": self.f1_score,
            "semantic_match": self.semantic_match,
            "llm_eval_decision": self.llm_eval_decision,
            "routing_state": self.routing_state.value if self.routing_state else None,
            "sufficiency_score": self.sufficiency_score,
            "is_abstention": self.is_abstention,
            "is_hedged": self.is_hedged,
            "is_correct": self.is_correct
        }


@dataclass
class AggregateMetrics:
    """Aggregated evaluation metrics."""
    # Basic metrics
    num_samples: int = 0
    accuracy: float = 0.0
    exact_match_rate: float = 0.0
    avg_f1: float = 0.0
    
    # Coverage metrics
    coverage: float = 0.0  # Fraction not abstained
    selective_accuracy: float = 0.0  # Accuracy on non-abstained
    
    # Routing distribution
    abstention_rate: float = 0.0
    hedging_rate: float = 0.0
    standard_rate: float = 0.0
    
    # Per-state accuracy
    accuracy_red: Optional[float] = None  # Should be N/A (abstained)
    accuracy_yellow: Optional[float] = None
    accuracy_green: Optional[float] = None
    
    # Hallucination metrics
    hallucination_rate: float = 0.0  # Answered but wrong
    
    # Calibration
    expected_calibration_error: Optional[float] = None
    brier_score: Optional[float] = None
    
    # Selective accuracy at different coverage levels
    selective_accuracy_curve: Dict[float, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_samples": self.num_samples,
            "accuracy": self.accuracy,
            "exact_match_rate": self.exact_match_rate,
            "avg_f1": self.avg_f1,
            "coverage": self.coverage,
            "selective_accuracy": self.selective_accuracy,
            "abstention_rate": self.abstention_rate,
            "hedging_rate": self.hedging_rate,
            "standard_rate": self.standard_rate,
            "accuracy_yellow": self.accuracy_yellow,
            "accuracy_green": self.accuracy_green,
            "hallucination_rate": self.hallucination_rate,
            "expected_calibration_error": self.expected_calibration_error,
            "brier_score": self.brier_score,
            "selective_accuracy_curve": self.selective_accuracy_curve
        }
    
    def __repr__(self) -> str:
        return (
            f"AggregateMetrics(\n"
            f"  accuracy={self.accuracy:.3f}, coverage={self.coverage:.3f},\n"
            f"  selective_accuracy={self.selective_accuracy:.3f},\n"
            f"  abstention={self.abstention_rate:.3f}, hedging={self.hedging_rate:.3f}, standard={self.standard_rate:.3f}\n"
            f")"
        )


def normalize_answer(text: str) -> str:
    """
    Normalize answer for comparison.
    
    - Lowercase
    - Remove articles (a, an, the)
    - Remove punctuation
    - Normalize whitespace
    """
    if not text:
        return ""
    
    text = text.lower()
    
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def compute_exact_match(prediction: str, ground_truth: str, aliases: List[str] = None) -> bool:
    """
    Check if prediction exactly matches ground truth (after normalization).
    
    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer
        aliases: Alternative correct answers
        
    Returns:
        True if exact match
    """
    pred_norm = normalize_answer(prediction)
    
    # Check main answer
    if pred_norm == normalize_answer(ground_truth):
        return True
    
    # Check aliases
    if aliases:
        for alias in aliases:
            if pred_norm == normalize_answer(alias):
                return True
    
    return False


def compute_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute token-level F1 score between prediction and ground truth.
    
    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer
        
    Returns:
        F1 score between 0 and 1
    """
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    
    if not pred_tokens or not gt_tokens:
        return 0.0
    
    # Count common tokens
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_f1_multi(prediction: str, ground_truths: List[str]) -> float:
    """
    Compute max F1 score across multiple ground truths.
    """
    if not ground_truths:
        return 0.0
    return max(compute_f1(prediction, gt) for gt in ground_truths)


class LLMEvaluator:
    """
    Use an LLM to evaluate answer correctness.
    More robust than exact match for semantic equivalence.
    """
    
    EVAL_PROMPT = """===Task===
Evaluate whether the predicted answer is correct compared to the ground truth.

===Instructions===
1. Compare the "Predicted Answer" with the "Ground Truth Answers"
2. Consider semantic equivalence, not just exact wording
3. Categorize as one of:
   - "perfect": Completely correct, matches ground truth
   - "acceptable": Partially correct or contains main idea
   - "incorrect": Wrong or contradicts ground truth
   - "missing": Answer is "I don't know", refuses to answer, or indicates inability

===Input===
Question: {question}
Predicted Answer: {prediction}
Ground Truth Answers: {ground_truth}

===Output===
Provide your evaluation:
Decision: (One of "perfect", "acceptable", "incorrect", or "missing")"""

    def __init__(self, config: EvaluationConfig, api_key: Optional[str] = None):
        self.config = config
        self.api_key = api_key
        self.client = None
        
        if config.use_llm_eval:
            self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize OpenAI client."""
        if OpenAI is None:
            logger.warning("openai package not available, LLM eval disabled")
            return
        
        if not self.api_key:
            logger.warning("No API key provided, LLM eval disabled")
            return
        
        self.client = OpenAI(api_key=self.api_key)
    
    def evaluate(
        self,
        question: str,
        prediction: str,
        ground_truth: str,
        aliases: List[str] = None
    ) -> Tuple[str, bool]:
        """
        Evaluate answer using LLM.
        
        Args:
            question: Original question
            prediction: Predicted answer
            ground_truth: Ground truth answer
            aliases: Alternative correct answers
            
        Returns:
            Tuple of (decision, is_correct)
        """
        if self.client is None:
            # Fall back to heuristic
            is_correct = compute_exact_match(prediction, ground_truth, aliases)
            return "perfect" if is_correct else "incorrect", is_correct
        
        # Format ground truths
        all_answers = [ground_truth] + (aliases or [])
        gt_str = " | ".join(all_answers)
        
        prompt = self.EVAL_PROMPT.format(
            question=question,
            prediction=prediction,
            ground_truth=gt_str
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.eval_model,
                messages=[
                    {"role": "system", "content": "You are an evaluation assistant. Be precise and fair."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=50
            )
            
            result = response.choices[0].message.content.strip().lower()
            
            # Parse decision
            if "perfect" in result:
                return "perfect", True
            elif "acceptable" in result:
                return "acceptable", True
            elif "missing" in result:
                return "missing", False
            else:
                return "incorrect", False
                
        except Exception as e:
            logger.error(f"LLM eval error: {e}")
            # Fall back to F1 heuristic
            f1 = compute_f1(prediction, ground_truth)
            if f1 > 0.8:
                return "perfect", True
            elif f1 > 0.4:
                return "acceptable", True
            else:
                return "incorrect", False


def is_abstention_response(answer: str) -> bool:
    """
    Check if answer is an abstention (refuses to answer).
    """
    abstention_patterns = [
        r"i (cannot|can't|don't|do not) (answer|provide|determine)",
        r"(cannot|unable to) (be determined|answer|find)",
        r"(not enough|insufficient) (information|context|data)",
        r"the (context|documents?|text) (do|does) not (contain|provide|mention)",
        r"i don'?t know",
        r"unknown",
        r"no answer",
        r"cannot be determined",
    ]
    
    answer_lower = answer.lower()
    
    for pattern in abstention_patterns:
        if re.search(pattern, answer_lower):
            return True
    
    return False


class Evaluator:
    """
    Main evaluation class for Prob-RAG.
    
    Computes comprehensive metrics including:
    - Accuracy (exact match, F1, semantic)
    - Coverage and selective accuracy
    - Hallucination rate
    - Calibration metrics
    """
    
    def __init__(self, config: EvaluationConfig, api_key: Optional[str] = None):
        self.config = config
        self.llm_evaluator = LLMEvaluator(config, api_key) if config.use_llm_eval else None
    
    def evaluate_single(
        self,
        sample: RAGSample,
        generation: GenerationResult
    ) -> EvaluationResult:
        """
        Evaluate a single prediction.
        
        Args:
            sample: Original RAG sample with ground truth
            generation: Generated result
            
        Returns:
            EvaluationResult with all metrics
        """
        predicted = generation.answer
        ground_truth = sample.answer
        aliases = sample.answer_aliases
        
        # Check if abstention
        is_abstention = generation.is_abstention or is_abstention_response(predicted)
        
        # Compute metrics
        exact_match = compute_exact_match(predicted, ground_truth, aliases)
        f1 = compute_f1_multi(predicted, [ground_truth] + aliases)
        
        # LLM evaluation
        llm_decision = None
        semantic_match = None
        if self.llm_evaluator and not is_abstention:
            llm_decision, semantic_match = self.llm_evaluator.evaluate(
                sample.question, predicted, ground_truth, aliases
            )
        
        # Determine overall correctness
        # For abstentions, we don't count as correct or incorrect
        if is_abstention:
            is_correct = False  # Abstained, so not correct
        elif semantic_match is not None:
            is_correct = semantic_match
        else:
            is_correct = exact_match or f1 > 0.5
        
        return EvaluationResult(
            sample_id=sample.id,
            question=sample.question,
            predicted_answer=predicted,
            ground_truth=ground_truth,
            ground_truth_aliases=aliases,
            exact_match=exact_match,
            f1_score=f1,
            semantic_match=semantic_match,
            llm_eval_decision=llm_decision,
            routing_state=generation.state,
            sufficiency_score=generation.score,
            is_abstention=is_abstention,
            is_hedged=generation.is_hedged,
            is_correct=is_correct
        )
    
    def evaluate_batch(
        self,
        samples: List[RAGSample],
        generations: List[GenerationResult],
        show_progress: bool = True
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple predictions.
        """
        if len(samples) != len(generations):
            raise ValueError("Samples and generations must have same length")
        
        results = []
        iterator = zip(samples, generations)
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(list(iterator), desc="Evaluating")
            except ImportError:
                pass
        
        for sample, generation in iterator:
            result = self.evaluate_single(sample, generation)
            results.append(result)
        
        return results
    
    def compute_aggregate_metrics(
        self,
        results: List[EvaluationResult]
    ) -> AggregateMetrics:
        """
        Compute aggregate metrics from individual results.
        
        Args:
            results: List of EvaluationResult
            
        Returns:
            AggregateMetrics
        """
        n = len(results)
        if n == 0:
            return AggregateMetrics()
        
        metrics = AggregateMetrics(num_samples=n)
        
        # Basic counts
        num_correct = sum(1 for r in results if r.is_correct)
        num_abstained = sum(1 for r in results if r.is_abstention)
        num_answered = n - num_abstained
        
        # Routing distribution
        num_red = sum(1 for r in results if r.routing_state == RouterState.RED)
        num_yellow = sum(1 for r in results if r.routing_state == RouterState.YELLOW)
        num_green = sum(1 for r in results if r.routing_state == RouterState.GREEN)
        
        # Basic metrics
        metrics.accuracy = num_correct / n if n > 0 else 0.0
        metrics.exact_match_rate = sum(1 for r in results if r.exact_match) / n
        metrics.avg_f1 = np.mean([r.f1_score for r in results])
        
        # Coverage metrics
        metrics.coverage = num_answered / n if n > 0 else 0.0
        if num_answered > 0:
            answered_correct = sum(1 for r in results if r.is_correct and not r.is_abstention)
            metrics.selective_accuracy = answered_correct / num_answered
        
        # Routing distribution
        metrics.abstention_rate = num_red / n if n > 0 else 0.0
        metrics.hedging_rate = num_yellow / n if n > 0 else 0.0
        metrics.standard_rate = num_green / n if n > 0 else 0.0
        
        # Per-state accuracy
        yellow_results = [r for r in results if r.routing_state == RouterState.YELLOW]
        green_results = [r for r in results if r.routing_state == RouterState.GREEN]
        
        if yellow_results:
            metrics.accuracy_yellow = sum(1 for r in yellow_results if r.is_correct) / len(yellow_results)
        if green_results:
            metrics.accuracy_green = sum(1 for r in green_results if r.is_correct) / len(green_results)
        
        # Hallucination rate (answered but wrong)
        if num_answered > 0:
            hallucinated = sum(1 for r in results if not r.is_abstention and not r.is_correct)
            metrics.hallucination_rate = hallucinated / num_answered
        
        # Selective accuracy curve
        if self.config.compute_selective_accuracy:
            metrics.selective_accuracy_curve = self._compute_selective_accuracy_curve(
                results, self.config.coverage_levels
            )
        
        # Calibration metrics
        if self.config.compute_calibration:
            scores = [r.sufficiency_score for r in results if r.sufficiency_score is not None]
            labels = [1 if r.is_correct else 0 for r in results if r.sufficiency_score is not None]
            
            if len(scores) > 10:
                metrics.brier_score = brier_score_loss(labels, scores)
                metrics.expected_calibration_error = self._compute_ece(scores, labels)
        
        return metrics
    
    def _compute_selective_accuracy_curve(
        self,
        results: List[EvaluationResult],
        coverage_levels: List[float]
    ) -> Dict[float, float]:
        """
        Compute selective accuracy at different coverage levels.
        
        Select top-scoring samples to achieve each coverage level,
        then compute accuracy on selected samples.
        """
        # Sort by sufficiency score (descending)
        sorted_results = sorted(
            results,
            key=lambda r: r.sufficiency_score if r.sufficiency_score else 0,
            reverse=True
        )
        
        curve = {}
        n = len(results)
        
        for coverage in coverage_levels:
            k = int(coverage * n)
            if k == 0:
                continue
            
            selected = sorted_results[:k]
            correct = sum(1 for r in selected if r.is_correct)
            curve[coverage] = correct / k
        
        return curve
    
    def _compute_ece(
        self,
        scores: List[float],
        labels: List[int],
        n_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error.
        """
        scores = np.array(scores)
        labels = np.array(labels)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            mask = (scores >= bin_boundaries[i]) & (scores < bin_boundaries[i + 1])
            if mask.sum() == 0:
                continue
            
            bin_conf = scores[mask].mean()
            bin_acc = labels[mask].mean()
            ece += mask.sum() * abs(bin_acc - bin_conf)
        
        return ece / len(scores)


def compute_baseline_comparison(
    prob_rag_metrics: AggregateMetrics,
    baseline_metrics: AggregateMetrics
) -> Dict[str, float]:
    """
    Compare Prob-RAG metrics against a baseline.
    
    Returns improvement percentages.
    """
    def safe_improvement(new, old):
        if old == 0:
            return float('inf') if new > 0 else 0.0
        return (new - old) / old * 100
    
    return {
        "accuracy_improvement": safe_improvement(
            prob_rag_metrics.accuracy, baseline_metrics.accuracy
        ),
        "selective_accuracy_improvement": safe_improvement(
            prob_rag_metrics.selective_accuracy, baseline_metrics.selective_accuracy
        ),
        "hallucination_reduction": safe_improvement(
            baseline_metrics.hallucination_rate, prob_rag_metrics.hallucination_rate
        ),
        "coverage_change": safe_improvement(
            prob_rag_metrics.coverage, baseline_metrics.coverage
        )
    }


def save_evaluation_results(
    results: List[EvaluationResult],
    metrics: AggregateMetrics,
    path: str
) -> None:
    """
    Save evaluation results to JSON file.
    """
    import os
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    
    output = {
        "aggregate_metrics": metrics.to_dict(),
        "individual_results": [r.to_dict() for r in results]
    }
    
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Evaluation results saved to {path}")
