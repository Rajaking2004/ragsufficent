"""
Module C: Traffic Light Router (Dynamic Router)

Routes queries to different generation protocols based on the
probabilistic sufficiency score using a 3-state traffic light system:

    🔴 RED (S < τ_low): Abstention Protocol
        - Context is insufficient
        - Trigger explicit abstention response
        
    🟡 YELLOW (τ_low ≤ S ≤ τ_high): Hedging Protocol  
        - Context is ambiguous/uncertain
        - Generate response with uncertainty markers
        
    🟢 GREEN (S > τ_high): Standard Protocol
        - Context is sufficient
        - Generate confident, direct response
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

import numpy as np

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
except ImportError:
    IsotonicRegression = None
    LogisticRegression = None

from ..config import RouterConfig, RouterState
from .scorer import SufficiencyScore


logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Container for routing decision results."""
    # Core decision
    state: RouterState
    score: float
    
    # Thresholds used
    tau_low: float
    tau_high: float
    
    # Confidence in the routing (how far from thresholds)
    confidence: float
    
    # Calibrated score (if calibration is used)
    calibrated_score: Optional[float] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        emoji = {"abstention": "🔴", "hedging": "🟡", "standard": "🟢"}
        return f"RoutingDecision({emoji[self.state.value]} {self.state.value}, score={self.score:.3f}, conf={self.confidence:.3f})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "score": self.score,
            "tau_low": self.tau_low,
            "tau_high": self.tau_high,
            "confidence": self.confidence,
            "calibrated_score": self.calibrated_score,
            "metadata": self.metadata
        }
    
    @property
    def is_abstention(self) -> bool:
        return self.state == RouterState.RED
    
    @property
    def is_hedging(self) -> bool:
        return self.state == RouterState.YELLOW
    
    @property
    def is_standard(self) -> bool:
        return self.state == RouterState.GREEN


class ScoreCalibrator:
    """
    Calibrates raw sufficiency scores to better reflect true probabilities.
    
    Uses either:
    - Isotonic Regression: Non-parametric, preserves ordering
    - Platt Scaling: Parametric (logistic regression)
    """
    
    def __init__(self, method: str = "isotonic"):
        """
        Initialize calibrator.
        
        Args:
            method: "isotonic" or "platt"
        """
        if method not in ["isotonic", "platt"]:
            raise ValueError(f"Unknown calibration method: {method}")
        
        self.method = method
        self.model = None
        self.is_fitted = False
    
    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "ScoreCalibrator":
        """
        Fit the calibration model.
        
        Args:
            scores: Raw sufficiency scores
            labels: Binary labels (1 = sufficient, 0 = insufficient)
            
        Returns:
            self
        """
        if IsotonicRegression is None:
            raise ImportError("scikit-learn required for calibration")
        
        scores = np.asarray(scores).reshape(-1)
        labels = np.asarray(labels).reshape(-1)
        
        if self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip")
            self.model.fit(scores, labels)
        else:  # platt
            self.model = LogisticRegression()
            self.model.fit(scores.reshape(-1, 1), labels)
        
        self.is_fitted = True
        logger.info(f"Calibrator fitted with {len(scores)} samples")
        return self
    
    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """
        Apply calibration to scores.
        
        Args:
            scores: Raw scores to calibrate
            
        Returns:
            Calibrated scores
        """
        if not self.is_fitted:
            logger.warning("Calibrator not fitted, returning raw scores")
            return scores
        
        scores = np.asarray(scores)
        original_shape = scores.shape
        scores = scores.reshape(-1)
        
        if self.method == "isotonic":
            calibrated = self.model.predict(scores)
        else:  # platt
            calibrated = self.model.predict_proba(scores.reshape(-1, 1))[:, 1]
        
        return calibrated.reshape(original_shape)
    
    def save(self, path: str) -> None:
        """Save calibrator to disk."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"method": self.method, "model": self.model, "is_fitted": self.is_fitted}, f)
    
    def load(self, path: str) -> "ScoreCalibrator":
        """Load calibrator from disk."""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.method = data["method"]
        self.model = data["model"]
        self.is_fitted = data["is_fitted"]
        return self


class TrafficLightRouter:
    """
    Module C: Traffic Light Router
    
    Routes queries to different generation protocols based on
    sufficiency scores using a 3-state traffic light system.
    
    This is a key innovation over binary routing:
    - Instead of just abstain/answer, we have a middle "hedging" state
    - This allows nuanced responses when context is partially sufficient
    """
    
    def __init__(self, config: RouterConfig):
        """
        Initialize the router.
        
        Args:
            config: RouterConfig with threshold and calibration settings
        """
        self.config = config
        self.tau_low = config.tau_low
        self.tau_high = config.tau_high
        
        # Validate thresholds
        if not (0 <= self.tau_low < self.tau_high <= 1):
            raise ValueError(
                f"Invalid thresholds: tau_low={self.tau_low}, tau_high={self.tau_high}. "
                "Must satisfy 0 ≤ tau_low < tau_high ≤ 1"
            )
        
        # Initialize calibrator if requested
        self.calibrator = None
        if config.use_calibration:
            self.calibrator = ScoreCalibrator(method=config.calibration_method)
        
        # Statistics tracking
        self.stats = {
            "total": 0,
            "red": 0,
            "yellow": 0,
            "green": 0
        }
    
    def _compute_confidence(self, score: float) -> float:
        """
        Compute confidence in the routing decision.
        
        Confidence is based on distance from the nearest threshold:
        - High confidence: Score is far from thresholds
        - Low confidence: Score is near a threshold
        
        Args:
            score: Sufficiency score
            
        Returns:
            Confidence value in [0, 1]
        """
        if score < self.tau_low:
            # RED zone: confidence based on distance from tau_low
            # At score=0, confidence=1; at score=tau_low, confidence=0
            confidence = (self.tau_low - score) / self.tau_low if self.tau_low > 0 else 1.0
            
        elif score > self.tau_high:
            # GREEN zone: confidence based on distance from tau_high
            # At score=tau_high, confidence=0; at score=1, confidence=1
            range_size = 1.0 - self.tau_high
            confidence = (score - self.tau_high) / range_size if range_size > 0 else 1.0
            
        else:
            # YELLOW zone: confidence is low (we're uncertain)
            # Confidence is highest at midpoint of yellow zone
            midpoint = (self.tau_low + self.tau_high) / 2
            half_width = (self.tau_high - self.tau_low) / 2
            
            if half_width > 0:
                distance_from_mid = abs(score - midpoint)
                confidence = 1.0 - (distance_from_mid / half_width)
            else:
                confidence = 0.5
            
            # Yellow zone confidence is inherently lower
            confidence *= 0.5
        
        return max(0.0, min(1.0, confidence))
    
    def route(self, score: SufficiencyScore) -> RoutingDecision:
        """
        Route a single score to the appropriate protocol.
        
        Args:
            score: SufficiencyScore from Module B
            
        Returns:
            RoutingDecision with state and metadata
        """
        raw_score = score.score
        
        # Apply calibration if available
        if self.calibrator is not None and self.calibrator.is_fitted:
            calibrated = float(self.calibrator.calibrate(np.array([raw_score]))[0])
        else:
            calibrated = None
        
        # Use calibrated score for routing if available
        routing_score = calibrated if calibrated is not None else raw_score
        
        # Determine state based on thresholds
        if routing_score < self.tau_low:
            state = RouterState.RED
            self.stats["red"] += 1
        elif routing_score > self.tau_high:
            state = RouterState.GREEN
            self.stats["green"] += 1
        else:
            state = RouterState.YELLOW
            self.stats["yellow"] += 1
        
        self.stats["total"] += 1
        
        # Compute confidence
        confidence = self._compute_confidence(routing_score)
        
        return RoutingDecision(
            state=state,
            score=raw_score,
            tau_low=self.tau_low,
            tau_high=self.tau_high,
            confidence=confidence,
            calibrated_score=calibrated,
            metadata={
                "logprob_yes": score.logprob_yes,
                "logprob_no": score.logprob_no,
                "response": score.response
            }
        )
    
    def route_batch(self, scores: List[SufficiencyScore]) -> List[RoutingDecision]:
        """
        Route multiple scores.
        
        Args:
            scores: List of SufficiencyScore objects
            
        Returns:
            List of RoutingDecision objects
        """
        return [self.route(score) for score in scores]
    
    def fit_calibrator(
        self,
        scores: List[float],
        labels: List[int]
    ) -> None:
        """
        Fit the score calibrator on labeled data.
        
        Args:
            scores: Raw sufficiency scores
            labels: Binary labels (1 = sufficient, 0 = insufficient)
        """
        if self.calibrator is None:
            self.calibrator = ScoreCalibrator(method=self.config.calibration_method)
        
        self.calibrator.fit(np.array(scores), np.array(labels))
        logger.info("Calibrator fitted successfully")
    
    def update_thresholds(self, tau_low: float, tau_high: float) -> None:
        """
        Update routing thresholds.
        
        Args:
            tau_low: New lower threshold
            tau_high: New upper threshold
        """
        if not (0 <= tau_low < tau_high <= 1):
            raise ValueError(f"Invalid thresholds: {tau_low}, {tau_high}")
        
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.config.tau_low = tau_low
        self.config.tau_high = tau_high
        
        logger.info(f"Thresholds updated: tau_low={tau_low}, tau_high={tau_high}")
    
    def optimize_thresholds(
        self,
        scores: List[float],
        labels: List[int],
        target_coverage: float = 0.8,
        target_accuracy: float = 0.9
    ) -> Tuple[float, float]:
        """
        Optimize thresholds to achieve target coverage and accuracy.
        
        Args:
            scores: Raw sufficiency scores
            labels: Binary labels
            target_coverage: Target fraction of samples to answer (not abstain)
            target_accuracy: Target accuracy on answered samples
            
        Returns:
            Tuple of (optimal_tau_low, optimal_tau_high)
        """
        scores = np.array(scores)
        labels = np.array(labels)
        
        best_tau_low = self.tau_low
        best_tau_high = self.tau_high
        best_score = float('-inf')
        
        # Grid search
        for tau_low in np.arange(0.1, 0.5, 0.05):
            for tau_high in np.arange(tau_low + 0.1, 0.9, 0.05):
                # Simulate routing
                red = scores < tau_low
                green = scores > tau_high
                yellow = ~red & ~green
                
                # Coverage: fraction not in red (we answer in yellow and green)
                coverage = (yellow.sum() + green.sum()) / len(scores)
                
                # Accuracy on green (where we're confident)
                if green.sum() > 0:
                    green_acc = labels[green].mean()
                else:
                    green_acc = 0
                
                # Objective: maximize weighted combination
                # Penalize if far from targets
                coverage_penalty = max(0, target_coverage - coverage) * 2
                accuracy_penalty = max(0, target_accuracy - green_acc) * 2
                
                score = coverage + green_acc - coverage_penalty - accuracy_penalty
                
                if score > best_score:
                    best_score = score
                    best_tau_low = tau_low
                    best_tau_high = tau_high
        
        logger.info(f"Optimal thresholds: tau_low={best_tau_low:.3f}, tau_high={best_tau_high:.3f}")
        return best_tau_low, best_tau_high
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get routing statistics."""
        total = self.stats["total"]
        if total == 0:
            return self.stats
        
        return {
            "total": total,
            "red_count": self.stats["red"],
            "yellow_count": self.stats["yellow"],
            "green_count": self.stats["green"],
            "red_ratio": self.stats["red"] / total,
            "yellow_ratio": self.stats["yellow"] / total,
            "green_ratio": self.stats["green"] / total,
            "tau_low": self.tau_low,
            "tau_high": self.tau_high
        }
    
    def reset_statistics(self) -> None:
        """Reset routing statistics."""
        self.stats = {"total": 0, "red": 0, "yellow": 0, "green": 0}
    
    def __repr__(self) -> str:
        return f"TrafficLightRouter(tau_low={self.tau_low}, tau_high={self.tau_high})"


def visualize_routing_distribution(
    decisions: List[RoutingDecision],
    title: str = "Routing Distribution"
) -> None:
    """
    Visualize the distribution of routing decisions.
    
    Args:
        decisions: List of RoutingDecision objects
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.warning("matplotlib required for visualization")
        return
    
    scores = [d.score for d in decisions]
    colors = []
    for d in decisions:
        if d.state == RouterState.RED:
            colors.append('red')
        elif d.state == RouterState.YELLOW:
            colors.append('gold')
        else:
            colors.append('green')
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(scores, bins=30, edgecolor='black', alpha=0.7)
    ax1.axvline(decisions[0].tau_low, color='red', linestyle='--', label=f'τ_low={decisions[0].tau_low}')
    ax1.axvline(decisions[0].tau_high, color='green', linestyle='--', label=f'τ_high={decisions[0].tau_high}')
    ax1.set_xlabel('Sufficiency Score')
    ax1.set_ylabel('Count')
    ax1.set_title('Score Distribution')
    ax1.legend()
    
    # Pie chart
    ax2 = axes[1]
    counts = [sum(1 for d in decisions if d.state == s) for s in [RouterState.RED, RouterState.YELLOW, RouterState.GREEN]]
    labels = ['Abstention', 'Hedging', 'Standard']
    colors_pie = ['red', 'gold', 'green']
    ax2.pie(counts, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Routing Distribution')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('routing_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info("Routing visualization saved to routing_distribution.png")
