"""
Modules package for Prob-RAG architecture.

Contains:
    - Module A: Retriever
    - Module B: Probabilistic Sufficiency Scorer
    - Module C: Traffic Light Router
    - Module D: Adaptive Generator
"""

from .retriever import Retriever, PassthroughRetriever, RetrievalResult, create_retriever
from .scorer import ProbabilisticSufficiencyScorer, SufficiencyScore
from .router import TrafficLightRouter, RoutingDecision
from .generator import AdaptiveGenerator, GenerationResult

__all__ = [
    # Module A
    "Retriever",
    "PassthroughRetriever",
    "RetrievalResult",
    "create_retriever",
    # Module B
    "ProbabilisticSufficiencyScorer",
    "SufficiencyScore",
    # Module C
    "TrafficLightRouter",
    "RoutingDecision",
    # Module D
    "AdaptiveGenerator",
    "GenerationResult",
]
