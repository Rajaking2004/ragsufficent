"""
Prob-RAG: Probabilistic Sufficient Context RAG Architecture

A novel RAG architecture that uses log-probabilities for continuous sufficiency
scoring and a traffic-light routing system for adaptive response generation.

Architecture:
    Module A: Standard Retriever - Vector search against document database
    Module B: Probabilistic Sufficiency Scorer - Log-prob based scoring
    Module C: Dynamic Router (Traffic Light) - 3-state routing system
    Module D: Adaptive Generator - Context-aware response generation

Reference:
    Extends "Sufficient Context: A New Lens on RAG Systems" (Joren et al., ICLR 2025)
    with probabilistic scoring and adaptive generation protocols.

Author: Prob-RAG Research Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Prob-RAG Research Team"

from .config import ProbRAGConfig
from .pipeline import ProbRAGPipeline

__all__ = [
    "ProbRAGConfig",
    "ProbRAGPipeline",
    "__version__",
]
