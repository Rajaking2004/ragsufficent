"""
Data module for Prob-RAG.

Contains dataset loaders and data utilities.
"""

from .datasets import (
    RAGSample,
    BaseDatasetLoader,
    HotPotQALoader,
    MusiqueLoader,
    NaturalQuestionsLoader,
    TriviaQALoader,
    SyntheticDatasetLoader,
    get_dataset_loader,
    load_dataset_samples,
    DATASETS,
)

__all__ = [
    "RAGSample",
    "BaseDatasetLoader",
    "HotPotQALoader",
    "MusiqueLoader",
    "NaturalQuestionsLoader",
    "TriviaQALoader",
    "SyntheticDatasetLoader",
    "get_dataset_loader",
    "load_dataset_samples",
    "DATASETS",
]
