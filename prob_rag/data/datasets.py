"""
Dataset Loaders for Prob-RAG

Implements loaders for standard RAG evaluation datasets:
- HotPotQA: Multi-hop reasoning
- Musique-Ans: Complex multi-hop QA
- Natural Questions: Single-hop factual QA
- TriviaQA: Trivia questions
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Iterator, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import random

try:
    from datasets import load_dataset, Dataset
except ImportError:
    load_dataset = None
    Dataset = None

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class RAGSample:
    """
    Standard format for RAG evaluation samples.
    """
    # Identifiers
    id: str
    
    # Core data
    question: str
    contexts: List[str]  # Retrieved/gold contexts
    answer: str  # Ground truth answer
    
    # Optional fields
    answer_aliases: List[str] = field(default_factory=list)  # Alternative correct answers
    supporting_facts: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    difficulty: Optional[str] = None  # easy, medium, hard
    question_type: Optional[str] = None  # comparison, bridge, etc.
    num_hops: Optional[int] = None  # Number of reasoning hops required
    
    # Dataset source
    dataset_name: str = ""
    split: str = ""
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"RAGSample(id={self.id}, question='{self.question[:50]}...', num_contexts={len(self.contexts)})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "contexts": self.contexts,
            "answer": self.answer,
            "answer_aliases": self.answer_aliases,
            "difficulty": self.difficulty,
            "question_type": self.question_type,
            "num_hops": self.num_hops,
            "dataset_name": self.dataset_name,
            "split": self.split,
            "metadata": self.metadata
        }
    
    @property
    def combined_context(self) -> str:
        """Combine all contexts into a single string."""
        return "\n\n---\n\n".join(self.contexts)
    
    @property
    def all_answers(self) -> List[str]:
        """Get all valid answers including aliases."""
        return [self.answer] + self.answer_aliases


class BaseDatasetLoader(ABC):
    """Abstract base class for dataset loaders."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self.dataset = None
        self.name = "base"
    
    @abstractmethod
    def load(self, split: str = "validation") -> List[RAGSample]:
        """Load dataset and return as list of RAGSample."""
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        pass
    
    def sample(self, n: int, split: str = "validation", seed: int = 42) -> List[RAGSample]:
        """Sample n examples from the dataset."""
        data = self.load(split)
        random.seed(seed)
        return random.sample(data, min(n, len(data)))
    
    def iterate(self, split: str = "validation", batch_size: int = 32) -> Iterator[List[RAGSample]]:
        """Iterate over dataset in batches."""
        data = self.load(split)
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]


class HotPotQALoader(BaseDatasetLoader):
    """
    Loader for HotPotQA dataset.
    
    HotPotQA is a multi-hop question answering dataset featuring
    natural, multi-hop questions that require reasoning over
    multiple Wikipedia paragraphs.
    """
    
    def __init__(self, cache_dir: Optional[str] = None, subset: str = "distractor"):
        """
        Initialize HotPotQA loader.
        
        Args:
            cache_dir: Cache directory for downloaded data
            subset: "distractor" or "fullwiki"
        """
        super().__init__(cache_dir)
        self.name = "hotpotqa"
        self.subset = subset
        self._loaded_splits = {}
    
    def load(self, split: str = "validation") -> List[RAGSample]:
        """
        Load HotPotQA dataset.
        
        Args:
            split: "train" or "validation"
            
        Returns:
            List of RAGSample objects
        """
        if split in self._loaded_splits:
            return self._loaded_splits[split]
        
        if load_dataset is None:
            raise ImportError("datasets package required. Install with: pip install datasets")
        
        logger.info(f"Loading HotPotQA ({self.subset}) {split} split...")
        
        dataset = load_dataset(
            "hotpot_qa",
            self.subset,
            split=split,
            cache_dir=self.cache_dir
        )
        
        samples = []
        for idx, item in enumerate(dataset):
            # Extract contexts from the supporting paragraphs
            contexts = []
            for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
                context_text = f"[{title}]\n" + " ".join(sentences)
                contexts.append(context_text)
            
            # Create sample
            sample = RAGSample(
                id=item["id"],
                question=item["question"],
                contexts=contexts,
                answer=item["answer"],
                answer_aliases=[],
                supporting_facts=list(zip(
                    item["supporting_facts"]["title"],
                    item["supporting_facts"]["sent_id"]
                )) if "supporting_facts" in item else [],
                difficulty=item.get("level", None),
                question_type=item.get("type", None),
                num_hops=2,  # HotPotQA is 2-hop by design
                dataset_name="hotpotqa",
                split=split,
                metadata={"subset": self.subset}
            )
            samples.append(sample)
        
        self._loaded_splits[split] = samples
        logger.info(f"Loaded {len(samples)} samples from HotPotQA {split}")
        return samples
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {"name": "hotpotqa", "subset": self.subset}
        
        for split in ["train", "validation"]:
            try:
                data = self.load(split)
                stats[split] = {
                    "num_samples": len(data),
                    "avg_contexts": np.mean([len(s.contexts) for s in data]),
                    "avg_question_length": np.mean([len(s.question.split()) for s in data]),
                    "avg_context_length": np.mean([
                        sum(len(c.split()) for c in s.contexts) for s in data
                    ]),
                    "difficulty_distribution": {
                        level: sum(1 for s in data if s.difficulty == level)
                        for level in ["easy", "medium", "hard"]
                    }
                }
            except Exception as e:
                logger.warning(f"Could not load {split}: {e}")
        
        return stats


class MusiqueLoader(BaseDatasetLoader):
    """
    Loader for Musique-Ans dataset.
    
    Musique is a multi-hop QA dataset with complex compositional
    questions requiring 2-4 hops of reasoning.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(cache_dir)
        self.name = "musique"
        self._loaded_splits = {}
    
    def load(self, split: str = "validation") -> List[RAGSample]:
        """
        Load Musique dataset.
        
        Args:
            split: "train" or "validation"
            
        Returns:
            List of RAGSample objects
        """
        if split in self._loaded_splits:
            return self._loaded_splits[split]
        
        if load_dataset is None:
            raise ImportError("datasets package required")
        
        logger.info(f"Loading Musique {split} split...")
        
        # Musique uses 'ans' subset
        dataset = load_dataset(
            "dgslibiern/MuSiQue",
            split=split,
            cache_dir=self.cache_dir
        )
        
        samples = []
        for item in dataset:
            # Extract paragraphs
            contexts = []
            if "paragraphs" in item:
                for para in item["paragraphs"]:
                    if isinstance(para, dict):
                        text = para.get("paragraph_text", para.get("text", ""))
                        title = para.get("title", "")
                        contexts.append(f"[{title}]\n{text}" if title else text)
                    else:
                        contexts.append(str(para))
            
            # Handle answer
            answer = item.get("answer", item.get("answers", [""])[0] if "answers" in item else "")
            if isinstance(answer, list):
                answer = answer[0] if answer else ""
            
            sample = RAGSample(
                id=item.get("id", str(len(samples))),
                question=item["question"],
                contexts=contexts if contexts else [item.get("context", "")],
                answer=answer,
                num_hops=item.get("num_hops", None),
                dataset_name="musique",
                split=split
            )
            samples.append(sample)
        
        self._loaded_splits[split] = samples
        logger.info(f"Loaded {len(samples)} samples from Musique {split}")
        return samples
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {"name": "musique"}
        
        for split in ["train", "validation"]:
            try:
                data = self.load(split)
                hop_dist = {}
                for s in data:
                    hops = s.num_hops or "unknown"
                    hop_dist[hops] = hop_dist.get(hops, 0) + 1
                
                stats[split] = {
                    "num_samples": len(data),
                    "avg_contexts": np.mean([len(s.contexts) for s in data]),
                    "hop_distribution": hop_dist
                }
            except Exception as e:
                logger.warning(f"Could not load {split}: {e}")
        
        return stats


class NaturalQuestionsLoader(BaseDatasetLoader):
    """
    Loader for Natural Questions Open dataset.
    
    Simplified version of NQ with single-hop factual questions from Google search queries.
    Source: google-research-datasets/nq_open
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(cache_dir)
        self.name = "natural_questions"
        self._loaded_splits = {}
    
    def load(self, split: str = "validation") -> List[RAGSample]:
        """Load Natural Questions Open dataset."""
        if split in self._loaded_splits:
            return self._loaded_splits[split]
        
        if load_dataset is None:
            raise ImportError("datasets package required")
        
        logger.info(f"Loading Natural Questions Open {split} split...")
        
        # Use nq_open which is the simplified, pre-processed version
        dataset = load_dataset(
            "google-research-datasets/nq_open",
            split=split,
            cache_dir=self.cache_dir
        )
        
        samples = []
        for idx, item in enumerate(dataset):
            question = item.get("question", "")
            answers = item.get("answer", [])
            
            # nq_open provides question and answer, but no context
            # For RAG testing, we use the question as a retrieval query
            # Context would be retrieved by the retriever module
            
            # Primary answer (first one)
            primary_answer = answers[0] if answers else ""
            
            sample = RAGSample(
                id=f"nq_{idx}",
                question=question,
                contexts=[],  # No pre-retrieved context in nq_open
                answer=primary_answer,
                answer_aliases=answers[1:] if len(answers) > 1 else [],
                num_hops=1,
                dataset_name="natural_questions",
                split=split
            )
            samples.append(sample)
        
        self._loaded_splits[split] = samples
        logger.info(f"Loaded {len(samples)} samples from Natural Questions Open {split}")
        return samples
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {"name": "natural_questions_open", "source": "google-research-datasets/nq_open"}


class TriviaQALoader(BaseDatasetLoader):
    """
    Loader for TriviaQA dataset.
    
    Large-scale QA dataset with trivia questions.
    """
    
    def __init__(self, cache_dir: Optional[str] = None, subset: str = "rc"):
        super().__init__(cache_dir)
        self.name = "triviaqa"
        self.subset = subset  # "rc" or "unfiltered"
        self._loaded_splits = {}
    
    def load(self, split: str = "validation") -> List[RAGSample]:
        """Load TriviaQA dataset."""
        if split in self._loaded_splits:
            return self._loaded_splits[split]
        
        if load_dataset is None:
            raise ImportError("datasets package required")
        
        logger.info(f"Loading TriviaQA ({self.subset}) {split} split...")
        
        dataset = load_dataset(
            "trivia_qa",
            self.subset,
            split=split,
            cache_dir=self.cache_dir
        )
        
        samples = []
        for item in dataset:
            # Get answer
            answer_obj = item.get("answer", {})
            answer = answer_obj.get("value", "")
            aliases = answer_obj.get("aliases", [])
            
            # Get context from search results or entity pages
            contexts = []
            search_results = item.get("search_results", {})
            if search_results:
                for result in search_results.get("search_context", [])[:3]:
                    if result:
                        contexts.append(result)
            
            entity_pages = item.get("entity_pages", {})
            if entity_pages:
                for page in entity_pages.get("wiki_context", [])[:2]:
                    if page:
                        contexts.append(page)
            
            if not contexts:
                continue
            
            sample = RAGSample(
                id=item.get("question_id", str(len(samples))),
                question=item["question"],
                contexts=contexts,
                answer=answer,
                answer_aliases=aliases,
                num_hops=1,
                dataset_name="triviaqa",
                split=split,
                metadata={"subset": self.subset}
            )
            samples.append(sample)
        
        self._loaded_splits[split] = samples
        logger.info(f"Loaded {len(samples)} samples from TriviaQA {split}")
        return samples
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {"name": "triviaqa", "subset": self.subset}


class SyntheticDatasetLoader(BaseDatasetLoader):
    """
    Generate synthetic dataset for testing.
    Useful when real datasets are not available.
    """
    
    def __init__(self, num_samples: int = 100, cache_dir: Optional[str] = None, **kwargs):
        super().__init__(cache_dir)
        self.name = "synthetic"
        self.num_samples = num_samples
        self._data = None
    
    def load(self, split: str = "validation") -> List[RAGSample]:
        """Generate synthetic samples."""
        if self._data is not None:
            return self._data
        
        logger.info(f"Generating {self.num_samples} synthetic samples...")
        
        # Sample templates
        templates = [
            {
                "question": "What year was {entity} founded?",
                "context": "{entity} is a {type}. It was founded in {year} by {founder}.",
                "answer": "{year}",
                "sufficient": True
            },
            {
                "question": "Who founded {entity}?",
                "context": "{entity} was established as a {type}. It began operations in {year}.",
                "answer": "Unknown",
                "sufficient": False  # Founder not mentioned
            },
            {
                "question": "What is the capital of {country}?",
                "context": "{country} is located in {region}. Its capital city is {capital}.",
                "answer": "{capital}",
                "sufficient": True
            },
            {
                "question": "How many people live in {city}?",
                "context": "{city} is a major city in {country}. It is known for {feature}.",
                "answer": "Unknown",
                "sufficient": False  # Population not mentioned
            },
        ]
        
        entities = ["Microsoft", "Google", "Apple", "Amazon", "Tesla", "OpenAI"]
        countries = ["France", "Germany", "Japan", "Brazil", "Canada", "Australia"]
        cities = ["Paris", "Berlin", "Tokyo", "São Paulo", "Toronto", "Sydney"]
        years = ["1975", "1998", "2004", "2010", "2015", "2020"]
        
        samples = []
        for i in range(self.num_samples):
            template = random.choice(templates)
            
            # Fill in template
            question = template["question"].format(
                entity=random.choice(entities),
                country=random.choice(countries),
                city=random.choice(cities)
            )
            context = template["context"].format(
                entity=random.choice(entities),
                type=random.choice(["company", "organization", "startup"]),
                year=random.choice(years),
                founder=random.choice(["founders", "entrepreneurs"]),
                country=random.choice(countries),
                region=random.choice(["Europe", "Asia", "Americas"]),
                capital=random.choice(cities),
                city=random.choice(cities),
                feature=random.choice(["culture", "technology", "history"])
            )
            
            sample = RAGSample(
                id=f"synthetic_{i}",
                question=question,
                contexts=[context],
                answer=template["answer"].format(
                    year=random.choice(years),
                    capital=random.choice(cities)
                ),
                dataset_name="synthetic",
                split=split,
                metadata={"sufficient": template["sufficient"]}
            )
            samples.append(sample)
        
        self._data = samples
        return samples
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            "name": "synthetic",
            "num_samples": self.num_samples
        }


# Dataset registry
DATASETS = {
    "hotpotqa": HotPotQALoader,
    "musique": MusiqueLoader,
    "natural_questions": NaturalQuestionsLoader,
    "triviaqa": TriviaQALoader,
    "synthetic": SyntheticDatasetLoader,
}


def get_dataset_loader(
    name: str,
    cache_dir: Optional[str] = None,
    **kwargs
) -> BaseDatasetLoader:
    """
    Get a dataset loader by name.
    
    Args:
        name: Dataset name (hotpotqa, musique, natural_questions, triviaqa, synthetic)
        cache_dir: Cache directory
        **kwargs: Additional arguments for specific loaders
        
    Returns:
        Dataset loader instance
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
    
    return DATASETS[name](cache_dir=cache_dir, **kwargs)


def load_dataset_samples(
    name: str,
    split: str = "validation",
    num_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    seed: int = 42,
    **kwargs
) -> List[RAGSample]:
    """
    Convenience function to load dataset samples.
    
    Args:
        name: Dataset name
        split: Data split
        num_samples: Number of samples (None for all)
        cache_dir: Cache directory
        seed: Random seed for sampling
        **kwargs: Additional loader arguments
        
    Returns:
        List of RAGSample objects
    """
    loader = get_dataset_loader(name, cache_dir, **kwargs)
    
    if num_samples is not None:
        return loader.sample(num_samples, split, seed)
    else:
        return loader.load(split)
