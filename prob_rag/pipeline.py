"""
Main Pipeline for Prob-RAG

Integrates all modules into a cohesive end-to-end pipeline:
    Query → Retrieval → Sufficiency Scoring → Routing → Adaptive Generation → Evaluation
"""

import logging
import os
import json
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import time

from .config import ProbRAGConfig, RouterState
from .modules.retriever import (
    Retriever, PassthroughRetriever, RetrievalResult, create_retriever
)
from .modules.scorer import (
    ProbabilisticSufficiencyScorer, SufficiencyScore, MockScorer, create_scorer
)
from .modules.router import (
    TrafficLightRouter, RoutingDecision, visualize_routing_distribution
)
from .modules.generator import (
    AdaptiveGenerator, GenerationResult, MockGenerator, create_generator
)
from .data.datasets import RAGSample, load_dataset_samples
from .evaluation.metrics import (
    Evaluator, EvaluationResult, AggregateMetrics, save_evaluation_results
)


logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Container for full pipeline results on a single sample."""
    # Input
    sample_id: str
    question: str
    
    # Module outputs
    retrieval: Optional[RetrievalResult] = None
    sufficiency: Optional[SufficiencyScore] = None
    routing: Optional[RoutingDecision] = None
    generation: Optional[GenerationResult] = None
    evaluation: Optional[EvaluationResult] = None
    
    # Timing
    retrieval_time: float = 0.0
    scoring_time: float = 0.0
    routing_time: float = 0.0
    generation_time: float = 0.0
    total_time: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "question": self.question,
            "retrieval": self.retrieval.to_dict() if self.retrieval else None,
            "sufficiency": self.sufficiency.to_dict() if self.sufficiency else None,
            "routing": self.routing.to_dict() if self.routing else None,
            "generation": self.generation.to_dict() if self.generation else None,
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
            "timing": {
                "retrieval": self.retrieval_time,
                "scoring": self.scoring_time,
                "routing": self.routing_time,
                "generation": self.generation_time,
                "total": self.total_time
            },
            "metadata": self.metadata
        }


@dataclass 
class ExperimentResults:
    """Container for full experiment results."""
    # Experiment info
    experiment_name: str
    dataset_name: str
    config: Dict[str, Any]
    timestamp: str
    
    # Results
    pipeline_results: List[PipelineResult]
    aggregate_metrics: AggregateMetrics
    
    # Timing
    total_runtime: float
    avg_time_per_sample: float
    
    # Additional analysis
    routing_statistics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp,
            "config": self.config,
            "aggregate_metrics": self.aggregate_metrics.to_dict(),
            "routing_statistics": self.routing_statistics,
            "timing": {
                "total_runtime": self.total_runtime,
                "avg_time_per_sample": self.avg_time_per_sample
            },
            "num_samples": len(self.pipeline_results)
        }
    
    def save(self, path: str) -> None:
        """Save experiment results to JSON."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        # Save detailed results separately
        detailed_path = path.replace('.json', '_detailed.json')
        with open(detailed_path, 'w') as f:
            json.dump({
                "results": [r.to_dict() for r in self.pipeline_results]
            }, f, indent=2)
        
        logger.info(f"Results saved to {path}")


class ProbRAGPipeline:
    """
    Main Prob-RAG Pipeline
    
    Orchestrates the full flow:
    1. Retrieval (Module A)
    2. Sufficiency Scoring (Module B)
    3. Traffic Light Routing (Module C)
    4. Adaptive Generation (Module D)
    5. Evaluation
    """
    
    def __init__(
        self,
        config: ProbRAGConfig,
        use_mock: bool = False
    ):
        """
        Initialize the pipeline.
        
        Args:
            config: ProbRAGConfig with all module configurations
            use_mock: If True, use mock modules (no API calls)
        """
        self.config = config
        self.use_mock = use_mock
        
        # Validate config (don't require API keys if using mock mode)
        if not config.validate(require_api_keys=(not use_mock)):
            raise ValueError("Invalid configuration")
        
        # Initialize modules
        self._initialize_modules()
        
        # Initialize evaluator
        self.evaluator = Evaluator(config.evaluation, config.openai_api_key)
        
        logger.info(f"ProbRAGPipeline initialized (mock={use_mock})")
    
    def _initialize_modules(self) -> None:
        """Initialize all pipeline modules."""
        # Module A: Retriever (passthrough for datasets with pre-retrieved contexts)
        self.retriever = PassthroughRetriever(self.config.retriever)
        
        # Determine which API key to use based on model name
        model_name = self.config.scorer.model_name.lower()
        if "gemini" in model_name:
            api_key = self.config.gemini_api_key
        elif "claude" in model_name:
            api_key = self.config.anthropic_api_key
        elif "llama" in model_name or "mixtral" in model_name or "gemma" in model_name:
            api_key = self.config.groq_api_key
        else:
            api_key = self.config.openai_api_key
        
        # Module B: Scorer
        self.scorer = create_scorer(
            self.config.scorer,
            api_key,
            use_mock=self.use_mock
        )
        
        # Module C: Router
        self.router = TrafficLightRouter(self.config.router)
        
        # Use same API key selection for generator
        gen_model = self.config.generator.model_name.lower()
        if "gemini" in gen_model:
            gen_api_key = self.config.gemini_api_key
        elif "claude" in gen_model:
            gen_api_key = self.config.anthropic_api_key
        elif "llama" in gen_model or "mixtral" in gen_model or "gemma" in gen_model:
            gen_api_key = self.config.groq_api_key
        else:
            gen_api_key = self.config.openai_api_key
        
        # Module D: Generator
        self.generator = create_generator(
            self.config.generator,
            gen_api_key,
            use_mock=self.use_mock
        )
    
    def process_single(
        self,
        sample: RAGSample,
        evaluate: bool = True
    ) -> PipelineResult:
        """
        Process a single sample through the full pipeline.
        
        Args:
            sample: RAGSample with question, contexts, and answer
            evaluate: Whether to evaluate against ground truth
            
        Returns:
            PipelineResult with all module outputs
        """
        start_time = time.time()
        result = PipelineResult(
            sample_id=sample.id,
            question=sample.question
        )
        
        # Step 1: Retrieval (passthrough since contexts provided)
        t0 = time.time()
        retrieval = self.retriever.retrieve(
            sample.question,
            sample.contexts
        )
        result.retrieval = retrieval
        result.retrieval_time = time.time() - t0
        
        # Step 2: Sufficiency Scoring
        t0 = time.time()
        sufficiency = self.scorer.score(
            sample.question,
            retrieval.combined_context
        )
        result.sufficiency = sufficiency
        result.scoring_time = time.time() - t0
        
        # Step 3: Routing
        t0 = time.time()
        routing = self.router.route(sufficiency)
        result.routing = routing
        result.routing_time = time.time() - t0
        
        # Step 4: Generation
        t0 = time.time()
        generation = self.generator.generate(
            sample.question,
            retrieval.combined_context,
            routing
        )
        result.generation = generation
        result.generation_time = time.time() - t0
        
        # Step 5: Evaluation (optional)
        if evaluate:
            evaluation = self.evaluator.evaluate_single(sample, generation)
            result.evaluation = evaluation
        
        result.total_time = time.time() - start_time
        
        return result
    
    def process_batch(
        self,
        samples: List[RAGSample],
        evaluate: bool = True,
        show_progress: bool = True
    ) -> List[PipelineResult]:
        """
        Process multiple samples through the pipeline.
        
        Args:
            samples: List of RAGSample objects
            evaluate: Whether to evaluate
            show_progress: Whether to show progress bar
            
        Returns:
            List of PipelineResult objects
        """
        results = []
        iterator = samples
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(samples, desc="Processing")
            except ImportError:
                pass
        
        for sample in iterator:
            try:
                result = self.process_single(sample, evaluate)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing sample {sample.id}: {e}")
                # Create error result
                results.append(PipelineResult(
                    sample_id=sample.id,
                    question=sample.question,
                    metadata={"error": str(e)}
                ))
        
        return results
    
    def run_experiment(
        self,
        dataset_name: str,
        split: str = "validation",
        num_samples: Optional[int] = None,
        experiment_name: Optional[str] = None,
        seed: int = 42
    ) -> ExperimentResults:
        """
        Run a full experiment on a dataset.
        
        Args:
            dataset_name: Name of dataset (hotpotqa, musique, etc.)
            split: Data split
            num_samples: Number of samples (None for all)
            experiment_name: Name for this experiment
            seed: Random seed
            
        Returns:
            ExperimentResults with full metrics
        """
        if experiment_name is None:
            experiment_name = f"prob_rag_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting experiment: {experiment_name}")
        logger.info(f"Dataset: {dataset_name}, Split: {split}, Samples: {num_samples or 'all'}")
        
        # Load dataset
        samples = load_dataset_samples(
            dataset_name,
            split=split,
            num_samples=num_samples,
            seed=seed
        )
        logger.info(f"Loaded {len(samples)} samples")
        
        # Process samples
        start_time = time.time()
        pipeline_results = self.process_batch(samples, evaluate=True)
        total_runtime = time.time() - start_time
        
        # Compute aggregate metrics
        generations = [r.generation for r in pipeline_results if r.generation]
        evaluations = [r.evaluation for r in pipeline_results if r.evaluation]
        
        aggregate_metrics = self.evaluator.compute_aggregate_metrics(evaluations)
        
        # Get routing statistics
        routing_stats = self.router.get_statistics()
        
        # Create experiment results
        results = ExperimentResults(
            experiment_name=experiment_name,
            dataset_name=dataset_name,
            config=self._serialize_config(),
            timestamp=datetime.now().isoformat(),
            pipeline_results=pipeline_results,
            aggregate_metrics=aggregate_metrics,
            total_runtime=total_runtime,
            avg_time_per_sample=total_runtime / len(samples) if samples else 0,
            routing_statistics=routing_stats
        )
        
        # Log summary
        logger.info(f"Experiment complete: {experiment_name}")
        logger.info(f"Accuracy: {aggregate_metrics.accuracy:.3f}")
        logger.info(f"Coverage: {aggregate_metrics.coverage:.3f}")
        logger.info(f"Selective Accuracy: {aggregate_metrics.selective_accuracy:.3f}")
        logger.info(f"Runtime: {total_runtime:.2f}s ({results.avg_time_per_sample:.2f}s per sample)")
        
        return results
    
    def _serialize_config(self) -> Dict[str, Any]:
        """Serialize config for saving."""
        return {
            "router": {
                "tau_low": self.config.router.tau_low,
                "tau_high": self.config.router.tau_high
            },
            "scorer": {
                "model_name": self.config.scorer.model_name
            },
            "generator": {
                "model_name": self.config.generator.model_name,
                "temperature": self.config.generator.temperature
            }
        }
    
    def update_thresholds(self, tau_low: float, tau_high: float) -> None:
        """Update router thresholds."""
        self.router.update_thresholds(tau_low, tau_high)
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get current routing statistics."""
        return self.router.get_statistics()
    
    def reset_statistics(self) -> None:
        """Reset routing statistics."""
        self.router.reset_statistics()


class BaselinePipeline:
    """
    Baseline RAG pipeline without probabilistic routing.
    
    For comparison: uses simple binary sufficiency (threshold at 0.5)
    and only standard generation (no hedging).
    """
    
    def __init__(
        self,
        config: ProbRAGConfig,
        use_mock: bool = False
    ):
        self.config = config
        self.use_mock = use_mock
        
        # Use binary threshold
        self.threshold = 0.5
        
        # Initialize modules
        self.retriever = PassthroughRetriever(config.retriever)
        self.scorer = create_scorer(config.scorer, config.openai_api_key, use_mock)
        self.generator = create_generator(config.generator, config.openai_api_key, use_mock)
        self.evaluator = Evaluator(config.evaluation, config.openai_api_key)
    
    def process_single(self, sample: RAGSample) -> PipelineResult:
        """Process with binary threshold."""
        result = PipelineResult(
            sample_id=sample.id,
            question=sample.question
        )
        
        # Retrieval
        retrieval = self.retriever.retrieve(sample.question, sample.contexts)
        result.retrieval = retrieval
        
        # Scoring
        sufficiency = self.scorer.score(sample.question, retrieval.combined_context)
        result.sufficiency = sufficiency
        
        # Binary routing: abstain if score < 0.5, else standard
        if sufficiency.score < self.threshold:
            state = RouterState.RED
        else:
            state = RouterState.GREEN
        
        routing = RoutingDecision(
            state=state,
            score=sufficiency.score,
            tau_low=self.threshold,
            tau_high=self.threshold,
            confidence=abs(sufficiency.score - self.threshold)
        )
        result.routing = routing
        
        # Generation
        generation = self.generator.generate(
            sample.question,
            retrieval.combined_context,
            routing
        )
        result.generation = generation
        
        # Evaluation
        result.evaluation = self.evaluator.evaluate_single(sample, generation)
        
        return result
    
    def run_experiment(
        self,
        dataset_name: str,
        split: str = "validation",
        num_samples: Optional[int] = None,
        experiment_name: Optional[str] = None,
        seed: int = 42
    ) -> ExperimentResults:
        """Run baseline experiment."""
        if experiment_name is None:
            experiment_name = f"baseline_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        samples = load_dataset_samples(dataset_name, split, num_samples, seed=seed)
        
        start_time = time.time()
        results = []
        
        try:
            from tqdm import tqdm
            iterator = tqdm(samples, desc="Baseline Processing")
        except ImportError:
            iterator = samples
        
        for sample in iterator:
            results.append(self.process_single(sample))
        
        total_runtime = time.time() - start_time
        
        evaluations = [r.evaluation for r in results if r.evaluation]
        metrics = self.evaluator.compute_aggregate_metrics(evaluations)
        
        return ExperimentResults(
            experiment_name=experiment_name,
            dataset_name=dataset_name,
            config={"threshold": self.threshold, "type": "baseline"},
            timestamp=datetime.now().isoformat(),
            pipeline_results=results,
            aggregate_metrics=metrics,
            total_runtime=total_runtime,
            avg_time_per_sample=total_runtime / len(samples) if samples else 0
        )


def compare_experiments(
    prob_rag_results: ExperimentResults,
    baseline_results: ExperimentResults
) -> Dict[str, Any]:
    """
    Compare Prob-RAG results against baseline.
    
    Returns comparison metrics and improvements.
    """
    pr_metrics = prob_rag_results.aggregate_metrics
    bl_metrics = baseline_results.aggregate_metrics
    
    def improvement(new, old):
        if old == 0:
            return float('inf') if new > 0 else 0
        return ((new - old) / old) * 100
    
    return {
        "prob_rag": pr_metrics.to_dict(),
        "baseline": bl_metrics.to_dict(),
        "improvements": {
            "accuracy": improvement(pr_metrics.accuracy, bl_metrics.accuracy),
            "selective_accuracy": improvement(pr_metrics.selective_accuracy, bl_metrics.selective_accuracy),
            "hallucination_reduction": improvement(bl_metrics.hallucination_rate, pr_metrics.hallucination_rate),
            "coverage_change": improvement(pr_metrics.coverage, bl_metrics.coverage)
        },
        "prob_rag_routing": prob_rag_results.routing_statistics
    }
