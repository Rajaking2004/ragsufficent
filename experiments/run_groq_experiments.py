"""
Prob-RAG Experiments with Groq API

This script runs experiments on real datasets (HotPotQA, Natural Questions, TriviaQA)
using the Groq API for fast inference with Llama models.
"""

import os
import sys
import json
import time
import random
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm

from prob_rag.config import (
    ProbRAGConfig, 
    ScorerConfig, 
    RouterConfig, 
    GeneratorConfig,
    RetrieverConfig,
    RouterState
)
from prob_rag.modules.scorer import ProbabilisticSufficiencyScorer, ScorerConfig
from prob_rag.modules.router import TrafficLightRouter, RouterConfig
from prob_rag.modules.generator import AdaptiveGenerator, GeneratorConfig
from prob_rag.data.datasets import (
    HotPotQALoader,
    NaturalQuestionsLoader, 
    TriviaQALoader,
    RAGSample
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Container for experiment results."""
    question_id: str
    question: str
    context: str
    gold_answer: str
    
    # Prob-RAG outputs
    sufficiency_score: float
    router_state: str
    confidence: float
    generated_answer: str
    
    # Evaluation
    is_correct: bool
    response_type: str  # "answered", "hedged", "abstained"
    
    # Timing
    scoring_time: float
    generation_time: float


class ProbRAGExperiment:
    """Run Prob-RAG experiments on datasets."""
    
    def __init__(
        self,
        groq_api_key: str,
        scorer_model: str = "llama-3.1-8b-instant",
        generator_model: str = "llama-3.1-8b-instant",
        tau_low: float = 0.3,
        tau_high: float = 0.7,
        output_dir: str = "./results"
    ):
        """
        Initialize experiment.
        
        Args:
            groq_api_key: Groq API key
            scorer_model: Model for sufficiency scoring
            generator_model: Model for answer generation
            tau_low: Lower threshold for RED state
            tau_high: Upper threshold for GREEN state
            output_dir: Directory to save results
        """
        self.api_key = groq_api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize scorer
        scorer_config = ScorerConfig(
            model_name=scorer_model,
            temperature=0.0,
            max_tokens=5
        )
        self.scorer = ProbabilisticSufficiencyScorer(scorer_config, api_key=groq_api_key)
        
        # Initialize router
        router_config = RouterConfig(
            tau_low=tau_low,
            tau_high=tau_high
        )
        self.router = TrafficLightRouter(router_config)
        
        # Initialize generator
        generator_config = GeneratorConfig(
            model_name=generator_model,
            temperature=0.7,
            max_tokens=256
        )
        self.generator = AdaptiveGenerator(generator_config, api_key=groq_api_key)
        
        logger.info(f"Initialized Prob-RAG with τ_low={tau_low}, τ_high={tau_high}")
        logger.info(f"Scorer: {scorer_model}, Generator: {generator_model}")
    
    def process_sample(
        self,
        sample: RAGSample,
        generate_answer: bool = True
    ) -> ExperimentResult:
        """Process a single sample through the pipeline."""
        
        # Get combined context from sample
        context = sample.combined_context
        gold_answer = sample.answer
        
        # Step 1: Score sufficiency
        start_time = time.time()
        try:
            score_result = self.scorer.score(sample.question, context)
            sufficiency_score = score_result.score
        except Exception as e:
            logger.warning(f"Scoring error: {e}")
            sufficiency_score = 0.5
        scoring_time = time.time() - start_time
        
        # Step 2: Route
        routing = self.router.route(sufficiency_score)
        
        # Step 3: Generate (if requested)
        generated_answer = ""
        generation_time = 0.0
        
        if generate_answer:
            start_time = time.time()
            try:
                gen_result = self.generator.generate(
                    sample.question,
                    context,
                    routing
                )
                generated_answer = gen_result.answer
            except Exception as e:
                logger.warning(f"Generation error: {e}")
                generated_answer = f"Error: {str(e)}"
            generation_time = time.time() - start_time
        
        # Determine response type
        if routing.state == RouterState.RED:
            response_type = "abstained"
        elif routing.state == RouterState.YELLOW:
            response_type = "hedged"
        else:
            response_type = "answered"
        
        # Simple correctness check (exact match or substring)
        is_correct = self._check_correctness(generated_answer, gold_answer)
        
        return ExperimentResult(
            question_id=sample.id,
            question=sample.question,
            context=context[:500] + "..." if len(context) > 500 else context,
            gold_answer=gold_answer,
            sufficiency_score=sufficiency_score,
            router_state=routing.state.value,
            confidence=routing.confidence,
            generated_answer=generated_answer,
            is_correct=is_correct,
            response_type=response_type,
            scoring_time=scoring_time,
            generation_time=generation_time
        )
    
    def _check_correctness(self, predicted: str, gold: str) -> bool:
        """Check if prediction matches gold answer."""
        if not predicted or not gold:
            return False
        
        pred_lower = predicted.lower()
        gold_lower = gold.lower()
        
        # Exact match
        if gold_lower in pred_lower:
            return True
        
        # Normalized match
        pred_normalized = ''.join(c for c in pred_lower if c.isalnum() or c.isspace())
        gold_normalized = ''.join(c for c in gold_lower if c.isalnum() or c.isspace())
        
        return gold_normalized in pred_normalized
    
    def run_experiment(
        self,
        dataset_name: str,
        num_samples: int = 100,
        generate_answers: bool = True,
        rate_limit_delay: float = 0.5
    ) -> Dict[str, Any]:
        """
        Run experiment on a dataset.
        
        Args:
            dataset_name: Name of dataset ("hotpotqa", "nq", "triviaqa")
            num_samples: Number of samples to evaluate
            generate_answers: Whether to generate answers (slower)
            rate_limit_delay: Delay between API calls (seconds)
        
        Returns:
            Dictionary with results and metrics
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Running experiment on {dataset_name}")
        logger.info(f"Samples: {num_samples}, Generate: {generate_answers}")
        logger.info(f"{'='*60}\n")
        
        # Load dataset
        dataset = self._load_dataset(dataset_name)
        
        # Sample subset
        if num_samples < len(dataset):
            indices = random.sample(range(len(dataset)), num_samples)
            samples = [dataset[i] for i in indices]
        else:
            samples = list(dataset)[:num_samples]
        
        logger.info(f"Processing {len(samples)} samples...")
        
        # Process samples
        results = []
        for sample in tqdm(samples, desc=f"Processing {dataset_name}"):
            try:
                result = self.process_sample(sample, generate_answers)
                results.append(result)
                
                # Rate limiting
                time.sleep(rate_limit_delay)
                
            except Exception as e:
                logger.error(f"Error processing sample {sample.id}: {e}")
                continue
        
        # Compute metrics
        metrics = self._compute_metrics(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"{dataset_name}_{timestamp}.json"
        
        output_data = {
            "experiment": {
                "dataset": dataset_name,
                "num_samples": len(results),
                "timestamp": timestamp,
                "tau_low": self.router.config.tau_low,
                "tau_high": self.router.config.tau_high
            },
            "metrics": metrics,
            "results": [asdict(r) for r in results]
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        
        return {
            "metrics": metrics,
            "results": results,
            "output_file": str(output_file)
        }
    
    def _load_dataset(self, name: str):
        """Load dataset by name."""
        data_dir = Path(__file__).parent.parent / "data"
        
        if name.lower() == "hotpotqa":
            loader = HotPotQALoader(str(data_dir / "hotpotqa"))
            return loader.load("validation")
        elif name.lower() in ["nq", "natural_questions"]:
            loader = NaturalQuestionsLoader(str(data_dir / "natural_questions"))
            return loader.load("validation")
        elif name.lower() == "triviaqa":
            loader = TriviaQALoader(str(data_dir / "triviaqa"))
            return loader.load("validation")
        else:
            raise ValueError(f"Unknown dataset: {name}")
    
    def _compute_metrics(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Compute evaluation metrics from results."""
        if not results:
            return {}
        
        # Basic counts
        total = len(results)
        answered = sum(1 for r in results if r.response_type == "answered")
        hedged = sum(1 for r in results if r.response_type == "hedged")
        abstained = sum(1 for r in results if r.response_type == "abstained")
        
        # Correctness
        correct_total = sum(1 for r in results if r.is_correct)
        correct_answered = sum(1 for r in results if r.response_type == "answered" and r.is_correct)
        correct_hedged = sum(1 for r in results if r.response_type == "hedged" and r.is_correct)
        
        # Score statistics
        scores = [r.sufficiency_score for r in results]
        
        # Traffic light distribution
        red_count = sum(1 for r in results if r.router_state == "abstention")
        yellow_count = sum(1 for r in results if r.router_state == "hedging")
        green_count = sum(1 for r in results if r.router_state == "standard")
        
        # Timing
        avg_scoring_time = np.mean([r.scoring_time for r in results])
        avg_generation_time = np.mean([r.generation_time for r in results if r.generation_time > 0])
        
        metrics = {
            # Coverage and Accuracy
            "total_samples": total,
            "accuracy_overall": correct_total / total if total > 0 else 0,
            "coverage": (answered + hedged) / total if total > 0 else 0,
            
            # By response type
            "answered_count": answered,
            "answered_accuracy": correct_answered / answered if answered > 0 else 0,
            "hedged_count": hedged,
            "hedged_accuracy": correct_hedged / hedged if hedged > 0 else 0,
            "abstained_count": abstained,
            "abstention_rate": abstained / total if total > 0 else 0,
            
            # Traffic light distribution
            "red_rate": red_count / total if total > 0 else 0,
            "yellow_rate": yellow_count / total if total > 0 else 0,
            "green_rate": green_count / total if total > 0 else 0,
            
            # Selective accuracy (accuracy when not abstaining)
            "selective_accuracy": (correct_answered + correct_hedged) / (answered + hedged) if (answered + hedged) > 0 else 0,
            
            # Score statistics
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores)),
            
            # Timing
            "avg_scoring_time_ms": avg_scoring_time * 1000,
            "avg_generation_time_ms": avg_generation_time * 1000 if avg_generation_time > 0 else 0
        }
        
        return metrics


def run_full_evaluation():
    """Run full evaluation across all datasets."""
    
    # Configuration - Get API key from environment
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable not set. Set it with: set GROQ_API_KEY=your_key")
    NUM_SAMPLES = 50  # Per dataset
    GENERATE_ANSWERS = True
    
    # Initialize experiment
    experiment = ProbRAGExperiment(
        groq_api_key=GROQ_API_KEY,
        scorer_model="llama-3.1-8b-instant",
        generator_model="llama-3.1-8b-instant",
        tau_low=0.3,
        tau_high=0.7,
        output_dir="./results"
    )
    
    # Datasets to evaluate
    datasets = ["hotpotqa", "nq", "triviaqa"]
    
    all_results = {}
    
    for dataset_name in datasets:
        try:
            logger.info(f"\n{'#'*60}")
            logger.info(f"EVALUATING: {dataset_name.upper()}")
            logger.info(f"{'#'*60}")
            
            result = experiment.run_experiment(
                dataset_name=dataset_name,
                num_samples=NUM_SAMPLES,
                generate_answers=GENERATE_ANSWERS,
                rate_limit_delay=0.3  # Groq has high rate limits
            )
            
            all_results[dataset_name] = result["metrics"]
            
            # Print summary
            metrics = result["metrics"]
            print(f"\n📊 {dataset_name.upper()} Results:")
            print(f"   Overall Accuracy: {metrics['accuracy_overall']:.2%}")
            print(f"   Selective Accuracy: {metrics['selective_accuracy']:.2%}")
            print(f"   Coverage: {metrics['coverage']:.2%}")
            print(f"   🔴 RED (Abstain): {metrics['red_rate']:.2%}")
            print(f"   🟡 YELLOW (Hedge): {metrics['yellow_rate']:.2%}")
            print(f"   🟢 GREEN (Answer): {metrics['green_rate']:.2%}")
            print(f"   Avg Score: {metrics['score_mean']:.3f} ± {metrics['score_std']:.3f}")
            
        except Exception as e:
            logger.error(f"Error evaluating {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save combined results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_file = Path("./results") / f"combined_results_{timestamp}.json"
    
    with open(combined_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "configuration": {
                "scorer_model": "llama-3.1-8b-instant",
                "generator_model": "llama-3.1-8b-instant",
                "tau_low": 0.3,
                "tau_high": 0.7,
                "num_samples_per_dataset": NUM_SAMPLES
            },
            "results": all_results
        }, f, indent=2)
    
    print(f"\n✅ Combined results saved to {combined_file}")
    
    return all_results


def run_quick_test():
    """Run a quick test with minimal samples."""
    
    # Get API key from environment
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable not set. Set it with: set GROQ_API_KEY=your_key")
    
    experiment = ProbRAGExperiment(
        groq_api_key=GROQ_API_KEY,
        scorer_model="llama-3.1-8b-instant",
        generator_model="llama-3.1-8b-instant",
        tau_low=0.3,
        tau_high=0.7
    )
    
    print("\n🧪 Running quick test on HotPotQA (5 samples)...")
    
    result = experiment.run_experiment(
        dataset_name="hotpotqa",
        num_samples=5,
        generate_answers=True,
        rate_limit_delay=0.3
    )
    
    metrics = result["metrics"]
    
    print("\n📊 Quick Test Results:")
    print(f"   Samples: {metrics['total_samples']}")
    print(f"   Accuracy: {metrics['accuracy_overall']:.2%}")
    print(f"   Score Mean: {metrics['score_mean']:.3f}")
    print(f"   🔴 RED: {metrics['red_rate']:.2%}")
    print(f"   🟡 YELLOW: {metrics['yellow_rate']:.2%}")
    print(f"   🟢 GREEN: {metrics['green_rate']:.2%}")
    
    print("\n📝 Sample Results:")
    for i, r in enumerate(result["results"][:3]):
        print(f"\n--- Sample {i+1} ---")
        print(f"Q: {r.question[:80]}...")
        print(f"Score: {r.sufficiency_score:.3f} → {r.router_state.upper()}")
        print(f"A: {r.generated_answer[:100]}...")
        print(f"Gold: {r.gold_answer}")
        print(f"Correct: {'✅' if r.is_correct else '❌'}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Prob-RAG experiments")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick",
                       help="Run mode: quick (5 samples) or full (50 samples per dataset)")
    parser.add_argument("--dataset", choices=["hotpotqa", "nq", "triviaqa", "all"], 
                       default="hotpotqa", help="Dataset to evaluate")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples")
    
    args = parser.parse_args()
    
    if args.mode == "quick":
        run_quick_test()
    else:
        run_full_evaluation()
