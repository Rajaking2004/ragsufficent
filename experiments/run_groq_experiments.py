"""
Prob-RAG Experiments with Groq API

This script runs experiments on real datasets (HotPotQA, Natural Questions, TriviaQA)
using the Groq API with Llama 3.1 models.
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

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

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
    answer_in_context: bool  # Whether gold answer appears in context
    
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
            # Create a mock score result for error cases
            from prob_rag.modules.scorer import SufficiencyScore
            score_result = SufficiencyScore(score=0.5, response="ERROR")
            sufficiency_score = 0.5
        scoring_time = time.time() - start_time
        
        # Step 2: Route (pass the SufficiencyScore object, not just the float)
        routing = self.router.route(score_result)
        
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
        
        # Check if gold answer is present in the context
        answer_in_context = self._check_answer_in_context(context, gold_answer)
        
        # Correctness logic:
        # - If answered/hedged: correct if gold answer in generated answer
        # - If abstained: correct if answer was NOT in context (rightfully abstained)
        if response_type == "abstained":
            # Abstention is correct if the answer wasn't available in context
            is_correct = not answer_in_context
        else:
            # For answered/hedged, check if generated answer contains gold
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
            answer_in_context=answer_in_context,
            scoring_time=scoring_time,
            generation_time=generation_time
        )
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""
        # Lowercase, remove punctuation, collapse whitespace
        normalized = text.lower()
        normalized = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in normalized)
        normalized = ' '.join(normalized.split())  # Collapse whitespace
        return normalized
    
    def _extract_tokens(self, text: str) -> set:
        """Extract significant tokens from text."""
        normalized = self._normalize_text(text)
        # Filter out very short words (like 'a', 'an', 'the')
        tokens = {w for w in normalized.split() if len(w) > 2}
        return tokens
    
    def _fuzzy_match(self, candidate: str, reference: str, threshold: float = 0.7) -> bool:
        """
        Fuzzy matching with multiple strategies:
        1. Direct substring match
        2. Normalized substring match
        3. Token overlap (for partial name matches)
        4. Key token matching (for names, dates, numbers)
        """
        if not candidate or not reference:
            return False
        
        cand_lower = candidate.lower()
        ref_lower = reference.lower()
        
        # Strategy 1: Direct substring
        if ref_lower in cand_lower:
            return True
        
        # Strategy 2: Normalized substring
        cand_norm = self._normalize_text(candidate)
        ref_norm = self._normalize_text(reference)
        
        if ref_norm in cand_norm:
            return True
        
        # Strategy 3: Token overlap for multi-word answers
        ref_tokens = self._extract_tokens(reference)
        cand_tokens = self._extract_tokens(candidate)
        
        if ref_tokens and cand_tokens:
            # For names: if all significant words from gold appear in candidate
            overlap = ref_tokens.intersection(cand_tokens)
            if len(ref_tokens) > 0:
                overlap_ratio = len(overlap) / len(ref_tokens)
                if overlap_ratio >= threshold:
                    return True
        
        # Strategy 4: Check if key parts match (first/last name, numbers)
        ref_parts = ref_norm.split()
        cand_parts = cand_norm.split()
        
        # For single-word answers, check if it appears as a standalone word
        if len(ref_parts) == 1 and ref_parts[0] in cand_parts:
            return True
        
        # For multi-word: check if first AND last significant word match
        if len(ref_parts) >= 2:
            first_word = ref_parts[0]
            last_word = ref_parts[-1]
            # Both first and last word should appear
            if first_word in cand_parts and last_word in cand_parts:
                return True
        
        # Strategy 5: Number extraction (for dates, quantities)
        import re
        ref_numbers = set(re.findall(r'\d+', reference))
        cand_numbers = set(re.findall(r'\d+', candidate))
        
        # If gold has numbers, they should all appear in candidate
        if ref_numbers and ref_numbers.issubset(cand_numbers):
            # Also need at least one non-number token match for context
            if len(ref_tokens - {'the', 'and', 'of'}) > 0:
                text_overlap = ref_tokens.intersection(cand_tokens)
                if text_overlap:
                    return True
            else:
                # Pure number answer (like years)
                return True
        
        return False
    
    def _check_correctness(self, predicted: str, gold: str) -> bool:
        """Check if prediction matches gold answer using fuzzy matching."""
        return self._fuzzy_match(predicted, gold, threshold=0.7)
    
    def _check_answer_in_context(self, context: str, gold: str) -> bool:
        """Check if the gold answer appears in the context using fuzzy matching."""
        return self._fuzzy_match(context, gold, threshold=0.8)  # Slightly stricter for context

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
        
        # Correctness (now includes correct abstentions!)
        correct_total = sum(1 for r in results if r.is_correct)
        correct_answered = sum(1 for r in results if r.response_type == "answered" and r.is_correct)
        correct_hedged = sum(1 for r in results if r.response_type == "hedged" and r.is_correct)
        correct_abstained = sum(1 for r in results if r.response_type == "abstained" and r.is_correct)
        
        # Abstention analysis
        abstained_with_answer = sum(1 for r in results if r.response_type == "abstained" and r.answer_in_context)
        abstained_without_answer = sum(1 for r in results if r.response_type == "abstained" and not r.answer_in_context)
        
        # Answer availability
        samples_with_answer = sum(1 for r in results if r.answer_in_context)
        samples_without_answer = total - samples_with_answer
        
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
            
            # Abstention quality metrics (NEW!)
            "correct_abstentions": correct_abstained,  # Abstained when answer NOT in context
            "wrong_abstentions": abstained_with_answer,  # Abstained when answer WAS in context
            "abstention_precision": correct_abstained / abstained if abstained > 0 else 0,
            
            # Context analysis
            "samples_with_answer_in_context": samples_with_answer,
            "samples_without_answer_in_context": samples_without_answer,
            
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
                rate_limit_delay=16.0  # 1 request per 16 seconds
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


def run_quick_test(num_samples: int = 50):
    """Run a quick test with specified samples using Groq Llama."""
    
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
    
    est_time = num_samples * 16 / 60
    print(f"\n🧪 Running experiment on HotPotQA ({num_samples} samples) with Groq Llama 3.1...")
    print(f"⏱️  Rate limit: 1 request per 16 seconds (~{est_time:.1f} minutes total)")
    
    result = experiment.run_experiment(
        dataset_name="hotpotqa",
        num_samples=num_samples,
        generate_answers=True,
        rate_limit_delay=16.0  # 1 request per 16 seconds
    )
    
    metrics = result["metrics"]
    
    print("\n📊 Quick Test Results:")
    print(f"   Samples: {metrics['total_samples']}")
    print(f"   Overall Accuracy: {metrics['accuracy_overall']:.2%}")
    print(f"   Selective Accuracy: {metrics['selective_accuracy']:.2%}")
    print(f"   Score Mean: {metrics['score_mean']:.3f}")
    print(f"   🔴 RED: {metrics['red_rate']:.2%}")
    print(f"   🟡 YELLOW: {metrics['yellow_rate']:.2%}")
    print(f"   🟢 GREEN: {metrics['green_rate']:.2%}")
    print(f"\n📈 Abstention Analysis:")
    print(f"   Correct Abstentions: {metrics['correct_abstentions']} (answer NOT in context)")
    print(f"   Wrong Abstentions: {metrics['wrong_abstentions']} (answer WAS in context)")
    print(f"   Abstention Precision: {metrics['abstention_precision']:.2%}")
    print(f"   Samples with answer in context: {metrics['samples_with_answer_in_context']}")
    print(f"   Samples without answer in context: {metrics['samples_without_answer_in_context']}")
    
    print("\n📝 All Sample Results:")
    for i, r in enumerate(result["results"]):
        print(f"\n--- Sample {i+1}/{len(result['results'])} ---")
        print(f"Q: {r.question}")
        print(f"Score: {r.sufficiency_score:.3f} → {r.router_state.upper()}")
        print(f"Answer in context: {'✅ Yes' if r.answer_in_context else '❌ No'}")
        print(f"A: {r.generated_answer}")
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
        run_quick_test(num_samples=args.samples)
    else:
        run_full_evaluation()
