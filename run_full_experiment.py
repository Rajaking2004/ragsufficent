#!/usr/bin/env python
"""
Full Prob-RAG Experiment with Gemini API
Runs experiments on HotPotQA, Natural Questions, and TriviaQA datasets.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Any

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prob_rag import ProbRAGConfig
from prob_rag.config import ScorerConfig, GeneratorConfig, RouterConfig
from prob_rag.data.datasets import load_dataset_samples, RAGSample
from prob_rag.modules.scorer import SufficiencyScore
from prob_rag.modules.router import TrafficLightRouter, RoutingDecision
from prob_rag.modules.generator import GenerationResult
from prob_rag.evaluation.metrics import compute_exact_match, compute_f1, normalize_answer

# Gemini API setup
import google.generativeai as genai

# Configuration
GEMINI_API_KEY = "AIzaSyAZ1BGJKGY_7v8dfeVCq2FTRbznX7lxfVc"
MODEL_NAME = "models/gemini-2.0-flash-lite"  # Least powerful model
SAMPLES_PER_DATASET = 50  # Reasonable sample size
RATE_LIMIT_DELAY = 1.5  # Seconds between API calls to avoid rate limiting


class GeminiProbRAG:
    """Prob-RAG implementation using Gemini API directly."""
    
    def __init__(self, api_key: str, model_name: str, tau_low: float = 0.3, tau_high: float = 0.7):
        self.api_key = api_key
        self.model_name = model_name
        self.tau_low = tau_low
        self.tau_high = tau_high
        
        # Initialize Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # Router for traffic light decisions
        router_config = RouterConfig(tau_low=tau_low, tau_high=tau_high)
        self.router = TrafficLightRouter(router_config)
        
        # Statistics
        self.stats = {
            'total': 0,
            'correct': 0,
            'abstentions': 0,
            'hedged': 0,
            'standard': 0,
            'api_errors': 0
        }
    
    def _call_gemini(self, prompt: str, max_retries: int = 3) -> str:
        """Call Gemini API with retry logic."""
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.0,
                        max_output_tokens=100
                    )
                )
                return response.text.strip() if response.text else ""
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    wait_time = (attempt + 1) * 5
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"API error: {e}")
                    if attempt == max_retries - 1:
                        self.stats['api_errors'] += 1
                        return ""
                    time.sleep(2)
        return ""
    
    def score_sufficiency(self, question: str, context: str) -> float:
        """Score context sufficiency using Gemini."""
        prompt = f"""You are an expert evaluator. Determine if the context contains sufficient information to answer the question.

QUESTION: {question}

CONTEXT: {context[:3000]}

Can this question be definitively answered using ONLY the provided context?
Answer with exactly one word: Yes or No"""

        response = self._call_gemini(prompt)
        response_lower = response.lower().strip()
        
        # Convert to score
        if response_lower.startswith("yes"):
            return 0.85
        elif response_lower.startswith("no"):
            return 0.15
        else:
            return 0.5
    
    def generate_answer(self, question: str, context: str, routing_state: str) -> str:
        """Generate answer based on routing state."""
        if routing_state == "abstention":
            return "I cannot answer this question based on the provided context. The information is insufficient."
        
        elif routing_state == "hedging":
            prompt = f"""Based on the context below, try to answer the question. Express uncertainty if needed.

CONTEXT: {context[:3000]}

QUESTION: {question}

Answer (be concise, express uncertainty if information is incomplete):"""
        
        else:  # standard
            prompt = f"""Answer the following question using ONLY the provided context.

CONTEXT: {context[:3000]}

QUESTION: {question}

Answer (be concise and direct):"""
        
        answer = self._call_gemini(prompt)
        return answer if answer else "Unable to generate answer."
    
    def process_sample(self, sample: RAGSample) -> Dict[str, Any]:
        """Process a single sample through the pipeline."""
        question = sample.question
        context = sample.combined_context
        ground_truth = sample.answer
        
        # Step 1: Score sufficiency
        time.sleep(RATE_LIMIT_DELAY)
        score = self.score_sufficiency(question, context)
        
        # Step 2: Route
        if score < self.tau_low:
            routing_state = "abstention"
            self.stats['abstentions'] += 1
        elif score < self.tau_high:
            routing_state = "hedging"
            self.stats['hedged'] += 1
        else:
            routing_state = "standard"
            self.stats['standard'] += 1
        
        # Step 3: Generate
        time.sleep(RATE_LIMIT_DELAY)
        answer = self.generate_answer(question, context, routing_state)
        
        # Step 4: Evaluate
        is_correct = compute_exact_match(answer, ground_truth, sample.answer_aliases)
        f1 = compute_f1(answer, ground_truth)
        
        self.stats['total'] += 1
        if is_correct:
            self.stats['correct'] += 1
        
        return {
            'question': question,
            'ground_truth': ground_truth,
            'predicted': answer,
            'score': score,
            'routing': routing_state,
            'is_correct': is_correct,
            'f1': f1
        }
    
    def get_metrics(self) -> Dict[str, float]:
        """Calculate final metrics."""
        total = self.stats['total']
        if total == 0:
            return {}
        
        answered = total - self.stats['abstentions']
        
        return {
            'accuracy': self.stats['correct'] / total if total > 0 else 0,
            'coverage': answered / total if total > 0 else 0,
            'selective_accuracy': self.stats['correct'] / answered if answered > 0 else 0,
            'abstention_rate': self.stats['abstentions'] / total if total > 0 else 0,
            'hedging_rate': self.stats['hedged'] / total if total > 0 else 0,
            'standard_rate': self.stats['standard'] / total if total > 0 else 0,
            'api_error_rate': self.stats['api_errors'] / total if total > 0 else 0
        }
    
    def reset_stats(self):
        """Reset statistics for new experiment."""
        self.stats = {
            'total': 0,
            'correct': 0,
            'abstentions': 0,
            'hedged': 0,
            'standard': 0,
            'api_errors': 0
        }


class BaselineRAG:
    """Baseline RAG without traffic light routing (always answers)."""
    
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.stats = {'total': 0, 'correct': 0, 'api_errors': 0}
    
    def _call_gemini(self, prompt: str, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.0,
                        max_output_tokens=100
                    )
                )
                return response.text.strip() if response.text else ""
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    wait_time = (attempt + 1) * 5
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    if attempt == max_retries - 1:
                        self.stats['api_errors'] += 1
                        return ""
                    time.sleep(2)
        return ""
    
    def process_sample(self, sample: RAGSample) -> Dict[str, Any]:
        """Process sample (always generates answer, no routing)."""
        question = sample.question
        context = sample.combined_context
        ground_truth = sample.answer
        
        prompt = f"""Answer the following question using the provided context.

CONTEXT: {context[:3000]}

QUESTION: {question}

Answer (be concise):"""
        
        time.sleep(RATE_LIMIT_DELAY)
        answer = self._call_gemini(prompt)
        if not answer:
            answer = "Unable to generate answer."
        
        is_correct = compute_exact_match(answer, ground_truth, sample.answer_aliases)
        f1 = compute_f1(answer, ground_truth)
        
        self.stats['total'] += 1
        if is_correct:
            self.stats['correct'] += 1
        
        return {
            'question': question,
            'ground_truth': ground_truth,
            'predicted': answer,
            'is_correct': is_correct,
            'f1': f1
        }
    
    def get_metrics(self) -> Dict[str, float]:
        total = self.stats['total']
        return {
            'accuracy': self.stats['correct'] / total if total > 0 else 0,
            'coverage': 1.0,  # Always answers
            'selective_accuracy': self.stats['correct'] / total if total > 0 else 0
        }
    
    def reset_stats(self):
        self.stats = {'total': 0, 'correct': 0, 'api_errors': 0}


def run_experiment(dataset_name: str, num_samples: int, prob_rag: GeminiProbRAG, baseline: BaselineRAG) -> Dict:
    """Run experiment on a single dataset."""
    logger.info(f"\n{'='*70}")
    logger.info(f"EXPERIMENT: {dataset_name.upper()}")
    logger.info(f"{'='*70}")
    
    # Load samples
    logger.info(f"Loading {num_samples} samples from {dataset_name}...")
    samples = load_dataset_samples(dataset_name, split='validation', num_samples=num_samples)
    logger.info(f"Loaded {len(samples)} samples")
    
    # Reset stats
    prob_rag.reset_stats()
    baseline.reset_stats()
    
    # Results storage
    prob_rag_results = []
    baseline_results = []
    
    # Process with Prob-RAG
    logger.info(f"\n🔵 Running Prob-RAG pipeline...")
    for i, sample in enumerate(samples):
        if (i + 1) % 10 == 0:
            logger.info(f"  Prob-RAG progress: {i+1}/{len(samples)}")
        result = prob_rag.process_sample(sample)
        prob_rag_results.append(result)
    
    prob_rag_metrics = prob_rag.get_metrics()
    
    # Process with Baseline
    logger.info(f"\n🔴 Running Baseline pipeline...")
    for i, sample in enumerate(samples):
        if (i + 1) % 10 == 0:
            logger.info(f"  Baseline progress: {i+1}/{len(samples)}")
        result = baseline.process_sample(sample)
        baseline_results.append(result)
    
    baseline_metrics = baseline.get_metrics()
    
    return {
        'dataset': dataset_name,
        'num_samples': len(samples),
        'prob_rag': {
            'metrics': prob_rag_metrics,
            'results': prob_rag_results
        },
        'baseline': {
            'metrics': baseline_metrics,
            'results': baseline_results
        }
    }


def print_results(all_results: List[Dict]):
    """Print formatted results."""
    print("\n" + "="*80)
    print("FINAL EXPERIMENT RESULTS - PROB-RAG VS BASELINE")
    print("="*80)
    print(f"Model: {MODEL_NAME}")
    print(f"Thresholds: τ_low=0.3, τ_high=0.7")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    for result in all_results:
        dataset = result['dataset']
        n = result['num_samples']
        pr = result['prob_rag']['metrics']
        bl = result['baseline']['metrics']
        
        print(f"\n📊 {dataset.upper()} ({n} samples)")
        print("-"*60)
        print(f"{'Metric':<25} {'Prob-RAG':>12} {'Baseline':>12} {'Δ':>10}")
        print("-"*60)
        
        acc_diff = (pr['accuracy'] - bl['accuracy']) * 100
        print(f"{'Accuracy':<25} {pr['accuracy']*100:>11.1f}% {bl['accuracy']*100:>11.1f}% {acc_diff:>+9.1f}%")
        
        cov_diff = (pr['coverage'] - bl['coverage']) * 100
        print(f"{'Coverage':<25} {pr['coverage']*100:>11.1f}% {bl['coverage']*100:>11.1f}% {cov_diff:>+9.1f}%")
        
        sel_diff = (pr['selective_accuracy'] - bl['selective_accuracy']) * 100
        print(f"{'Selective Accuracy':<25} {pr['selective_accuracy']*100:>11.1f}% {bl['selective_accuracy']*100:>11.1f}% {sel_diff:>+9.1f}%")
        
        print(f"\n  Routing Distribution (Prob-RAG):")
        print(f"    🔴 Abstention: {pr['abstention_rate']*100:.1f}%")
        print(f"    🟡 Hedging:    {pr['hedging_rate']*100:.1f}%")
        print(f"    🟢 Standard:   {pr['standard_rate']*100:.1f}%")
    
    # Summary across all datasets
    print("\n" + "="*80)
    print("AGGREGATE SUMMARY")
    print("="*80)
    
    total_pr_correct = sum(r['prob_rag']['metrics']['accuracy'] * r['num_samples'] for r in all_results)
    total_bl_correct = sum(r['baseline']['metrics']['accuracy'] * r['num_samples'] for r in all_results)
    total_samples = sum(r['num_samples'] for r in all_results)
    
    avg_pr_acc = total_pr_correct / total_samples if total_samples > 0 else 0
    avg_bl_acc = total_bl_correct / total_samples if total_samples > 0 else 0
    
    print(f"Total Samples: {total_samples}")
    print(f"Average Prob-RAG Accuracy: {avg_pr_acc*100:.1f}%")
    print(f"Average Baseline Accuracy: {avg_bl_acc*100:.1f}%")
    print(f"Improvement: {(avg_pr_acc - avg_bl_acc)*100:+.1f}%")
    print("="*80)


def main():
    print("="*80)
    print("PROB-RAG FULL EXPERIMENT")
    print("Probabilistic Sufficient Context RAG with Traffic Light Routing")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_NAME} (Gemini Lite - least powerful)")
    print(f"  Samples per dataset: {SAMPLES_PER_DATASET}")
    print(f"  Datasets: HotPotQA, Natural Questions, TriviaQA")
    print(f"  Rate limit delay: {RATE_LIMIT_DELAY}s between calls")
    print("="*80)
    
    # Initialize models
    logger.info("Initializing Gemini models...")
    prob_rag = GeminiProbRAG(
        api_key=GEMINI_API_KEY,
        model_name=MODEL_NAME,
        tau_low=0.3,
        tau_high=0.7
    )
    baseline = BaselineRAG(
        api_key=GEMINI_API_KEY,
        model_name=MODEL_NAME
    )
    
    # Datasets to test
    datasets = ['hotpotqa', 'natural_questions', 'triviaqa']
    
    all_results = []
    
    for dataset in datasets:
        try:
            result = run_experiment(dataset, SAMPLES_PER_DATASET, prob_rag, baseline)
            all_results.append(result)
            
            # Save intermediate results
            with open(f'results_{dataset}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error on {dataset}: {e}")
            continue
    
    # Print final results
    if all_results:
        print_results(all_results)
        
        # Save all results
        output_file = f'full_experiment_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"\nResults saved to {output_file}")
    else:
        logger.error("No results generated!")
    
    return all_results


if __name__ == "__main__":
    results = main()
