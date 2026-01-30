"""
Prob-RAG Experiments with Google Gemini API

Uses Gemini 2.0 Flash (free tier with higher rate limits):
- 15 RPM (requests per minute)
- 1M TPM (tokens per minute)
- 1500 RPD (requests per day)
"""

import os
import sys
import json
import time
import random
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import re

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import google.generativeai as genai
from tqdm import tqdm

from prob_rag.config import RouterState
from prob_rag.data.datasets import HotPotQALoader, RAGSample

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
    sufficiency_score: float
    router_state: str
    generated_answer: str
    is_correct: bool
    response_type: str
    answer_in_context: bool


class GeminiProbRAG:
    """Prob-RAG using Gemini API."""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.0-flash-lite",  # Lite version - may have separate quota
        tau_low: float = 0.3,
        tau_high: float = 0.7
    ):
        """Initialize with Gemini."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.model_name = model_name
        
        logger.info(f"Initialized Gemini Prob-RAG with {model_name}")
        logger.info(f"Thresholds: τ_low={tau_low}, τ_high={tau_high}")
    
    def _call_gemini(self, prompt: str, max_tokens: int = 256) -> str:
        """Call Gemini API with retry."""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.1
                )
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
    def score_sufficiency(self, question: str, context: str) -> float:
        """Score context sufficiency."""
        prompt = f"""### QUESTION
{question}

### CONTEXT
{context}

### TASK
Does the context contain information that answers the question?

Think briefly:
1. What does the question ask for?
2. Is that information in the context?

If the answer or relevant information is in the context, say Yes.
If the context has nothing relevant, say No.

Answer (Yes/No):"""
        
        response = self._call_gemini(prompt, max_tokens=50)
        response_lower = response.lower()
        
        # Extract Yes/No from response
        if "yes" in response_lower:
            return 0.85
        elif "no" in response_lower:
            return 0.15
        else:
            return 0.5  # Uncertain
    
    def route(self, score: float) -> tuple:
        """Route based on score."""
        if score < self.tau_low:
            return RouterState.RED, "abstained"
        elif score < self.tau_high:
            return RouterState.YELLOW, "hedged"
        else:
            return RouterState.GREEN, "answered"
    
    def generate_answer(self, question: str, context: str, state: RouterState) -> str:
        """Generate answer based on state."""
        if state == RouterState.RED:
            return "I cannot find the answer in the provided documents."
        
        if state == RouterState.YELLOW:
            system = "Answer using ONLY the documents. Be brief - just the key words/phrase. Use hedging like 'possibly' or 'likely'. One line max."
        else:
            system = "Answer using ONLY the documents. Give ONLY the answer - just the name, date, number, or phrase. No explanation. No sentences. Just the answer itself."
        
        prompt = f"""{system}

### QUESTION
{question}

### CONTEXT
{context}

### ANSWER"""
        
        return self._call_gemini(prompt, max_tokens=100)
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""
        normalized = text.lower()
        normalized = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in normalized)
        normalized = ' '.join(normalized.split())
        return normalized
    
    def _extract_tokens(self, text: str) -> set:
        """Extract significant tokens from text."""
        normalized = self._normalize_text(text)
        tokens = {w for w in normalized.split() if len(w) > 2}
        return tokens
    
    def _fuzzy_match(self, candidate: str, reference: str, threshold: float = 0.7) -> bool:
        """Fuzzy matching with multiple strategies."""
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
        
        # Strategy 3: Token overlap
        ref_tokens = self._extract_tokens(reference)
        cand_tokens = self._extract_tokens(candidate)
        
        if ref_tokens and cand_tokens:
            overlap = ref_tokens.intersection(cand_tokens)
            if len(ref_tokens) > 0:
                overlap_ratio = len(overlap) / len(ref_tokens)
                if overlap_ratio >= threshold:
                    return True
        
        # Strategy 4: Check key parts
        ref_parts = ref_norm.split()
        cand_parts = cand_norm.split()
        
        if len(ref_parts) == 1 and ref_parts[0] in cand_parts:
            return True
        
        if len(ref_parts) >= 2:
            first_word = ref_parts[0]
            last_word = ref_parts[-1]
            if first_word in cand_parts and last_word in cand_parts:
                return True
        
        # Strategy 5: Number extraction
        ref_numbers = set(re.findall(r'\d+', reference))
        cand_numbers = set(re.findall(r'\d+', candidate))
        
        if ref_numbers and ref_numbers.issubset(cand_numbers):
            text_overlap = ref_tokens.intersection(cand_tokens)
            if text_overlap or len(ref_tokens - {'the', 'and', 'of'}) == 0:
                return True
        
        return False
    
    def check_answer_in_context(self, context: str, gold: str) -> bool:
        """Check if gold answer is in context."""
        return self._fuzzy_match(context, gold, threshold=0.8)
    
    def check_correctness(self, predicted: str, gold: str) -> bool:
        """Check if prediction matches gold."""
        return self._fuzzy_match(predicted, gold, threshold=0.7)
    
    def process_sample(self, sample: RAGSample) -> ExperimentResult:
        """Process a single sample."""
        context = sample.combined_context
        gold = sample.answer
        
        # Score
        score = self.score_sufficiency(sample.question, context)
        
        # Route
        state, response_type = self.route(score)
        
        # Generate
        answer = self.generate_answer(sample.question, context, state)
        
        # Evaluate
        answer_in_context = self.check_answer_in_context(context, gold)
        
        if response_type == "abstained":
            is_correct = not answer_in_context
        else:
            is_correct = self.check_correctness(answer, gold)
        
        return ExperimentResult(
            question_id=sample.id,
            question=sample.question,
            context=context[:500] + "..." if len(context) > 500 else context,
            gold_answer=gold,
            sufficiency_score=score,
            router_state=state.value,
            generated_answer=answer,
            is_correct=is_correct,
            response_type=response_type,
            answer_in_context=answer_in_context
        )


def run_experiment(api_key: str, num_samples: int = 10):
    """Run the experiment."""
    print("=" * 60)
    print("PROB-RAG EXPERIMENT (Gemini 2.0 Flash Lite)")
    print("=" * 60)
    
    # Initialize
    prob_rag = GeminiProbRAG(api_key=api_key, model_name="gemini-2.0-flash-lite")
    
    # Load dataset
    print("\n📂 Loading HotPotQA dataset...")
    loader = HotPotQALoader()
    dataset = loader.load(split="validation")
    
    # Sample
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    samples = [dataset[i] for i in indices]
    
    print(f"📊 Processing {len(samples)} samples...")
    print(f"⏱️  Rate limit: ~4 seconds between samples (15 RPM)")
    
    results = []
    for i, sample in enumerate(tqdm(samples, desc="Processing")):
        try:
            result = prob_rag.process_sample(sample)
            results.append(result)
        except Exception as e:
            logger.error(f"Error on sample {i}: {e}")
            continue
        
        # Rate limit: 15 RPM = 4 seconds per request, but we make 2 calls per sample
        if i < len(samples) - 1:
            time.sleep(8)  # 8 seconds between samples (2 calls × 4s)
    
    # Compute metrics
    total = len(results)
    correct = sum(1 for r in results if r.is_correct)
    answered = [r for r in results if r.response_type != "abstained"]
    correct_answered = sum(1 for r in answered if r.is_correct)
    
    abstentions = [r for r in results if r.response_type == "abstained"]
    correct_abstentions = sum(1 for r in abstentions if r.is_correct)
    
    red = sum(1 for r in results if r.router_state == "red")
    yellow = sum(1 for r in results if r.router_state == "yellow")
    green = sum(1 for r in results if r.router_state == "green")
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 RESULTS")
    print("=" * 60)
    print(f"   Samples: {total}")
    print(f"   Overall Accuracy: {correct/total:.2%}")
    print(f"   Selective Accuracy: {correct_answered/len(answered):.2%}" if answered else "   Selective Accuracy: N/A")
    print(f"   🔴 RED: {red/total:.2%}")
    print(f"   🟡 YELLOW: {yellow/total:.2%}")
    print(f"   🟢 GREEN: {green/total:.2%}")
    
    print(f"\n📈 Abstention Analysis:")
    print(f"   Correct Abstentions: {correct_abstentions}/{len(abstentions)}")
    if abstentions:
        print(f"   Abstention Precision: {correct_abstentions/len(abstentions):.2%}")
    
    print("\n📝 Sample Results:")
    for i, r in enumerate(results):
        state_emoji = {"red": "🔴", "yellow": "🟡", "green": "🟢"}[r.router_state]
        correct_emoji = "✅" if r.is_correct else "❌"
        context_emoji = "✅" if r.answer_in_context else "❌"
        
        print(f"\n--- Sample {i+1}/{total} ---")
        print(f"Q: {r.question}")
        print(f"Score: {r.sufficiency_score:.3f} → {state_emoji} {r.router_state.upper()}")
        print(f"Answer in context: {context_emoji}")
        print(f"A: {r.generated_answer}")
        print(f"Gold: {r.gold_answer}")
        print(f"Correct: {correct_emoji}")
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"gemini_hotpotqa_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            "model": "gemini-2.0-flash",
            "dataset": "hotpotqa",
            "num_samples": total,
            "metrics": {
                "overall_accuracy": correct/total,
                "selective_accuracy": correct_answered/len(answered) if answered else 0,
                "red_rate": red/total,
                "yellow_rate": yellow/total,
                "green_rate": green/total,
                "abstention_precision": correct_abstentions/len(abstentions) if abstentions else 0
            },
            "results": [asdict(r) for r in results]
        }, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Prob-RAG with Gemini")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples")
    parser.add_argument("--api-key", type=str, help="Gemini API key")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Please provide API key via --api-key or GOOGLE_API_KEY env var")
    
    run_experiment(api_key=api_key, num_samples=args.samples)
