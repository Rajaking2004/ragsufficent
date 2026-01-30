#!/usr/bin/env python3
"""
Prob-RAG v2: Enhanced with Paper Insights
==========================================

This version combines our novel contributions with insights from Joren et al. ICLR 2025:

OUR NOVEL CONTRIBUTIONS:
1. Continuous scoring [0,1] instead of binary
2. 3-state traffic light routing (RED/YELLOW/GREEN)
3. Adaptive prompts per routing state
4. Smart accuracy (correct abstentions count)

INTEGRATED FROM PAPER:
1. CoT-style autorater prompt for better calibration
2. LLMEval for semantic answer checking
3. Proper handling of insufficient context
4. Detailed logging of sufficient vs insufficient context

This is the BEST version combining both approaches.
"""

import os
import json
import time
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# ============================================================
# ROUTING STATES (Our Novel 3-State System)
# ============================================================

class RouterState(Enum):
    RED = "RED"         # Abstain - insufficient context
    YELLOW = "YELLOW"   # Hedge - uncertain
    GREEN = "GREEN"     # Standard - confident answer


# ============================================================
# ENHANCED PROMPTS (Combining Paper + Our Approach)
# ============================================================

# Paper's CoT autorater prompt adapted for CONTINUOUS scoring
ENHANCED_AUTORATER_PROMPT = """You are an expert evaluator assessing whether a CONTEXT provides sufficient information to answer a QUESTION.

### QUESTION
{question}

### CONTEXT
{context}

### TASK
Evaluate if the context is sufficient to answer the question.

Step 1: What specific information does the question ask for?
Step 2: Is that exact information present in the context?
Step 3: Are there any ambiguities or missing pieces?

Based on your analysis, provide:
1. A confidence score from 0.0 to 1.0 where:
   - 0.0-0.3 = Clearly INSUFFICIENT (answer not in context)
   - 0.3-0.7 = UNCERTAIN (partial info, ambiguous)
   - 0.7-1.0 = SUFFICIENT (answer is clearly present)

2. Your assessment in JSON format.

### ANALYSIS
(Provide your step-by-step reasoning here)

### JSON
{{"score": <0.0-1.0>, "sufficient": <true/false>, "reason": "<brief reason>"}}
"""

# Our adaptive prompts per routing state
GENERATION_PROMPTS = {
    RouterState.RED: """The provided documents do NOT contain sufficient information to answer this question.
Reply with: "I cannot find the answer in the provided documents."
Do not guess or make up information.

Question: {question}
Context: {context}

Response:""",

    RouterState.YELLOW: """The provided documents contain PARTIAL information. Answer cautiously with hedging language.
Use phrases like "possibly", "likely", "based on available information".
State limitations clearly.

Question: {question}
Context: {context}

Give ONLY a brief hedged answer (1-2 sentences max):""",

    RouterState.GREEN: """Answer the question using ONLY the provided documents.
Give ONLY the answer - just the name, date, number, or key phrase.
No explanation, no sentences, just the answer itself.

Question: {question}
Context: {context}

Answer:"""
}

# Paper's LLMEval prompt for semantic checking
LLMEVAL_PROMPT = """Compare the predicted answer with ground truth. Output one of:
- "correct": Answers match (semantically equivalent)
- "partial": Partially correct
- "incorrect": Wrong answer  
- "abstain": Refused to answer

Question: {question}
Predicted: {predicted}
Ground Truth: {ground_truth}

Decision:"""


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class EnhancedResult:
    """Result combining paper's methodology with our 3-state approach"""
    sample_id: str
    question: str
    gold_answer: str
    
    # Enhanced scoring (continuous)
    sufficiency_score: float  # 0.0 to 1.0
    answer_in_context: bool   # Ground truth availability
    
    # 3-state routing
    routing_state: RouterState
    
    # Generation
    generated_answer: str
    
    # Evaluation (Paper's LLMEval style)
    eval_decision: str
    is_correct: bool
    
    # Smart accuracy
    correct_abstention: bool  # True if abstained AND answer wasn't available


# ============================================================
# GROQ CLIENT
# ============================================================

class GroqClient:
    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant"):
        from groq import Groq
        self.client = Groq(api_key=api_key)
        self.model = model
        self.rate_limit_delay = 16
        
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()


# ============================================================
# ENHANCED CONTINUOUS SCORER
# ============================================================

class EnhancedSufficiencyScorer:
    """
    Combines paper's CoT reasoning with continuous scoring.
    Returns score in [0, 1] instead of binary.
    """
    
    def __init__(self, client: GroqClient):
        self.client = client
        
    def score(self, question: str, context: str) -> Tuple[float, str]:
        """Returns: (score: float, explanation: str)"""
        prompt = ENHANCED_AUTORATER_PROMPT.format(
            question=question,
            context=context[:6000]
        )
        
        response = self.client.generate(prompt, max_tokens=1024)
        
        # Parse score from JSON
        score = 0.5  # default uncertain
        reason = response
        
        # Try to extract JSON
        json_match = re.search(r'\{[^}]*"score"[^}]*\}', response, re.IGNORECASE)
        if json_match:
            try:
                result = json.loads(json_match.group())
                score = float(result.get("score", 0.5))
                reason = result.get("reason", response)
            except:
                pass
        
        # Fallback: look for explicit score mentions
        if score == 0.5:
            score_match = re.search(r'score[:\s]*(\d+\.?\d*)', response, re.IGNORECASE)
            if score_match:
                score = float(score_match.group(1))
                if score > 1:
                    score = score / 100  # Handle percentage
                    
        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))
        
        return score, reason


# ============================================================
# 3-STATE TRAFFIC LIGHT ROUTER (Our Novel Contribution)
# ============================================================

class TrafficLightRouter:
    """
    Our novel 3-state routing:
    - RED: Insufficient context → Abstain
    - YELLOW: Uncertain → Hedge
    - GREEN: Sufficient → Answer confidently
    """
    
    def __init__(self, tau_low: float = 0.3, tau_high: float = 0.7):
        self.tau_low = tau_low
        self.tau_high = tau_high
        
    def route(self, score: float) -> RouterState:
        if score < self.tau_low:
            return RouterState.RED
        elif score < self.tau_high:
            return RouterState.YELLOW
        else:
            return RouterState.GREEN


# ============================================================
# ADAPTIVE GENERATOR
# ============================================================

class AdaptiveGenerator:
    """Generates different responses based on routing state"""
    
    def __init__(self, client: GroqClient):
        self.client = client
        
    def generate(self, question: str, context: str, state: RouterState) -> str:
        prompt = GENERATION_PROMPTS[state].format(
            question=question,
            context=context[:4000]
        )
        return self.client.generate(prompt, max_tokens=256)


# ============================================================
# ENHANCED EVALUATOR
# ============================================================

class EnhancedEvaluator:
    """Combines LLMEval + fuzzy matching + smart accuracy"""
    
    def __init__(self, client: GroqClient):
        self.client = client
        
    def check_answer_in_context(self, answer: str, context: str) -> bool:
        """Check if gold answer is present in context (for smart accuracy)"""
        answer_lower = answer.lower().strip()
        context_lower = context.lower()
        
        # Direct match
        if answer_lower in context_lower:
            return True
            
        # Token overlap (50%+ tokens match)
        answer_tokens = set(answer_lower.split())
        context_tokens = set(context_lower.split())
        if answer_tokens and len(answer_tokens & context_tokens) / len(answer_tokens) >= 0.5:
            return True
            
        return False
    
    def evaluate(self, question: str, predicted: str, gold: str) -> Tuple[str, bool]:
        """Returns: (decision: str, is_correct: bool)"""
        
        # Check for abstention
        abstain_phrases = ["i cannot", "i don't know", "cannot find", "not enough"]
        if any(p in predicted.lower() for p in abstain_phrases):
            return "abstain", False
            
        # Fuzzy matching first
        pred_normalized = self._normalize(predicted)
        gold_normalized = self._normalize(gold)
        
        if gold_normalized in pred_normalized or pred_normalized in gold_normalized:
            return "correct", True
            
        # Token overlap
        pred_tokens = set(pred_normalized.split())
        gold_tokens = set(gold_normalized.split())
        if gold_tokens and len(pred_tokens & gold_tokens) / len(gold_tokens) >= 0.5:
            return "correct", True
            
        # LLMEval for complex cases
        prompt = LLMEVAL_PROMPT.format(
            question=question,
            predicted=predicted,
            ground_truth=gold
        )
        
        response = self.client.generate(prompt, max_tokens=50).lower()
        
        if "correct" in response or "partial" in response:
            return "correct", True
        elif "abstain" in response:
            return "abstain", False
        else:
            return "incorrect", False
            
    def _normalize(self, text: str) -> str:
        """Normalize text for comparison"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'of', 'in', 'to'}
        words = [w for w in text.split() if w not in stopwords]
        return ' '.join(words)


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_prob_rag_v2_experiment(num_samples: int = 10):
    """Run enhanced Prob-RAG v2 experiment"""
    
    print("=" * 70)
    print("PROB-RAG v2: Enhanced with Paper Insights")
    print("=" * 70)
    print("\nNovel contributions:")
    print("  ✓ Continuous scoring [0,1]")
    print("  ✓ 3-state traffic light routing")
    print("  ✓ Adaptive prompts per state")
    print("  ✓ Smart accuracy (correct abstentions)")
    print("\nIntegrated from paper:")
    print("  ✓ CoT-style autorater prompt")
    print("  ✓ LLMEval for semantic checking")
    print("=" * 70)
    
    # Initialize
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found")
        
    client = GroqClient(api_key)
    scorer = EnhancedSufficiencyScorer(client)
    router = TrafficLightRouter()
    generator = AdaptiveGenerator(client)
    evaluator = EnhancedEvaluator(client)
    
    # Load data
    print(f"\nLoading HotPotQA ({num_samples} samples)...")
    from prob_rag.data.datasets import load_dataset_samples
    samples = load_dataset_samples('hotpotqa', split='validation', num_samples=num_samples)
    
    results = []
    stats = {
        "total": 0,
        "red": 0, "yellow": 0, "green": 0,
        "correct": 0,
        "correct_abstention": 0,
        "wrong_abstention": 0,
        "hallucinated": 0,
        "answer_in_context": 0
    }
    
    print(f"\nProcessing {len(samples)} samples...")
    print("-" * 70)
    
    for i, sample in enumerate(samples):
        stats["total"] += 1
        
        question = sample.question
        context = sample.combined_context
        gold = sample.answer
        
        print(f"\n[{i+1}/{len(samples)}] Q: {question[:70]}...")
        
        # Check if answer is in context
        answer_in_ctx = evaluator.check_answer_in_context(gold, context)
        if answer_in_ctx:
            stats["answer_in_context"] += 1
        print(f"  Answer in context: {'✅' if answer_in_ctx else '❌'}")
        
        # Step 1: Enhanced scoring
        time.sleep(client.rate_limit_delay)
        score, reason = scorer.score(question, context)
        print(f"  Sufficiency score: {score:.3f}")
        
        # Step 2: 3-state routing
        state = router.route(score)
        state_emoji = {"RED": "🔴", "YELLOW": "🟡", "GREEN": "🟢"}[state.value]
        stats[state.value.lower()] += 1
        print(f"  Route: {state_emoji} {state.value}")
        
        # Step 3: Adaptive generation
        time.sleep(client.rate_limit_delay)
        answer = generator.generate(question, context, state)
        print(f"  Answer: {answer[:80]}...")
        
        # Step 4: Evaluate
        time.sleep(client.rate_limit_delay)
        decision, is_correct = evaluator.evaluate(question, answer, gold)
        
        # Smart accuracy calculation
        correct_abstention = False
        if decision == "abstain":
            if not answer_in_ctx:
                correct_abstention = True
                stats["correct_abstention"] += 1
                is_correct = True  # Correct behavior!
            else:
                stats["wrong_abstention"] += 1
        elif is_correct:
            stats["correct"] += 1
        else:
            stats["hallucinated"] += 1
            
        print(f"  Gold: {gold}")
        print(f"  Eval: {decision.upper()} {'✅' if is_correct else '❌'}")
        if correct_abstention:
            print(f"  Note: CORRECT ABSTENTION (answer wasn't in context)")
        
        results.append(EnhancedResult(
            sample_id=sample.id,
            question=question,
            gold_answer=gold,
            sufficiency_score=score,
            answer_in_context=answer_in_ctx,
            routing_state=state,
            generated_answer=answer,
            eval_decision=decision,
            is_correct=is_correct,
            correct_abstention=correct_abstention
        ))
    
    # Calculate metrics
    print("\n" + "=" * 70)
    print("RESULTS: Prob-RAG v2")
    print("=" * 70)
    
    total = stats["total"]
    correct = stats["correct"] + stats["correct_abstention"]
    
    # Smart accuracy includes correct abstentions
    smart_accuracy = correct / total * 100 if total > 0 else 0
    
    # Traditional accuracy (ignores abstentions)
    answered = stats["green"] + stats["yellow"]
    trad_accuracy = stats["correct"] / total * 100 if total > 0 else 0
    
    # Selective accuracy
    selective_acc = stats["correct"] / answered * 100 if answered > 0 else 0
    
    # Abstention metrics
    total_abstentions = stats["red"]
    abstention_precision = stats["correct_abstention"] / total_abstentions * 100 if total_abstentions > 0 else 0
    
    print(f"\n📊 Core Metrics:")
    print(f"  Total Samples:         {total}")
    print(f"  Answer in Context:     {stats['answer_in_context']} ({stats['answer_in_context']/total*100:.1f}%)")
    print()
    print(f"  🎯 SMART Accuracy:     {smart_accuracy:.1f}% (includes correct abstentions)")
    print(f"  📈 Traditional Acc:    {trad_accuracy:.1f}% (abstentions = wrong)")
    print(f"  🎯 Selective Acc:      {selective_acc:.1f}% (among answered)")
    print()
    print(f"📊 Routing Distribution:")
    print(f"  🔴 RED (Abstain):      {stats['red']} ({stats['red']/total*100:.1f}%)")
    print(f"  🟡 YELLOW (Hedge):     {stats['yellow']} ({stats['yellow']/total*100:.1f}%)")
    print(f"  🟢 GREEN (Standard):   {stats['green']} ({stats['green']/total*100:.1f}%)")
    print()
    print(f"📊 Abstention Quality:")
    print(f"  Correct Abstentions:   {stats['correct_abstention']}")
    print(f"  Wrong Abstentions:     {stats['wrong_abstention']}")
    print(f"  Abstention Precision:  {abstention_precision:.1f}%")
    print()
    print(f"📊 Error Analysis:")
    print(f"  Hallucinated:          {stats['hallucinated']} ({stats['hallucinated']/total*100:.1f}%)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/prob_rag_v2_{timestamp}.json"
    os.makedirs("results", exist_ok=True)
    
    output = {
        "experiment": "prob_rag_v2",
        "timestamp": timestamp,
        "methodology": "Enhanced Prob-RAG with paper insights",
        "metrics": {
            "smart_accuracy": smart_accuracy,
            "traditional_accuracy": trad_accuracy,
            "selective_accuracy": selective_acc,
            "abstention_precision": abstention_precision,
            "routing": {
                "red_pct": stats['red']/total*100,
                "yellow_pct": stats['yellow']/total*100,
                "green_pct": stats['green']/total*100
            }
        },
        "results": [
            {
                "sample_id": r.sample_id,
                "question": r.question,
                "gold_answer": r.gold_answer,
                "sufficiency_score": r.sufficiency_score,
                "answer_in_context": r.answer_in_context,
                "routing_state": r.routing_state.value,
                "generated_answer": r.generated_answer,
                "eval_decision": r.eval_decision,
                "is_correct": r.is_correct,
                "correct_abstention": r.correct_abstention
            }
            for r in results
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 Results saved to: {output_file}")
    
    return results, stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    
    run_prob_rag_v2_experiment(args.samples)
