#!/usr/bin/env python3
"""
Paper Baseline Experiment
=========================
Implementation following Joren et al. ICLR 2025 "Sufficient Context" paper EXACTLY.

Key differences from Prob-RAG:
1. Binary autorater (0 or 1) instead of continuous score
2. CoT-style prompt from paper's Appendix C.1
3. 2-state routing (Answer or Abstain) instead of 3-state
4. LLMEval for answer checking

This serves as the baseline for comparison with our Prob-RAG approach.
"""

import os
import json
import time
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================
# PAPER'S EXACT PROMPTS (from Appendix C.1)
# ============================================================

SUFFICIENT_CONTEXT_AUTORATER_PROMPT = """You are an expert LLM evaluator that excels at evaluating a QUESTION and REFERENCES.
Consider the following criteria:
Sufficient Context: 1 IF the CONTEXT is sufficient to infer the answer to the question and 0 IF the CONTEXT cannot be used to infer the answer to the question

First, output a list of step-by-step questions that would be used to arrive at a label for the criteria. Make sure to include questions about assumptions implicit in the QUESTION.
Include questions about any mathematical calculations or arithmetic that would be required.
Next, answer each of the questions. Make sure to work step by step through any required mathematical calculations or arithmetic. Finally, use these answers to evaluate the criteria.
Output the ### EXPLANATION (Text). Then, use the EXPLANATION to output the ### EVALUATION (JSON)

EXAMPLE:
### QUESTION
In which year did the publisher of Roald Dahl's Guide to Railway Safety cease to exist?

### References
Roald Dahl's Guide to Railway Safety was published in 1991 by the British Railways Board. The British Railways Board had asked Roald Dahl to write the text of the booklet, and Quentin Blake to illustrate it, to help young people enjoy using the railways safely. The British Railways Board (BRB) was a nationalised industry in the United Kingdom that operated from 1963 to 2001. Until 1997 it was responsible for most railway services in Great Britain, trading under the brand name British Railways and, from 1965, British Rail.

### EXPLANATION
The context mentions that Roald Dahl's Guide to Railway Safety was published by the British Railways Board. It also states that the British Railways Board operated from 1963 to 2001, meaning the year it ceased to exist was 2001. Therefore, the context does provide a precise answer to the question.

### JSON
{{"Sufficient Context": 1}}

Now evaluate the following:

### QUESTION
{question}

### References
{context}
"""

# Paper's Chain of Thought prompt for answer generation
COT_ANSWER_PROMPT = """Write an accurate and concise answer for the given question using only the provided search results (some of which might be irrelevant). Start with an accurate, engaging, and concise explanation based only on the provided documents. Must end with "The answer is:". Use an unbiased and journalistic tone.

### Question
{question}

### References
{context}

### Answer
"""

# Paper's LLMEval prompt for answer checking
LLMEVAL_PROMPT = """===Task===
I need your help in evaluating an answer provided by an LLM against ground truth answers.
Your task is to determine if the LLM's response matches the ground truth answers. Please analyze the provided data and make a decision.

===Instructions===
1. Carefully compare the "Predicted Answer" with the "Ground Truth Answers".
2. Consider the substance of the answers– look for equivalent information or correct answers. Do not focus on exact wording unless the exact wording is crucial to the meaning.
3. Your final decision should be based on whether the meaning and the vital facts of the "Ground Truth Answers" are present in the "Predicted Answer."
4. Categorize the answer as one of the following:
- "perfect": The answer is completely correct and matches the ground truth.
- "acceptable": The answer is partially correct or contains the main idea of the ground truth.
- "incorrect": The answer is wrong or contradicts the ground truth.
- "missing": The answer is "I don't know", "invalid question", or similar responses indicating lack of knowledge.

===Input Data===
- Question: {question}
- Predicted Answer: {predicted}
- Ground Truth Answers: {ground_truth}

===Output Format===
Provide your evaluation in the following format:
Explanation: (How you made the decision)
Decision: (One of "perfect", "acceptable", "incorrect", or "missing")

Please proceed with the evaluation.
"""

# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class PaperBaselineResult:
    """Result from paper's binary autorater approach"""
    sample_id: str
    question: str
    context: str
    gold_answer: str
    
    # Autorater output
    sufficient_context: bool  # Binary: True or False
    autorater_explanation: str
    
    # Generation output
    generated_answer: str
    abstained: bool
    
    # Evaluation
    llmeval_decision: str  # "perfect", "acceptable", "incorrect", "missing"
    llmeval_explanation: str
    is_correct: bool


# ============================================================
# GROQ API CLIENT (Free tier with higher limits)
# ============================================================

class GroqClient:
    """Groq API client for paper baseline experiments"""
    
    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant"):
        from groq import Groq
        self.client = Groq(api_key=api_key)
        self.model = model
        self.rate_limit_delay = 16  # seconds between requests
        
    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate response from Groq"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()


# ============================================================
# PAPER'S BINARY AUTORATER
# ============================================================

class BinarySufficientContextAutorater:
    """
    Paper's binary autorater implementation.
    Returns 0 or 1 based on whether context is sufficient.
    Uses CoT-style prompt from Appendix C.1.
    """
    
    def __init__(self, client: GroqClient):
        self.client = client
        
    def evaluate(self, question: str, context: str) -> tuple:
        """
        Returns: (is_sufficient: bool, explanation: str)
        """
        prompt = SUFFICIENT_CONTEXT_AUTORATER_PROMPT.format(
            question=question,
            context=context[:6000]  # Truncate to 6000 chars as in paper
        )
        
        response = self.client.generate(prompt, max_tokens=1024)
        
        # Parse the response
        is_sufficient = False
        explanation = response
        
        # Try to extract JSON
        json_match = re.search(r'\{[^}]*"Sufficient Context"[^}]*\}', response, re.IGNORECASE)
        if json_match:
            try:
                result = json.loads(json_match.group())
                is_sufficient = result.get("Sufficient Context", 0) == 1
            except:
                # Fallback: look for "1" in the response
                is_sufficient = '"Sufficient Context": 1' in response or "'Sufficient Context': 1" in response
        
        # Extract explanation
        expl_match = re.search(r'### EXPLANATION\s*(.*?)(?=### |$)', response, re.DOTALL | re.IGNORECASE)
        if expl_match:
            explanation = expl_match.group(1).strip()
            
        return is_sufficient, explanation


# ============================================================
# PAPER'S 2-STATE ROUTING (Answer or Abstain)
# ============================================================

class TwoStateRouter:
    """
    Paper's simple 2-state routing:
    - If sufficient context → Answer
    - If insufficient context → Abstain
    """
    
    def route(self, is_sufficient: bool) -> str:
        return "ANSWER" if is_sufficient else "ABSTAIN"


# ============================================================
# PAPER'S ANSWER GENERATOR
# ============================================================

class PaperAnswerGenerator:
    """
    Uses paper's CoT prompt for answer generation.
    Extracts answer from "The answer is:" pattern.
    """
    
    def __init__(self, client: GroqClient):
        self.client = client
        
    def generate(self, question: str, context: str, should_answer: bool) -> tuple:
        """
        Returns: (answer: str, abstained: bool)
        """
        if not should_answer:
            return "I don't know", True
            
        prompt = COT_ANSWER_PROMPT.format(
            question=question,
            context=context[:6000]
        )
        
        response = self.client.generate(prompt, max_tokens=512)
        
        # Extract answer from "The answer is:" pattern
        answer_match = re.search(r'The answer is[:\s]*(.+?)(?:\.|$)', response, re.IGNORECASE | re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
        else:
            # Use last sentence as answer
            sentences = response.split('.')
            answer = sentences[-1].strip() if sentences else response
            
        return answer, False


# ============================================================
# PAPER'S LLMEVAL
# ============================================================

class LLMEvaluator:
    """
    Paper's LLM-based evaluation (from Appendix C.3).
    More robust than exact match - handles semantic equivalence.
    """
    
    def __init__(self, client: GroqClient):
        self.client = client
        
    def evaluate(self, question: str, predicted: str, ground_truth: str) -> tuple:
        """
        Returns: (decision: str, explanation: str)
        Decision is one of: "perfect", "acceptable", "incorrect", "missing"
        """
        # Check for abstention first
        abstention_phrases = ["i don't know", "i cannot", "not enough information", 
                            "insufficient", "cannot answer", "unable to"]
        if any(phrase in predicted.lower() for phrase in abstention_phrases):
            return "missing", "Response indicates abstention"
            
        prompt = LLMEVAL_PROMPT.format(
            question=question,
            predicted=predicted,
            ground_truth=ground_truth
        )
        
        response = self.client.generate(prompt, max_tokens=256)
        
        # Extract decision
        decision = "incorrect"  # default
        explanation = response
        
        decision_match = re.search(r'Decision[:\s]*["\']?(\w+)["\']?', response, re.IGNORECASE)
        if decision_match:
            decision = decision_match.group(1).lower()
            if decision not in ["perfect", "acceptable", "incorrect", "missing"]:
                decision = "incorrect"
                
        expl_match = re.search(r'Explanation[:\s]*(.*?)(?=Decision|$)', response, re.DOTALL | re.IGNORECASE)
        if expl_match:
            explanation = expl_match.group(1).strip()
            
        return decision, explanation


# ============================================================
# MAIN EXPERIMENT RUNNER
# ============================================================

def run_paper_baseline_experiment(num_samples: int = 10):
    """Run experiment using paper's exact methodology"""
    
    print("=" * 70)
    print("PAPER BASELINE EXPERIMENT")
    print("Implementation of Joren et al. ICLR 2025")
    print("=" * 70)
    
    # Initialize
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment")
        
    client = GroqClient(api_key)
    autorater = BinarySufficientContextAutorater(client)
    router = TwoStateRouter()
    generator = PaperAnswerGenerator(client)
    evaluator = LLMEvaluator(client)
    
    # Load dataset
    print(f"\nLoading HotPotQA dataset ({num_samples} samples)...")
    from prob_rag.data.datasets import load_dataset_samples
    samples = load_dataset_samples('hotpotqa', split='validation', num_samples=num_samples)
    
    results = []
    stats = {
        "total": 0,
        "sufficient_context": 0,
        "insufficient_context": 0,
        "correct": 0,
        "abstained": 0,
        "hallucinated": 0
    }
    
    print(f"\nProcessing {len(samples)} samples...")
    print("-" * 70)
    
    for i, sample in enumerate(samples):
        stats["total"] += 1
        
        question = sample.question
        context = sample.combined_context
        gold = sample.answer
        
        print(f"\n[{i+1}/{len(samples)}] Q: {question[:80]}...")
        
        # Step 1: Binary autorater (paper's approach)
        time.sleep(client.rate_limit_delay)
        is_sufficient, autorater_expl = autorater.evaluate(question, context)
        
        if is_sufficient:
            stats["sufficient_context"] += 1
            print(f"  Autorater: SUFFICIENT ✓")
        else:
            stats["insufficient_context"] += 1
            print(f"  Autorater: INSUFFICIENT ✗")
        
        # Step 2: 2-state routing
        action = router.route(is_sufficient)
        print(f"  Route: {action}")
        
        # Step 3: Generate answer
        time.sleep(client.rate_limit_delay)
        answer, abstained = generator.generate(question, context, action == "ANSWER")
        
        if abstained:
            stats["abstained"] += 1
            print(f"  Answer: [ABSTAINED]")
        else:
            print(f"  Answer: {answer[:100]}...")
        
        # Step 4: LLMEval
        time.sleep(client.rate_limit_delay)
        decision, eval_expl = evaluator.evaluate(question, answer, gold)
        
        is_correct = decision in ["perfect", "acceptable"]
        if is_correct:
            stats["correct"] += 1
        elif decision == "missing":
            pass  # already counted as abstained
        else:
            stats["hallucinated"] += 1
            
        print(f"  Gold: {gold}")
        print(f"  Eval: {decision.upper()} {'✅' if is_correct else '❌'}")
        
        result = PaperBaselineResult(
            sample_id=sample.id,
            question=question,
            context=context[:500] + "...",
            gold_answer=gold,
            sufficient_context=is_sufficient,
            autorater_explanation=autorater_expl[:200],
            generated_answer=answer,
            abstained=abstained,
            llmeval_decision=decision,
            llmeval_explanation=eval_expl[:200],
            is_correct=is_correct
        )
        results.append(result)
    
    # Calculate metrics (paper's approach)
    print("\n" + "=" * 70)
    print("RESULTS (Paper's Methodology)")
    print("=" * 70)
    
    total = stats["total"]
    correct = stats["correct"]
    abstained = stats["abstained"]
    hallucinated = stats["hallucinated"]
    sufficient = stats["sufficient_context"]
    insufficient = stats["insufficient_context"]
    
    # Paper's metrics
    accuracy = correct / total * 100 if total > 0 else 0
    coverage = (total - abstained) / total * 100 if total > 0 else 0
    selective_accuracy = correct / (total - abstained) * 100 if (total - abstained) > 0 else 0
    hallucination_rate = hallucinated / (total - abstained) * 100 if (total - abstained) > 0 else 0
    
    print(f"\n📊 Metrics:")
    print(f"  Total Samples:        {total}")
    print(f"  Sufficient Context:   {sufficient} ({sufficient/total*100:.1f}%)")
    print(f"  Insufficient Context: {insufficient} ({insufficient/total*100:.1f}%)")
    print(f"")
    print(f"  Correct:              {correct} ({accuracy:.1f}%)")
    print(f"  Abstained:            {abstained} ({abstained/total*100:.1f}%)")
    print(f"  Hallucinated:         {hallucinated} ({hallucination_rate:.1f}% of answered)")
    print(f"")
    print(f"  Coverage:             {coverage:.1f}%")
    print(f"  Selective Accuracy:   {selective_accuracy:.1f}%")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/paper_baseline_{timestamp}.json"
    os.makedirs("results", exist_ok=True)
    
    output = {
        "experiment": "paper_baseline",
        "timestamp": timestamp,
        "config": {
            "model": "llama-3.1-8b-instant",
            "dataset": "hotpotqa",
            "num_samples": num_samples,
            "methodology": "Joren et al. ICLR 2025"
        },
        "metrics": {
            "accuracy": accuracy,
            "coverage": coverage,
            "selective_accuracy": selective_accuracy,
            "hallucination_rate": hallucination_rate,
            "sufficient_context_pct": sufficient/total*100,
            "insufficient_context_pct": insufficient/total*100
        },
        "results": [
            {
                "sample_id": r.sample_id,
                "question": r.question,
                "gold_answer": r.gold_answer,
                "sufficient_context": r.sufficient_context,
                "generated_answer": r.generated_answer,
                "abstained": r.abstained,
                "llmeval_decision": r.llmeval_decision,
                "is_correct": r.is_correct
            }
            for r in results
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 Results saved to: {output_file}")
    
    return results, stats


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Paper Baseline Experiment")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples")
    args = parser.parse_args()
    
    run_paper_baseline_experiment(args.samples)
