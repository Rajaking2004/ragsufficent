# Comparative Analysis: Prob-RAG vs. ICLR 2025 "Sufficient Context" Paper

## Paper Overview

**Title**: Sufficient Context: A New Lens on Retrieval Augmented Generation Systems  
**Authors**: Hailey Joren et al. (UCSD, Google, Duke)  
**Venue**: ICLR 2025  
**Paper**: arXiv:2411.06037v3

---

## Key Contributions of the Paper

### 1. Definition of Sufficient Context
> "An instance (Q, C) has sufficient context if and only if there exists an answer A' such that A' is a plausible answer to the question Q given the information in C."

**Key distinction from entailment**: Does NOT require knowing the answer in advance.

### 2. Sufficient Context Autorater
- Uses **Gemini 1.5 Pro (1-shot)** achieving **93% accuracy**
- Binary classification: Sufficient (1) or Insufficient (0)
- CoT-style prompt with step-by-step reasoning

### 3. Key Findings
| Finding | Description |
|---------|-------------|
| **Models hallucinate > abstain** | Even with sufficient context, models output incorrect answers rather than saying "I don't know" |
| **35-62% correct with insufficient context** | Models can answer correctly without full context (parametric memory, lucky guesses) |
| **RAG reduces abstention** | Adding context makes models more confident, even when they shouldn't be |

### 4. Selective Generation Framework
- Combines: **Sufficient Context Signal + Self-Rated Confidence (P(True) or P(Correct))**
- Uses logistic regression to predict hallucinations
- Achieves **2-10% improvement** in selective accuracy

---

## Comparison: Paper vs. Prob-RAG

| Aspect | Paper (Joren et al.) | Prob-RAG (Ours) |
|--------|---------------------|-----------------|
| **Scoring** | Binary (0 or 1) | Continuous [0, 1] |
| **Autorater** | Gemini 1.5 Pro (93% acc) | Groq Llama 3.1 8B |
| **Routing** | 2-state (Answer/Abstain) | **3-state (RED/YELLOW/GREEN)** |
| **Uncertainty** | Not modeled in routing | **Hedging protocol (YELLOW)** |
| **Prompt Style** | CoT with step-by-step questions | Simplified direct question |
| **Confidence** | P(True)/P(Correct) sampling | Single-call binary |
| **Selective Gen** | Logistic regression head | Threshold-based routing |

---

## What We Can Learn from the Paper

### 1. Better Autorater Prompt (from Appendix C.1)
```
You are an expert LLM evaluator...
First, output a list of step-by-step questions that would be used to arrive at a label.
Next, answer each of the questions.
Finally, use these answers to evaluate the criteria.
Output ### EXPLANATION (Text). Then ### EVALUATION (JSON)
```

### 2. Self-Rated Confidence Signals
- **P(True)**: Sample 20 responses, query model 5 times to evaluate correctness
- **P(Correct)**: Direct probability from model (cheaper for proprietary APIs)

### 3. LLMEval for Answer Checking
Instead of exact match, use LLM to classify:
- "perfect": Completely correct
- "acceptable": Partially correct
- "incorrect": Wrong
- "missing": Abstention ("I don't know")

### 4. Fine-tuning Insights
- Training with "I don't know" answers can increase abstention
- But may reduce overall correct answers
- Balance is tricky

---

## Novel Contributions of Prob-RAG

### 1. Continuous Scoring (vs Binary)
**Paper**: S ∈ {0, 1}  
**Ours**: S ∈ [0, 1] via softmax over logprobs

This allows for **calibrated confidence** rather than hard decisions.

### 2. Traffic Light Router (3-State)
| State | Score Range | Action |
|-------|-------------|--------|
| 🔴 RED | S < 0.3 | Abstain |
| 🟡 YELLOW | 0.3 ≤ S < 0.7 | Hedge |
| 🟢 GREEN | S ≥ 0.7 | Answer |

**Paper only has 2 states** (answer or abstain). Our YELLOW state handles uncertainty gracefully.

### 3. Adaptive Prompts Per State
Different system prompts for:
- Abstention: "I cannot find the answer..."
- Hedging: "Based on available information, possibly..."
- Standard: Direct confident answer

### 4. Smart Accuracy Calculation
Abstention is **correct** if answer was NOT in context.
This aligns with the paper's ideal: "output correct answer OR abstain"

---

## Experimental Comparison

### Paper Results (from Table in paper)
| Model | Dataset | Sufficient Context | Correct % |
|-------|---------|-------------------|-----------|
| Gemini 1.5 Pro | HotPotQA | Yes | 67.5% |
| Gemini 1.5 Pro | HotPotQA | No | 49.4% |
| GPT-4o | HotPotQA | Yes | 71.9% |
| GPT-4o | HotPotQA | No | 59.5% |

### Our Results (Groq Llama 3.1 8B)
| Metric | Value |
|--------|-------|
| Overall Accuracy | 80% (5 samples) |
| GREEN Rate | 100% |
| Fuzzy Matching | Enabled |

**Note**: Our results use a much smaller model but with improved prompts and fuzzy matching.

---

## Recommendations for NIT Conference

### What Makes This Publishable

1. **Novel 3-State Routing**: Paper only explores 2-state. We add hedging.
2. **Continuous vs Binary**: More information-theoretic approach
3. **Adaptive Prompting**: State-dependent generation
4. **Smart Evaluation**: Correct abstention counting

### Experiments to Add

1. **Compare 2-state vs 3-state** routing on same data
2. **Calibration analysis**: ECE (Expected Calibration Error)
3. **Hedging quality**: Human eval of YELLOW responses
4. **Threshold optimization**: Find optimal τ_low, τ_high

### Baseline Comparisons

1. Paper's binary autorater (our implementation)
2. Standard RAG (always answer)
3. Prob-RAG with 3-state routing

---

## Git Branch Structure

```
main                    ← Current implementation
│
├── paper-baseline      ← Exact paper implementation (binary, CoT prompt)
│
├── prob-rag-v2        ← Enhanced with paper insights
│
└── experiments        ← All experimental scripts
```

---

## Citation

```bibtex
@inproceedings{joren2025sufficient,
  title={Sufficient Context: A New Lens on Retrieval Augmented Generation Systems},
  author={Joren, Hailey and Juan, Da-Cheng and Zhang, Jianyi and Taly, Ankur and Ferng, Chun-Sung and Rashtchian, Cyrus},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```
