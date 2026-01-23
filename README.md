# Prob-RAG: Probabilistic Sufficient Context RAG

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A novel Retrieval-Augmented Generation (RAG) system that uses **probabilistic sufficiency scoring** with **traffic light routing** to improve answer reliability and reduce hallucinations.

## 🎯 Key Innovation

Unlike traditional RAG systems that always generate answers regardless of context quality, Prob-RAG:

1. **Scores context sufficiency** using LLM log-probabilities (continuous 0-1 score)
2. **Routes dynamically** using a 3-state traffic light system
3. **Adapts generation** based on confidence level

```
                    ┌─────────────────────────────────────────────────────┐
                    │              PROB-RAG ARCHITECTURE                   │
                    └─────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  MODULE A: RETRIEVER                                                         │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────────┐         │
│  │   Query     │───▶│  Embedding   │───▶│  Vector Search (FAISS)  │         │
│  │   (q)       │    │   Model      │    │    Top-k retrieval      │         │
│  └─────────────┘    └──────────────┘    └────────────┬────────────┘         │
│                                                       │                      │
│                                         Retrieved Contexts (C)               │
└──────────────────────────────────────────────────────│───────────────────────┘
                                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  MODULE B: PROBABILISTIC SUFFICIENCY SCORER                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Prompt: "Given the context, can you answer the question? Yes/No"   │    │
│  │                                                                      │    │
│  │  Score = exp(logit_Yes) / (exp(logit_Yes) + exp(logit_No))          │    │
│  │                                                                      │    │
│  │  Continuous score S ∈ [0, 1]                                        │    │
│  └─────────────────────────────────────────────────┬───────────────────┘    │
│                                                     │                        │
│                                         Sufficiency Score (S)                │
└─────────────────────────────────────────────────────│────────────────────────┘
                                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  MODULE C: TRAFFIC LIGHT ROUTER                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                      │    │
│  │    S < τ_low (0.3)        τ_low ≤ S < τ_high      S ≥ τ_high (0.7) │    │
│  │         │                        │                       │          │    │
│  │         ▼                        ▼                       ▼          │    │
│  │    ┌────────┐              ┌────────┐              ┌────────┐       │    │
│  │    │  🔴   │              │  🟡   │              │  🟢   │       │    │
│  │    │  RED   │              │ YELLOW │              │ GREEN  │       │    │
│  │    └────────┘              └────────┘              └────────┘       │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    ▼                     ▼                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  MODULE D: ADAPTIVE GENERATOR                                                │
│  ┌────────────────────┬────────────────────┬─────────────────────────┐      │
│  │     ABSTENTION     │      HEDGING       │        STANDARD         │      │
│  │     PROTOCOL       │      PROTOCOL      │        PROTOCOL         │      │
│  │                    │                    │                         │      │
│  │  "The provided     │  "Based on the     │  "Answer the question   │      │
│  │   context does     │   context, it      │   using only the        │      │
│  │   not contain      │   appears that...  │   provided context."    │      │
│  │   sufficient       │   However, please  │                         │      │
│  │   information."    │   verify..."       │                         │      │
│  └────────────────────┴────────────────────┴─────────────────────────┘      │
└──────────────────────────────────────────────────────────────────────────────┘
```

## 🏗️ Project Structure

```
prob_rag/
├── __init__.py              # Package exports
├── config.py                # Configuration dataclasses
├── pipeline.py              # Main pipeline integration
├── modules/
│   ├── retriever.py         # Module A: Vector search retrieval
│   ├── scorer.py            # Module B: Probabilistic scoring
│   ├── router.py            # Module C: Traffic light routing
│   └── generator.py         # Module D: Adaptive generation
├── data/
│   └── datasets.py          # Dataset loaders (HotPotQA, Musique, etc.)
└── evaluation/
    └── metrics.py           # Evaluation metrics

experiments/
├── run_experiments.py       # Experiment runner CLI
└── visualization.py         # Publication-quality plots

tests/
└── test_modules.py          # Unit tests

main.py                      # Main entry point
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/prob-rag.git
cd prob-rag

# Install dependencies
pip install -r requirements.txt

# (Optional) Set up API keys for real LLM usage
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
```

### Run Demo

```bash
# Quick demonstration with mock components
python main.py --mode demo
```

### Interactive Mode

```bash
# Ask questions interactively
python main.py --mode interactive
```

### Run Experiments

```bash
# Single experiment on synthetic data
python main.py --mode experiment --experiment-type single --num-samples 100

# Threshold sweep experiment
python main.py --mode experiment --experiment-type sweep --dataset synthetic

# Multi-dataset comparison
python main.py --mode experiment --experiment-type multi

# Use real API (requires API keys)
python main.py --mode experiment --use-api
```

## 📊 Key Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall answer correctness |
| **Coverage** | % of questions answered (non-abstentions) |
| **Selective Accuracy** | Accuracy among answered questions |
| **Abstention Rate** | % of questions abstained from |
| **Calibration (ECE)** | Expected Calibration Error |
| **Hallucination Rate** | % of confident but wrong answers |

## 🔬 Technical Details

### Sufficiency Scoring Formula

The core innovation uses LLM log-probabilities for continuous scoring:

$$S = \frac{\exp(\text{logit}_{\text{Yes}})}{\exp(\text{logit}_{\text{Yes}}) + \exp(\text{logit}_{\text{No}})}$$

Where:
- $\text{logit}_{\text{Yes}}$ = log-probability of "Yes" token
- $\text{logit}_{\text{No}}$ = log-probability of "No" token

### Routing Thresholds

| Score Range | State | Action |
|-------------|-------|--------|
| $S < \tau_{\text{low}}$ (default 0.3) | 🔴 RED | Abstention - Decline to answer |
| $\tau_{\text{low}} \leq S < \tau_{\text{high}}$ | 🟡 YELLOW | Hedging - Answer with caveats |
| $S \geq \tau_{\text{high}}$ (default 0.7) | 🟢 GREEN | Standard - Confident answer |

## 📈 Comparison with Existing Work

| Feature | Joren et al. (ICLR 2025) | **Prob-RAG (Ours)** |
|---------|--------------------------|---------------------|
| Scoring | Binary autorater | **Continuous log-prob score** |
| Routing | 2-state (answer/abstain) | **3-state traffic light** |
| Uncertainty | Not modeled | **Hedging protocol** |
| Calibration | Not addressed | **Calibrated confidence** |

## 📚 Supported Datasets

- **HotPotQA**: Multi-hop reasoning
- **Musique**: Multi-step compositional QA
- **Natural Questions**: Real Google queries
- **TriviaQA**: Trivia questions
- **Synthetic**: Generated test data (no API required)

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_modules.py -v

# Run with coverage
pytest tests/ --cov=prob_rag --cov-report=html
```

## 📄 Citation

If you use this work, please cite:

```bibtex
@article{probrag2024,
  title={Prob-RAG: Probabilistic Sufficient Context RAG with Traffic Light Routing},
  author={Your Name},
  journal={NIT Conference},
  year={2024}
}
```

## 🔗 Related Work

- [Sufficient Context: A New Lens on Retrieval Augmented Generation Systems](https://arxiv.org/abs/2411.06037) (Joren et al., ICLR 2025)

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

---

**Note**: This is a research implementation. For production use, ensure proper error handling, rate limiting, and cost management for LLM API calls.
