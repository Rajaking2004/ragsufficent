"""
Configuration module for Prob-RAG architecture.

Defines all hyperparameters, thresholds, and model configurations.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import json
import os


class RouterState(Enum):
    """Traffic light states for the dynamic router."""
    RED = "abstention"      # Score < τ_low: Insufficient context
    YELLOW = "hedging"      # τ_low ≤ Score ≤ τ_high: Ambiguous context
    GREEN = "standard"      # Score > τ_high: Sufficient context


@dataclass
class RetrieverConfig:
    """Configuration for Module A: Retriever."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5
    similarity_threshold: float = 0.3
    max_context_length: int = 4096
    chunk_size: int = 512
    chunk_overlap: int = 50
    

@dataclass
class ScorerConfig:
    """Configuration for Module B: Probabilistic Sufficiency Scorer."""
    model_name: str = "gpt-3.5-turbo"  # Or local model
    temperature: float = 0.0  # Deterministic for scoring
    max_tokens: int = 10
    logprobs: bool = True
    top_logprobs: int = 5
    # Tokens to extract probabilities for
    positive_tokens: List[str] = field(default_factory=lambda: ["Yes", "yes", "YES", "True", "true"])
    negative_tokens: List[str] = field(default_factory=lambda: ["No", "no", "NO", "False", "false"])


@dataclass
class RouterConfig:
    """Configuration for Module C: Traffic Light Router."""
    # Thresholds for 3-state classification
    tau_low: float = 0.3    # Below this: RED (abstention)
    tau_high: float = 0.7   # Above this: GREEN (standard)
    # Between tau_low and tau_high: YELLOW (hedging)
    
    # Calibration parameters
    calibration_method: str = "isotonic"  # or "platt"
    use_calibration: bool = False


@dataclass
class GeneratorConfig:
    """Configuration for Module D: Adaptive Generator."""
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 512
    
    # System prompts for each state
    abstention_prompt: str = """You are a helpful assistant. The provided documents do not contain sufficient information to answer the question reliably.

State clearly that the provided documents do not contain the answer. Do NOT attempt to answer from your own knowledge. Be honest about the limitation."""

    hedging_prompt: str = """You are a helpful assistant. The provided documents contain some relevant information, but there may be gaps or ambiguities.

Answer the question based on the documents, but use uncertainty markers like "The text suggests", "It appears that", "Based on the available information". Do not be definitive. Acknowledge any limitations in the provided context."""

    standard_prompt: str = """You are a helpful assistant. The provided documents contain sufficient information to answer the question definitively.

Answer the question based on the documents. Be clear, accurate, and confident in your response."""


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    # Answer evaluation
    use_llm_eval: bool = True
    eval_model: str = "gpt-3.5-turbo"
    
    # Metrics to compute
    compute_accuracy: bool = True
    compute_coverage: bool = True
    compute_selective_accuracy: bool = True
    compute_hallucination_rate: bool = True
    compute_calibration: bool = True
    
    # Coverage levels for selective accuracy curve
    coverage_levels: List[float] = field(default_factory=lambda: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0])


@dataclass
class ProbRAGConfig:
    """Master configuration for the entire Prob-RAG pipeline."""
    # Module configurations
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    scorer: ScorerConfig = field(default_factory=ScorerConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # API Keys (from environment)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    
    # General settings
    seed: int = 42
    device: str = "cuda"  # or "cpu"
    verbose: bool = True
    log_level: str = "INFO"
    
    # Output settings
    output_dir: str = "./results"
    save_predictions: bool = True
    save_scores: bool = True
    
    def __post_init__(self):
        """Load API keys from environment if not provided."""
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.anthropic_api_key is None:
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if self.gemini_api_key is None:
            self.gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    @classmethod
    def from_json(cls, path: str) -> "ProbRAGConfig":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Parse nested configs
        if 'retriever' in data:
            data['retriever'] = RetrieverConfig(**data['retriever'])
        if 'scorer' in data:
            data['scorer'] = ScorerConfig(**data['scorer'])
        if 'router' in data:
            data['router'] = RouterConfig(**data['router'])
        if 'generator' in data:
            data['generator'] = GeneratorConfig(**data['generator'])
        if 'evaluation' in data:
            data['evaluation'] = EvaluationConfig(**data['evaluation'])
        
        return cls(**data)
    
    def to_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        def serialize(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: serialize(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, list):
                return [serialize(item) for item in obj]
            else:
                return obj
        
        data = serialize(self)
        # Remove sensitive data
        data.pop('openai_api_key', None)
        data.pop('anthropic_api_key', None)
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def validate(self, require_api_keys: bool = False) -> bool:
        """
        Validate configuration parameters.
        
        Args:
            require_api_keys: If True, check API keys. If False (default), skip API key check
                             (useful for mock mode).
        """
        errors = []
        
        # Check thresholds
        if not (0 <= self.router.tau_low <= 1):
            errors.append(f"tau_low must be in [0,1], got {self.router.tau_low}")
        if not (0 <= self.router.tau_high <= 1):
            errors.append(f"tau_high must be in [0,1], got {self.router.tau_high}")
        if self.router.tau_low >= self.router.tau_high:
            errors.append(f"tau_low ({self.router.tau_low}) must be < tau_high ({self.router.tau_high})")
        
        # Check retriever
        if self.retriever.top_k < 1:
            errors.append(f"top_k must be >= 1, got {self.retriever.top_k}")
        
        # Check API keys for cloud models only if required
        if require_api_keys:
            if any(m in self.scorer.model_name.lower() for m in ["gpt"]):
                if not self.openai_api_key:
                    errors.append("OpenAI API key required for GPT models")
        
        if errors:
            for e in errors:
                print(f"Configuration Error: {e}")
            return False
        return True


# Preset configurations for different scenarios
PRESETS = {
    "default": ProbRAGConfig(),
    
    "conservative": ProbRAGConfig(
        router=RouterConfig(tau_low=0.4, tau_high=0.8),
        generator=GeneratorConfig(temperature=0.3)
    ),
    
    "aggressive": ProbRAGConfig(
        router=RouterConfig(tau_low=0.2, tau_high=0.6),
        generator=GeneratorConfig(temperature=0.9)
    ),
    
    "high_precision": ProbRAGConfig(
        router=RouterConfig(tau_low=0.5, tau_high=0.85),
        retriever=RetrieverConfig(top_k=10)
    ),
}


def get_preset(name: str) -> ProbRAGConfig:
    """Get a preset configuration by name."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    return PRESETS[name]
