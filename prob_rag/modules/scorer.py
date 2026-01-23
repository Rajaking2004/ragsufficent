"""
Module B: Probabilistic Sufficiency Scorer

The core innovation of Prob-RAG. Instead of binary classification,
this module extracts log-probabilities from the LLM's response to
compute a continuous sufficiency score.

Math:
    S = exp(logit(Yes)) / (exp(logit(Yes)) + exp(logit(No)))
    
    This gives a continuous score S ∈ [0, 1] where:
    - S ≈ 0: Context is clearly insufficient
    - S ≈ 0.5: Ambiguous/uncertain
    - S ≈ 1: Context is clearly sufficient
"""

import logging
import math
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import re

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from ..config import ScorerConfig


logger = logging.getLogger(__name__)


# Sufficiency prompt template - asks LLM to judge if context is sufficient
SUFFICIENCY_PROMPT_TEMPLATE = """You are an expert evaluator. Your task is to determine if the provided context contains sufficient information to answer the given question.

### QUESTION
{question}

### CONTEXT
{context}

### TASK
Based ONLY on the provided context, can the question be definitively answered?
- Answer "Yes" if the context contains all necessary information to answer the question completely and accurately.
- Answer "No" if the context is missing information, is ambiguous, or would require external knowledge.

Your response must be exactly one word: Yes or No"""


# Alternative prompt with more nuance
SUFFICIENCY_PROMPT_V2 = """Evaluate whether the following context is sufficient to answer the question.

Question: {question}

Context:
{context}

Criteria for sufficiency:
1. All facts needed to answer are explicitly stated in the context
2. No external knowledge or inference beyond the context is required
3. The answer can be definitively determined (not ambiguous)

Is the context sufficient? Answer with exactly one word: Yes or No"""


@dataclass
class SufficiencyScore:
    """Container for sufficiency scoring results."""
    # Core score
    score: float  # Continuous score in [0, 1]
    
    # Raw log probabilities
    logprob_yes: Optional[float] = None
    logprob_no: Optional[float] = None
    
    # Token-level details
    top_tokens: Optional[List[Dict[str, Any]]] = None
    
    # Generated response (the actual Yes/No)
    response: Optional[str] = None
    
    # Metadata
    model: str = ""
    prompt_version: str = "v1"
    
    def __repr__(self) -> str:
        return f"SufficiencyScore(score={self.score:.4f}, response={self.response})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "logprob_yes": self.logprob_yes,
            "logprob_no": self.logprob_no,
            "response": self.response,
            "model": self.model,
            "prompt_version": self.prompt_version
        }


class ProbabilisticSufficiencyScorer:
    """
    Module B: Probabilistic Sufficiency Scorer
    
    Uses log-probabilities from LLM to compute continuous sufficiency scores.
    This is the key innovation over binary autoraters.
    """
    
    def __init__(self, config: ScorerConfig, api_key: Optional[str] = None):
        """
        Initialize the scorer.
        
        Args:
            config: ScorerConfig with model and token parameters
            api_key: API key for the LLM provider
        """
        self.config = config
        self.api_key = api_key
        self.client = None
        
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the appropriate API client."""
        model_lower = self.config.model_name.lower()
        
        if "gpt" in model_lower or "openai" in model_lower:
            if OpenAI is None:
                raise ImportError("openai package required. Install with: pip install openai")
            if not self.api_key:
                raise ValueError("OpenAI API key required for GPT models")
            self.client = OpenAI(api_key=self.api_key)
            self.provider = "openai"
            
        elif "claude" in model_lower:
            if anthropic is None:
                raise ImportError("anthropic package required. Install with: pip install anthropic")
            if not self.api_key:
                raise ValueError("Anthropic API key required for Claude models")
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self.provider = "anthropic"
        
        elif "gemini" in model_lower:
            if genai is None:
                raise ImportError("google-generativeai package required. Install with: pip install google-generativeai")
            if not self.api_key:
                raise ValueError("Google API key required for Gemini models")
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.config.model_name)
            self.provider = "gemini"
            
        else:
            # Assume OpenAI-compatible API
            logger.warning(f"Unknown model {self.config.model_name}, assuming OpenAI-compatible")
            if OpenAI is None:
                raise ImportError("openai package required")
            self.client = OpenAI(api_key=self.api_key)
            self.provider = "openai"
    
    def _build_prompt(self, question: str, context: str, version: str = "v1") -> str:
        """Build the sufficiency evaluation prompt."""
        template = SUFFICIENCY_PROMPT_TEMPLATE if version == "v1" else SUFFICIENCY_PROMPT_V2
        return template.format(question=question, context=context)
    
    def _extract_logprobs_openai(
        self,
        question: str,
        context: str
    ) -> Tuple[Optional[float], Optional[float], str, List[Dict]]:
        """
        Extract log probabilities using OpenAI API.
        
        Returns:
            Tuple of (logprob_yes, logprob_no, response_text, top_tokens)
        """
        prompt = self._build_prompt(question, context)
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "You are a precise evaluator. Answer only Yes or No."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                logprobs=True,
                top_logprobs=self.config.top_logprobs
            )
            
            # Get response text
            response_text = response.choices[0].message.content.strip()
            
            # Extract logprobs
            logprob_yes = None
            logprob_no = None
            top_tokens = []
            
            if response.choices[0].logprobs and response.choices[0].logprobs.content:
                # Get first token's logprobs (should be Yes/No)
                first_token_logprobs = response.choices[0].logprobs.content[0]
                
                if first_token_logprobs.top_logprobs:
                    for token_info in first_token_logprobs.top_logprobs:
                        token = token_info.token.strip().lower()
                        logprob = token_info.logprob
                        
                        top_tokens.append({
                            "token": token_info.token,
                            "logprob": logprob
                        })
                        
                        # Match positive tokens
                        if any(t.lower() == token for t in self.config.positive_tokens):
                            if logprob_yes is None or logprob > logprob_yes:
                                logprob_yes = logprob
                        
                        # Match negative tokens
                        if any(t.lower() == token for t in self.config.negative_tokens):
                            if logprob_no is None or logprob > logprob_no:
                                logprob_no = logprob
            
            return logprob_yes, logprob_no, response_text, top_tokens
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def _extract_logprobs_anthropic(
        self,
        question: str,
        context: str
    ) -> Tuple[Optional[float], Optional[float], str, List[Dict]]:
        """
        Extract probabilities using Anthropic API.
        Note: Anthropic doesn't provide logprobs directly, so we use
        multiple samples to estimate probability.
        """
        prompt = self._build_prompt(question, context)
        
        # Sample multiple times to estimate probability
        n_samples = 10
        yes_count = 0
        responses = []
        
        try:
            for _ in range(n_samples):
                response = self.client.messages.create(
                    model=self.config.model_name,
                    max_tokens=self.config.max_tokens,
                    temperature=0.7,  # Need some temperature for variety
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                text = response.content[0].text.strip().lower()
                responses.append(text)
                
                if text.startswith("yes"):
                    yes_count += 1
            
            # Estimate probability
            p_yes = yes_count / n_samples
            p_no = 1 - p_yes
            
            # Convert to log probabilities (with smoothing to avoid log(0))
            eps = 1e-10
            logprob_yes = math.log(max(p_yes, eps))
            logprob_no = math.log(max(p_no, eps))
            
            # Use majority vote for response
            response_text = "Yes" if yes_count > n_samples / 2 else "No"
            
            return logprob_yes, logprob_no, response_text, []
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    def _extract_logprobs_gemini(
        self,
        question: str,
        context: str
    ) -> Tuple[Optional[float], Optional[float], str, List[Dict]]:
        """
        Extract probabilities using Google Gemini API.
        Gemini doesn't provide logprobs directly, so we use response text
        and multiple sampling to estimate.
        """
        prompt = self._build_prompt(question, context)
        
        try:
            # Configure generation
            generation_config = genai.types.GenerationConfig(
                temperature=0.0,  # Deterministic
                max_output_tokens=self.config.max_tokens,
            )
            
            # Generate response
            response = self.client.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            response_text = response.text.strip() if response.text else ""
            
            # Parse response - Gemini should respond Yes or No
            response_lower = response_text.lower()
            
            # Estimate logprobs based on response
            # Since Gemini doesn't give logprobs, we use confidence heuristics
            if response_lower.startswith("yes"):
                # Sample multiple times to estimate confidence
                yes_count = 0
                n_samples = 5
                
                # Quick sampling with higher temperature
                sampling_config = genai.types.GenerationConfig(
                    temperature=0.5,
                    max_output_tokens=5,
                )
                
                for _ in range(n_samples):
                    try:
                        sample_response = self.client.generate_content(
                            prompt,
                            generation_config=sampling_config
                        )
                        if sample_response.text and sample_response.text.strip().lower().startswith("yes"):
                            yes_count += 1
                    except:
                        yes_count += 1  # Assume yes if error
                
                confidence = yes_count / n_samples
                logprob_yes = math.log(max(confidence, 0.01))
                logprob_no = math.log(max(1 - confidence, 0.01))
                
            elif response_lower.startswith("no"):
                # Same sampling approach
                no_count = 0
                n_samples = 5
                
                sampling_config = genai.types.GenerationConfig(
                    temperature=0.5,
                    max_output_tokens=5,
                )
                
                for _ in range(n_samples):
                    try:
                        sample_response = self.client.generate_content(
                            prompt,
                            generation_config=sampling_config
                        )
                        if sample_response.text and sample_response.text.strip().lower().startswith("no"):
                            no_count += 1
                    except:
                        no_count += 1
                
                confidence = no_count / n_samples
                logprob_no = math.log(max(confidence, 0.01))
                logprob_yes = math.log(max(1 - confidence, 0.01))
            else:
                # Ambiguous response
                logprob_yes = math.log(0.5)
                logprob_no = math.log(0.5)
            
            return logprob_yes, logprob_no, response_text, []
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
    def _compute_score(
        self,
        logprob_yes: Optional[float],
        logprob_no: Optional[float],
        response_text: str
    ) -> float:
        """
        Compute the continuous sufficiency score from log probabilities.
        
        Formula:
            S = exp(logprob_yes) / (exp(logprob_yes) + exp(logprob_no))
            
        This is equivalent to softmax([logprob_yes, logprob_no])[0]
        
        Args:
            logprob_yes: Log probability of "Yes" token
            logprob_no: Log probability of "No" token
            response_text: The actual generated response
            
        Returns:
            Continuous score in [0, 1]
        """
        # If we have both logprobs, compute softmax
        if logprob_yes is not None and logprob_no is not None:
            # Numerically stable softmax
            max_logprob = max(logprob_yes, logprob_no)
            exp_yes = math.exp(logprob_yes - max_logprob)
            exp_no = math.exp(logprob_no - max_logprob)
            
            score = exp_yes / (exp_yes + exp_no)
            return score
        
        # Fallback: if we only have one logprob, use it directly
        if logprob_yes is not None:
            # Convert logprob to probability
            return min(math.exp(logprob_yes), 1.0)
        
        if logprob_no is not None:
            # If only No logprob, score is 1 - P(No)
            return max(1.0 - math.exp(logprob_no), 0.0)
        
        # Last resort: use the response text
        logger.warning("No logprobs available, using response text for scoring")
        response_lower = response_text.lower().strip()
        if response_lower.startswith("yes"):
            return 0.85  # High but not certain
        elif response_lower.startswith("no"):
            return 0.15  # Low but not certain
        else:
            return 0.5  # Uncertain
    
    def score(self, question: str, context: str) -> SufficiencyScore:
        """
        Compute the probabilistic sufficiency score for a question-context pair.
        
        Args:
            question: The question to be answered
            context: The retrieved context
            
        Returns:
            SufficiencyScore with continuous score and metadata
        """
        # Extract log probabilities based on provider
        if self.provider == "openai":
            logprob_yes, logprob_no, response_text, top_tokens = \
                self._extract_logprobs_openai(question, context)
        elif self.provider == "anthropic":
            logprob_yes, logprob_no, response_text, top_tokens = \
                self._extract_logprobs_anthropic(question, context)
        elif self.provider == "gemini":
            logprob_yes, logprob_no, response_text, top_tokens = \
                self._extract_logprobs_gemini(question, context)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
        
        # Compute continuous score
        score = self._compute_score(logprob_yes, logprob_no, response_text)
        
        return SufficiencyScore(
            score=score,
            logprob_yes=logprob_yes,
            logprob_no=logprob_no,
            top_tokens=top_tokens,
            response=response_text,
            model=self.config.model_name,
            prompt_version="v1"
        )
    
    def score_batch(
        self,
        questions: List[str],
        contexts: List[str],
        show_progress: bool = True
    ) -> List[SufficiencyScore]:
        """
        Score multiple question-context pairs.
        
        Args:
            questions: List of questions
            contexts: List of contexts (same length as questions)
            show_progress: Whether to show progress bar
            
        Returns:
            List of SufficiencyScore objects
        """
        if len(questions) != len(contexts):
            raise ValueError("Questions and contexts must have same length")
        
        scores = []
        iterator = zip(questions, contexts)
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(list(iterator), desc="Scoring")
            except ImportError:
                pass
        
        for question, context in iterator:
            try:
                score = self.score(question, context)
                scores.append(score)
            except Exception as e:
                logger.error(f"Error scoring: {e}")
                # Return uncertain score on error
                scores.append(SufficiencyScore(
                    score=0.5,
                    response="ERROR",
                    model=self.config.model_name
                ))
        
        return scores


class MockScorer:
    """
    Mock scorer for testing without API calls.
    Uses simple heuristics to estimate sufficiency.
    """
    
    def __init__(self, config: Optional[ScorerConfig] = None):
        self.config = config or ScorerConfig()
    
    def score(self, question: str, context: str) -> SufficiencyScore:
        """
        Estimate sufficiency using simple heuristics.
        
        Heuristics:
        - Context length relative to question
        - Keyword overlap
        - Presence of answer-like patterns
        """
        # Simple heuristics
        q_words = set(question.lower().split())
        c_words = set(context.lower().split())
        
        # Remove stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "who", 
                      "where", "when", "why", "how", "in", "on", "at", "to", "for"}
        q_words = q_words - stop_words
        c_words = c_words - stop_words
        
        # Overlap ratio
        if len(q_words) > 0:
            overlap = len(q_words & c_words) / len(q_words)
        else:
            overlap = 0
        
        # Context length factor
        length_factor = min(len(context.split()) / 100, 1.0)
        
        # Combine factors
        score = 0.3 * overlap + 0.4 * length_factor + 0.3 * (0.5 if len(context) > 50 else 0.2)
        score = max(0.0, min(1.0, score))
        
        return SufficiencyScore(
            score=score,
            response="Yes" if score > 0.5 else "No",
            model="mock"
        )
    
    def score_batch(
        self,
        questions: List[str],
        contexts: List[str],
        show_progress: bool = True
    ) -> List[SufficiencyScore]:
        """Score multiple pairs."""
        return [self.score(q, c) for q, c in zip(questions, contexts)]


def create_scorer(
    config: ScorerConfig,
    api_key: Optional[str] = None,
    use_mock: bool = False
) -> ProbabilisticSufficiencyScorer:
    """
    Factory function to create appropriate scorer.
    
    Args:
        config: ScorerConfig
        api_key: API key for LLM provider
        use_mock: If True, create MockScorer for testing
        
    Returns:
        Scorer instance
    """
    if use_mock:
        return MockScorer(config)
    return ProbabilisticSufficiencyScorer(config, api_key)
