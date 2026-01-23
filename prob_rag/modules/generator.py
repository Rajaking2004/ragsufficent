"""
Module D: Adaptive Generator

Generates responses based on the routing decision from Module C.
Uses different system prompts for each traffic light state:

    🔴 RED (Abstention): Explicitly states inability to answer
    🟡 YELLOW (Hedging): Answers with uncertainty markers
    🟢 GREEN (Standard): Provides confident, direct answer
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

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

try:
    from groq import Groq
except ImportError:
    Groq = None

from ..config import GeneratorConfig, RouterState
from .router import RoutingDecision


logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Container for generation results."""
    # Core output
    answer: str
    
    # Routing information
    state: RouterState
    score: float
    
    # Generation metadata
    model: str
    prompt_type: str  # "abstention", "hedging", or "standard"
    
    # Token counts
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    
    # Full prompt used (for debugging)
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        preview = self.answer[:50] + "..." if len(self.answer) > 50 else self.answer
        return f"GenerationResult(state={self.state.value}, answer='{preview}')"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "state": self.state.value,
            "score": self.score,
            "model": self.model,
            "prompt_type": self.prompt_type,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "metadata": self.metadata
        }
    
    @property
    def is_abstention(self) -> bool:
        return self.state == RouterState.RED
    
    @property
    def is_hedged(self) -> bool:
        return self.state == RouterState.YELLOW


# User prompt template
USER_PROMPT_TEMPLATE = """### QUESTION
{question}

### CONTEXT
{context}

### ANSWER
Based on the above context, please answer the question."""


# Alternative templates for different domains
USER_PROMPT_TEMPLATE_QA = """Question: {question}

Reference Documents:
{context}

Please provide your answer based on the reference documents above."""


USER_PROMPT_TEMPLATE_FACTUAL = """I need to answer the following question using only the provided documents.

Question: {question}

Documents:
{context}

Answer:"""


class AdaptiveGenerator:
    """
    Module D: Adaptive Generator
    
    Generates responses with prompts adapted to the sufficiency state.
    This enables nuanced responses that match the confidence level.
    """
    
    def __init__(self, config: GeneratorConfig, api_key: Optional[str] = None):
        """
        Initialize the generator.
        
        Args:
            config: GeneratorConfig with model and prompt settings
            api_key: API key for the LLM provider
        """
        self.config = config
        self.api_key = api_key
        self.client = None
        
        # System prompts for each state
        self.prompts = {
            RouterState.RED: config.abstention_prompt,
            RouterState.YELLOW: config.hedging_prompt,
            RouterState.GREEN: config.standard_prompt
        }
        
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
        
        elif "llama" in model_lower or "mixtral" in model_lower or "gemma" in model_lower:
            # Groq models (Llama, Mixtral, Gemma)
            if Groq is None:
                raise ImportError("groq package required. Install with: pip install groq")
            if not self.api_key:
                raise ValueError("Groq API key required for Llama/Mixtral models")
            self.client = Groq(api_key=self.api_key)
            self.provider = "groq"
            
        else:
            logger.warning(f"Unknown model {self.config.model_name}, assuming OpenAI-compatible")
            if OpenAI is None:
                raise ImportError("openai package required")
            self.client = OpenAI(api_key=self.api_key)
            self.provider = "openai"
    
    def _get_system_prompt(self, state: RouterState) -> str:
        """Get the appropriate system prompt for the state."""
        return self.prompts.get(state, self.prompts[RouterState.GREEN])
    
    def _build_user_prompt(
        self,
        question: str,
        context: str,
        template: str = "default"
    ) -> str:
        """Build the user prompt."""
        if template == "qa":
            return USER_PROMPT_TEMPLATE_QA.format(question=question, context=context)
        elif template == "factual":
            return USER_PROMPT_TEMPLATE_FACTUAL.format(question=question, context=context)
        else:
            return USER_PROMPT_TEMPLATE.format(question=question, context=context)
    
    def _generate_openai(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> tuple:
        """Generate response using OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        answer = response.choices[0].message.content.strip()
        prompt_tokens = response.usage.prompt_tokens if response.usage else None
        completion_tokens = response.usage.completion_tokens if response.usage else None
        
        return answer, prompt_tokens, completion_tokens
    
    def _generate_anthropic(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> tuple:
        """Generate response using Anthropic API."""
        response = self.client.messages.create(
            model=self.config.model_name,
            max_tokens=self.config.max_tokens,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        
        answer = response.content[0].text.strip()
        prompt_tokens = response.usage.input_tokens if hasattr(response, 'usage') else None
        completion_tokens = response.usage.output_tokens if hasattr(response, 'usage') else None
        
        return answer, prompt_tokens, completion_tokens
    
    def _generate_gemini(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> tuple:
        """Generate response using Google Gemini API."""
        # Combine system and user prompts for Gemini
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        generation_config = genai.types.GenerationConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
        )
        
        response = self.client.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        
        answer = response.text.strip() if response.text else ""
        
        # Gemini doesn't provide exact token counts in the same way
        prompt_tokens = None
        completion_tokens = None
        
        return answer, prompt_tokens, completion_tokens
    
    def _generate_groq(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> tuple:
        """Generate response using Groq API (Llama/Mixtral models)."""
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        answer = response.choices[0].message.content.strip()
        prompt_tokens = response.usage.prompt_tokens if response.usage else None
        completion_tokens = response.usage.completion_tokens if response.usage else None
        
        return answer, prompt_tokens, completion_tokens
    
    def generate(
        self,
        question: str,
        context: str,
        routing: RoutingDecision,
        template: str = "default"
    ) -> GenerationResult:
        """
        Generate a response based on the routing decision.
        
        Args:
            question: The question to answer
            context: The retrieved context
            routing: RoutingDecision from Module C
            template: Prompt template to use
            
        Returns:
            GenerationResult with answer and metadata
        """
        state = routing.state
        
        # Get appropriate prompts
        system_prompt = self._get_system_prompt(state)
        user_prompt = self._build_user_prompt(question, context, template)
        
        # Generate based on provider
        try:
            if self.provider == "openai":
                answer, prompt_tokens, completion_tokens = \
                    self._generate_openai(system_prompt, user_prompt)
            elif self.provider == "anthropic":
                answer, prompt_tokens, completion_tokens = \
                    self._generate_anthropic(system_prompt, user_prompt)
            elif self.provider == "gemini":
                answer, prompt_tokens, completion_tokens = \
                    self._generate_gemini(system_prompt, user_prompt)
            elif self.provider == "groq":
                answer, prompt_tokens, completion_tokens = \
                    self._generate_groq(system_prompt, user_prompt)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
                
        except Exception as e:
            logger.error(f"Generation error: {e}")
            # Return error response
            answer = f"Error generating response: {str(e)}"
            prompt_tokens = None
            completion_tokens = None
        
        return GenerationResult(
            answer=answer,
            state=state,
            score=routing.score,
            model=self.config.model_name,
            prompt_type=state.value,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            metadata={
                "confidence": routing.confidence,
                "calibrated_score": routing.calibrated_score
            }
        )
    
    def generate_batch(
        self,
        questions: List[str],
        contexts: List[str],
        routings: List[RoutingDecision],
        template: str = "default",
        show_progress: bool = True
    ) -> List[GenerationResult]:
        """
        Generate responses for multiple queries.
        
        Args:
            questions: List of questions
            contexts: List of contexts
            routings: List of routing decisions
            template: Prompt template
            show_progress: Whether to show progress bar
            
        Returns:
            List of GenerationResult objects
        """
        if not (len(questions) == len(contexts) == len(routings)):
            raise ValueError("All inputs must have the same length")
        
        results = []
        iterator = zip(questions, contexts, routings)
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(list(iterator), desc="Generating")
            except ImportError:
                pass
        
        for question, context, routing in iterator:
            result = self.generate(question, context, routing, template)
            results.append(result)
        
        return results
    
    def update_prompt(self, state: RouterState, prompt: str) -> None:
        """
        Update the system prompt for a specific state.
        
        Args:
            state: RouterState to update
            prompt: New system prompt
        """
        self.prompts[state] = prompt
        logger.info(f"Updated {state.value} prompt")
    
    def get_prompt(self, state: RouterState) -> str:
        """Get the current prompt for a state."""
        return self.prompts.get(state, "")


class MockGenerator:
    """
    Mock generator for testing without API calls.
    """
    
    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
    
    def generate(
        self,
        question: str,
        context: str,
        routing: RoutingDecision,
        template: str = "default"
    ) -> GenerationResult:
        """Generate mock response based on routing state."""
        state = routing.state
        
        if state == RouterState.RED:
            answer = "I cannot answer this question based on the provided documents. The context does not contain sufficient information."
        elif state == RouterState.YELLOW:
            answer = f"Based on the available information, it appears that the answer may be related to the context provided, but I cannot be certain. The text suggests a possible answer, but there may be gaps in the information."
        else:
            # Extract something from context for mock answer
            words = context.split()[:20]
            answer = f"Based on the provided context: {' '.join(words)}..."
        
        return GenerationResult(
            answer=answer,
            state=state,
            score=routing.score,
            model="mock",
            prompt_type=state.value,
            prompt_tokens=100,
            completion_tokens=50
        )
    
    def generate_batch(
        self,
        questions: List[str],
        contexts: List[str],
        routings: List[RoutingDecision],
        template: str = "default",
        show_progress: bool = True
    ) -> List[GenerationResult]:
        """Generate mock responses."""
        return [
            self.generate(q, c, r, template)
            for q, c, r in zip(questions, contexts, routings)
        ]


def create_generator(
    config: GeneratorConfig,
    api_key: Optional[str] = None,
    use_mock: bool = False
) -> AdaptiveGenerator:
    """
    Factory function to create appropriate generator.
    
    Args:
        config: GeneratorConfig
        api_key: API key for LLM provider
        use_mock: If True, create MockGenerator for testing
        
    Returns:
        Generator instance
    """
    if use_mock:
        return MockGenerator(config)
    return AdaptiveGenerator(config, api_key)


def analyze_hedging_language(answer: str) -> Dict[str, Any]:
    """
    Analyze the presence of hedging language in an answer.
    
    Args:
        answer: Generated answer text
        
    Returns:
        Dictionary with hedging analysis
    """
    # Hedging markers
    uncertainty_markers = [
        "may", "might", "could", "possibly", "perhaps", "likely",
        "appears", "seems", "suggests", "indicates", "it appears",
        "it seems", "based on", "according to", "uncertain",
        "not clear", "unclear", "ambiguous", "approximately"
    ]
    
    certainty_markers = [
        "definitely", "certainly", "clearly", "obviously",
        "undoubtedly", "is", "are", "was", "were", "will",
        "the answer is", "specifically", "exactly", "precisely"
    ]
    
    answer_lower = answer.lower()
    
    # Count markers
    uncertainty_count = sum(1 for m in uncertainty_markers if m in answer_lower)
    certainty_count = sum(1 for m in certainty_markers if m in answer_lower)
    
    # Hedging score
    total = uncertainty_count + certainty_count
    if total > 0:
        hedging_ratio = uncertainty_count / total
    else:
        hedging_ratio = 0.5  # Neutral
    
    return {
        "uncertainty_count": uncertainty_count,
        "certainty_count": certainty_count,
        "hedging_ratio": hedging_ratio,
        "is_hedged": hedging_ratio > 0.3,
        "found_uncertainty_markers": [m for m in uncertainty_markers if m in answer_lower],
        "found_certainty_markers": [m for m in certainty_markers if m in answer_lower]
    }
