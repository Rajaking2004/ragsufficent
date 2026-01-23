"""
Unit tests for Prob-RAG modules.
"""

import pytest
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prob_rag.config import (
    ProbRAGConfig, RouterConfig, ScorerConfig, GeneratorConfig, RouterState
)
from prob_rag.modules.retriever import PassthroughRetriever, RetrievalResult
from prob_rag.modules.scorer import SufficiencyScore, MockScorer
from prob_rag.modules.router import TrafficLightRouter, RoutingDecision
from prob_rag.modules.generator import MockGenerator, GenerationResult
from prob_rag.data.datasets import RAGSample, SyntheticDatasetLoader
from prob_rag.evaluation.metrics import (
    normalize_answer, compute_exact_match, compute_f1, is_abstention_response
)


class TestConfig:
    """Test configuration classes."""
    
    def test_router_config_validation(self):
        """Test that invalid thresholds are caught."""
        config = ProbRAGConfig(
            router=RouterConfig(tau_low=0.3, tau_high=0.7)
        )
        assert config.validate() == True
    
    def test_router_state_enum(self):
        """Test RouterState enum values."""
        assert RouterState.RED.value == "abstention"
        assert RouterState.YELLOW.value == "hedging"
        assert RouterState.GREEN.value == "standard"


class TestRetriever:
    """Test retriever module."""
    
    def test_passthrough_retriever(self):
        """Test PassthroughRetriever wraps contexts correctly."""
        retriever = PassthroughRetriever()
        
        result = retriever.retrieve(
            query="What is Python?",
            contexts=["Python is a programming language.", "It was created by Guido."]
        )
        
        assert isinstance(result, RetrievalResult)
        assert len(result.contexts) == 2
        assert result.query == "What is Python?"
        assert "Python is a programming language" in result.combined_context
    
    def test_single_context(self):
        """Test with single context string."""
        retriever = PassthroughRetriever()
        
        result = retriever.retrieve(
            query="Test",
            contexts="Single context string"
        )
        
        assert len(result.contexts) == 1


class TestScorer:
    """Test scorer module."""
    
    def test_mock_scorer(self):
        """Test MockScorer returns valid scores."""
        scorer = MockScorer()
        
        score = scorer.score(
            question="What is the capital of France?",
            context="Paris is the capital and largest city of France."
        )
        
        assert isinstance(score, SufficiencyScore)
        assert 0 <= score.score <= 1
        assert score.response in ["Yes", "No"]
    
    def test_batch_scoring(self):
        """Test batch scoring."""
        scorer = MockScorer()
        
        questions = ["Q1?", "Q2?", "Q3?"]
        contexts = ["C1", "C2", "C3"]
        
        scores = scorer.score_batch(questions, contexts)
        
        assert len(scores) == 3
        assert all(isinstance(s, SufficiencyScore) for s in scores)


class TestRouter:
    """Test router module."""
    
    def test_routing_decisions(self):
        """Test routing produces correct states."""
        router = TrafficLightRouter(RouterConfig(tau_low=0.3, tau_high=0.7))
        
        # Low score -> RED
        low_score = SufficiencyScore(score=0.1)
        decision = router.route(low_score)
        assert decision.state == RouterState.RED
        
        # Mid score -> YELLOW
        mid_score = SufficiencyScore(score=0.5)
        decision = router.route(mid_score)
        assert decision.state == RouterState.YELLOW
        
        # High score -> GREEN
        high_score = SufficiencyScore(score=0.9)
        decision = router.route(high_score)
        assert decision.state == RouterState.GREEN
    
    def test_threshold_boundaries(self):
        """Test exact threshold boundaries."""
        router = TrafficLightRouter(RouterConfig(tau_low=0.3, tau_high=0.7))
        
        # Exactly at tau_low -> YELLOW (not RED)
        score = SufficiencyScore(score=0.3)
        decision = router.route(score)
        assert decision.state == RouterState.YELLOW
        
        # Exactly at tau_high -> YELLOW (not GREEN)
        score = SufficiencyScore(score=0.7)
        decision = router.route(score)
        assert decision.state == RouterState.YELLOW
    
    def test_statistics_tracking(self):
        """Test routing statistics."""
        router = TrafficLightRouter(RouterConfig(tau_low=0.3, tau_high=0.7))
        
        scores = [
            SufficiencyScore(score=0.1),  # RED
            SufficiencyScore(score=0.2),  # RED
            SufficiencyScore(score=0.5),  # YELLOW
            SufficiencyScore(score=0.8),  # GREEN
            SufficiencyScore(score=0.9),  # GREEN
        ]
        
        for s in scores:
            router.route(s)
        
        stats = router.get_statistics()
        assert stats["total"] == 5
        assert stats["red_count"] == 2
        assert stats["yellow_count"] == 1
        assert stats["green_count"] == 2


class TestGenerator:
    """Test generator module."""
    
    def test_mock_generator(self):
        """Test MockGenerator produces appropriate responses."""
        generator = MockGenerator()
        
        routing = RoutingDecision(
            state=RouterState.RED,
            score=0.1,
            tau_low=0.3,
            tau_high=0.7,
            confidence=0.9
        )
        
        result = generator.generate(
            question="Test question",
            context="Test context",
            routing=routing
        )
        
        assert isinstance(result, GenerationResult)
        assert result.is_abstention
        assert "cannot" in result.answer.lower() or "insufficient" in result.answer.lower()
    
    def test_hedging_response(self):
        """Test hedging response."""
        generator = MockGenerator()
        
        routing = RoutingDecision(
            state=RouterState.YELLOW,
            score=0.5,
            tau_low=0.3,
            tau_high=0.7,
            confidence=0.5
        )
        
        result = generator.generate("Q?", "Context", routing)
        
        assert result.is_hedged
        assert "appears" in result.answer.lower() or "suggests" in result.answer.lower()


class TestEvaluation:
    """Test evaluation metrics."""
    
    def test_normalize_answer(self):
        """Test answer normalization."""
        assert normalize_answer("The Answer") == "answer"
        assert normalize_answer("  spaces  ") == "spaces"
        assert normalize_answer("UPPERCASE") == "uppercase"
        assert normalize_answer("with, punctuation!") == "with punctuation"
    
    def test_exact_match(self):
        """Test exact match computation."""
        assert compute_exact_match("Paris", "Paris") == True
        assert compute_exact_match("paris", "Paris") == True
        assert compute_exact_match("The Paris", "Paris") == True
        assert compute_exact_match("London", "Paris") == False
    
    def test_exact_match_with_aliases(self):
        """Test exact match with aliases."""
        assert compute_exact_match("NYC", "New York City", ["NYC", "New York"]) == True
        assert compute_exact_match("New York", "New York City", ["NYC", "New York"]) == True
    
    def test_f1_score(self):
        """Test F1 score computation."""
        # Perfect match
        assert compute_f1("hello world", "hello world") == 1.0
        
        # Partial match
        f1 = compute_f1("hello there", "hello world")
        assert 0 < f1 < 1
        
        # No match
        assert compute_f1("abc", "xyz") == 0.0
    
    def test_abstention_detection(self):
        """Test abstention response detection."""
        assert is_abstention_response("I cannot answer this question") == True
        assert is_abstention_response("The context does not contain") == True
        assert is_abstention_response("I don't know") == True
        assert is_abstention_response("The answer is Paris") == False


class TestDatasets:
    """Test dataset loaders."""
    
    def test_synthetic_loader(self):
        """Test synthetic dataset generation."""
        loader = SyntheticDatasetLoader(num_samples=10)
        samples = loader.load()
        
        assert len(samples) == 10
        assert all(isinstance(s, RAGSample) for s in samples)
        assert all(s.question for s in samples)
        assert all(s.contexts for s in samples)
    
    def test_rag_sample(self):
        """Test RAGSample structure."""
        sample = RAGSample(
            id="test_1",
            question="What is Python?",
            contexts=["Python is a programming language."],
            answer="A programming language"
        )
        
        assert sample.combined_context == "Python is a programming language."
        assert "A programming language" in sample.all_answers


class TestIntegration:
    """Integration tests."""
    
    def test_full_pipeline_mock(self):
        """Test full pipeline with mock components."""
        from prob_rag.pipeline import ProbRAGPipeline
        
        config = ProbRAGConfig(
            router=RouterConfig(tau_low=0.3, tau_high=0.7)
        )
        
        pipeline = ProbRAGPipeline(config, use_mock=True)
        
        sample = RAGSample(
            id="test",
            question="What is the capital of France?",
            contexts=["Paris is the capital of France."],
            answer="Paris"
        )
        
        result = pipeline.process_single(sample, evaluate=True)
        
        assert result.retrieval is not None
        assert result.sufficiency is not None
        assert result.routing is not None
        assert result.generation is not None
        assert result.evaluation is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
