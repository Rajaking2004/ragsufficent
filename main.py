#!/usr/bin/env python
"""
Prob-RAG: Probabilistic Sufficient Context RAG

Main entry point for running the Prob-RAG system.

Usage:
    python main.py --mode demo          # Run a quick demo
    python main.py --mode experiment    # Run experiments
    python main.py --mode interactive   # Interactive mode
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prob_rag import ProbRAGConfig, ProbRAGPipeline
from prob_rag.config import get_preset, RouterConfig
from prob_rag.data.datasets import RAGSample, load_dataset_samples, SyntheticDatasetLoader


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_demo():
    """
    Run a quick demonstration of Prob-RAG.
    """
    print("\n" + "="*70)
    print("🚀 PROB-RAG DEMONSTRATION")
    print("="*70)
    
    # Create config
    config = ProbRAGConfig(
        router=RouterConfig(tau_low=0.3, tau_high=0.7)
    )
    
    # Create pipeline (mock mode for demo)
    pipeline = ProbRAGPipeline(config, use_mock=True)
    
    # Create sample questions
    samples = [
        RAGSample(
            id="demo_1",
            question="What year was Microsoft founded?",
            contexts=["Microsoft Corporation is an American multinational technology corporation. "
                     "It was founded by Bill Gates and Paul Allen on April 4, 1975."],
            answer="1975"
        ),
        RAGSample(
            id="demo_2", 
            question="Who is the CEO of Apple?",
            contexts=["Apple Inc. is headquartered in Cupertino, California. "
                     "The company was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne."],
            answer="Tim Cook"  # Context doesn't have this - insufficient!
        ),
        RAGSample(
            id="demo_3",
            question="What is the population of Tokyo?",
            contexts=["Tokyo is the capital of Japan. It is one of the largest cities in the world. "
                     "The Greater Tokyo Area has approximately 37 million people."],
            answer="37 million"
        ),
    ]
    
    print("\n📊 Processing samples through Prob-RAG pipeline...\n")
    
    for sample in samples:
        print("-" * 60)
        print(f"Question: {sample.question}")
        print(f"Context: {sample.contexts[0][:100]}...")
        
        result = pipeline.process_single(sample, evaluate=True)
        
        # Display routing decision
        routing = result.routing
        emoji = {"abstention": "🔴", "hedging": "🟡", "standard": "🟢"}
        state_emoji = emoji.get(routing.state.value, "❓")
        
        print(f"\n  Sufficiency Score: {result.sufficiency.score:.3f}")
        print(f"  Routing Decision: {state_emoji} {routing.state.value.upper()}")
        print(f"  Confidence: {routing.confidence:.3f}")
        print(f"\n  Generated Answer: {result.generation.answer[:200]}")
        
        if result.evaluation:
            print(f"  Ground Truth: {sample.answer}")
            print(f"  Correct: {'✅' if result.evaluation.is_correct else '❌'}")
    
    print("\n" + "="*70)
    print("📈 ROUTING STATISTICS")
    print("="*70)
    stats = pipeline.get_routing_statistics()
    print(f"  Total samples: {stats['total']}")
    print(f"  🔴 Abstention: {stats['red_ratio']*100:.1f}%")
    print(f"  🟡 Hedging: {stats['yellow_ratio']*100:.1f}%")
    print(f"  🟢 Standard: {stats['green_ratio']*100:.1f}%")
    
    print("\n✅ Demo complete!\n")


def run_interactive():
    """
    Run in interactive mode - ask questions interactively.
    """
    print("\n" + "="*70)
    print("🎯 PROB-RAG INTERACTIVE MODE")
    print("="*70)
    print("Enter questions with context to see Prob-RAG in action.")
    print("Type 'quit' to exit.\n")
    
    config = ProbRAGConfig()
    pipeline = ProbRAGPipeline(config, use_mock=True)
    
    while True:
        try:
            question = input("\n📝 Enter question (or 'quit'): ").strip()
            if question.lower() == 'quit':
                break
            
            context = input("📄 Enter context: ").strip()
            if not context:
                print("Context required!")
                continue
            
            sample = RAGSample(
                id="interactive",
                question=question,
                contexts=[context],
                answer=""  # Unknown
            )
            
            result = pipeline.process_single(sample, evaluate=False)
            
            emoji = {"abstention": "🔴", "hedging": "🟡", "standard": "🟢"}
            state_emoji = emoji.get(result.routing.state.value, "❓")
            
            print(f"\n  📊 Sufficiency Score: {result.sufficiency.score:.3f}")
            print(f"  🚦 Routing: {state_emoji} {result.routing.state.value.upper()}")
            print(f"  💬 Answer: {result.generation.answer}")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def run_experiment_mode(args):
    """
    Run experiment mode with specified parameters.
    """
    from experiments.run_experiments import (
        run_single_experiment,
        run_threshold_sweep,
        run_multi_dataset_experiment
    )
    
    config = ProbRAGConfig(
        router=RouterConfig(tau_low=args.tau_low, tau_high=args.tau_high)
    )
    
    if args.experiment_type == "single":
        run_single_experiment(
            config,
            args.dataset,
            num_samples=args.num_samples,
            use_mock=not args.use_api,
            output_dir=args.output_dir
        )
    elif args.experiment_type == "sweep":
        run_threshold_sweep(
            config,
            args.dataset,
            num_samples=args.num_samples,
            use_mock=not args.use_api,
            output_dir=args.output_dir
        )
    elif args.experiment_type == "multi":
        run_multi_dataset_experiment(
            config,
            ["synthetic", "hotpotqa"] if args.use_api else ["synthetic"],
            num_samples=args.num_samples,
            use_mock=not args.use_api,
            output_dir=args.output_dir
        )


def main():
    parser = argparse.ArgumentParser(
        description="Prob-RAG: Probabilistic Sufficient Context RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode demo              # Quick demonstration
  python main.py --mode interactive       # Interactive Q&A
  python main.py --mode experiment        # Run experiments
  python main.py --mode experiment --experiment-type sweep  # Threshold sweep
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["demo", "interactive", "experiment"],
        default="demo",
        help="Operation mode"
    )
    
    # Experiment parameters
    parser.add_argument("--experiment-type", choices=["single", "sweep", "multi"], default="single")
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--tau-low", type=float, default=0.3)
    parser.add_argument("--tau-high", type=float, default=0.7)
    parser.add_argument("--use-api", action="store_true")
    parser.add_argument("--output-dir", default="./results")
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        run_demo()
    elif args.mode == "interactive":
        run_interactive()
    elif args.mode == "experiment":
        run_experiment_mode(args)


if __name__ == "__main__":
    main()
