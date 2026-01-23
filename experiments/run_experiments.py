"""
Experiment Runner for Prob-RAG

Run comprehensive experiments comparing Prob-RAG against baselines
on multiple datasets.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import json

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prob_rag.config import ProbRAGConfig, RouterConfig, get_preset
from prob_rag.pipeline import ProbRAGPipeline, BaselinePipeline, compare_experiments
from prob_rag.data.datasets import DATASETS, load_dataset_samples
from prob_rag.evaluation.metrics import AggregateMetrics

try:
    from visualization import (
        plot_accuracy_vs_coverage,
        plot_routing_distribution,
        plot_calibration_curve,
        plot_threshold_sensitivity,
        create_experiment_report
    )
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_single_experiment(
    config: ProbRAGConfig,
    dataset_name: str,
    split: str = "validation",
    num_samples: int = 100,
    use_mock: bool = True,
    output_dir: str = "./results"
) -> Dict[str, Any]:
    """
    Run a single experiment with Prob-RAG and baseline.
    
    Args:
        config: ProbRAGConfig
        dataset_name: Dataset to use
        split: Data split
        num_samples: Number of samples
        use_mock: Whether to use mock modules (no API calls)
        output_dir: Output directory for results
        
    Returns:
        Comparison results dictionary
    """
    os.makedirs(output_dir, exist_ok=True)
    
    experiment_id = f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"=" * 60)
    logger.info(f"Running experiment: {experiment_id}")
    logger.info(f"Dataset: {dataset_name}, Samples: {num_samples}")
    logger.info(f"Thresholds: tau_low={config.router.tau_low}, tau_high={config.router.tau_high}")
    logger.info(f"=" * 60)
    
    # Run Prob-RAG
    logger.info("\n🔵 Running Prob-RAG pipeline...")
    prob_rag = ProbRAGPipeline(config, use_mock=use_mock)
    prob_rag_results = prob_rag.run_experiment(
        dataset_name=dataset_name,
        split=split,
        num_samples=num_samples,
        experiment_name=f"prob_rag_{experiment_id}"
    )
    
    # Run Baseline
    logger.info("\n🔴 Running Baseline pipeline...")
    baseline = BaselinePipeline(config, use_mock=use_mock)
    baseline_results = baseline.run_experiment(
        dataset_name=dataset_name,
        split=split,
        num_samples=num_samples,
        experiment_name=f"baseline_{experiment_id}"
    )
    
    # Compare results
    comparison = compare_experiments(prob_rag_results, baseline_results)
    
    # Save results
    prob_rag_results.save(os.path.join(output_dir, f"prob_rag_{experiment_id}.json"))
    baseline_results.save(os.path.join(output_dir, f"baseline_{experiment_id}.json"))
    
    with open(os.path.join(output_dir, f"comparison_{experiment_id}.json"), 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    
    pr = comparison["prob_rag"]
    bl = comparison["baseline"]
    imp = comparison["improvements"]
    
    logger.info(f"\n{'Metric':<25} {'Prob-RAG':<15} {'Baseline':<15} {'Improvement':<15}")
    logger.info("-" * 70)
    logger.info(f"{'Accuracy':<25} {pr['accuracy']:.3f}{'':>10} {bl['accuracy']:.3f}{'':>10} {imp['accuracy']:+.1f}%")
    logger.info(f"{'Selective Accuracy':<25} {pr['selective_accuracy']:.3f}{'':>10} {bl['selective_accuracy']:.3f}{'':>10} {imp['selective_accuracy']:+.1f}%")
    logger.info(f"{'Coverage':<25} {pr['coverage']:.3f}{'':>10} {bl['coverage']:.3f}{'':>10} {imp['coverage_change']:+.1f}%")
    logger.info(f"{'Hallucination Rate':<25} {pr['hallucination_rate']:.3f}{'':>10} {bl['hallucination_rate']:.3f}{'':>10} {imp['hallucination_reduction']:+.1f}%")
    
    logger.info(f"\nProb-RAG Routing Distribution:")
    routing = comparison["prob_rag_routing"]
    logger.info(f"  🔴 Abstention: {routing.get('red_ratio', 0)*100:.1f}%")
    logger.info(f"  🟡 Hedging:    {routing.get('yellow_ratio', 0)*100:.1f}%")
    logger.info(f"  🟢 Standard:   {routing.get('green_ratio', 0)*100:.1f}%")
    
    return comparison


def run_threshold_sweep(
    config: ProbRAGConfig,
    dataset_name: str,
    num_samples: int = 100,
    use_mock: bool = True,
    output_dir: str = "./results"
) -> List[Dict[str, Any]]:
    """
    Run experiments with different threshold configurations.
    
    Sweeps over different tau_low and tau_high values to find
    optimal configuration.
    """
    results = []
    
    # Threshold configurations to test
    configs = [
        (0.2, 0.6),  # Aggressive
        (0.3, 0.7),  # Default
        (0.4, 0.8),  # Conservative
        (0.25, 0.75),  # Balanced
        (0.35, 0.65),  # Narrow yellow band
    ]
    
    logger.info(f"\n{'='*60}")
    logger.info("THRESHOLD SWEEP EXPERIMENT")
    logger.info(f"Testing {len(configs)} configurations")
    logger.info(f"{'='*60}")
    
    for tau_low, tau_high in configs:
        logger.info(f"\n--- Testing tau_low={tau_low}, tau_high={tau_high} ---")
        
        # Update config
        test_config = ProbRAGConfig(
            router=RouterConfig(tau_low=tau_low, tau_high=tau_high),
            scorer=config.scorer,
            generator=config.generator,
            openai_api_key=config.openai_api_key
        )
        
        # Run experiment
        comparison = run_single_experiment(
            test_config,
            dataset_name,
            num_samples=num_samples,
            use_mock=use_mock,
            output_dir=output_dir
        )
        
        results.append({
            "tau_low": tau_low,
            "tau_high": tau_high,
            "comparison": comparison
        })
    
    # Find best configuration
    best_idx = max(
        range(len(results)),
        key=lambda i: results[i]["comparison"]["prob_rag"]["selective_accuracy"]
    )
    best = results[best_idx]
    
    logger.info(f"\n{'='*60}")
    logger.info("BEST CONFIGURATION")
    logger.info(f"tau_low={best['tau_low']}, tau_high={best['tau_high']}")
    logger.info(f"Selective Accuracy: {best['comparison']['prob_rag']['selective_accuracy']:.3f}")
    logger.info(f"{'='*60}")
    
    # Save sweep results
    with open(os.path.join(output_dir, "threshold_sweep.json"), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def run_multi_dataset_experiment(
    config: ProbRAGConfig,
    datasets: List[str],
    num_samples: int = 100,
    use_mock: bool = True,
    output_dir: str = "./results"
) -> Dict[str, Dict[str, Any]]:
    """
    Run experiments across multiple datasets.
    """
    all_results = {}
    
    logger.info(f"\n{'='*60}")
    logger.info("MULTI-DATASET EXPERIMENT")
    logger.info(f"Datasets: {datasets}")
    logger.info(f"{'='*60}")
    
    for dataset_name in datasets:
        logger.info(f"\n>>> Processing {dataset_name}")
        
        try:
            comparison = run_single_experiment(
                config,
                dataset_name,
                num_samples=num_samples,
                use_mock=use_mock,
                output_dir=output_dir
            )
            all_results[dataset_name] = comparison
        except Exception as e:
            logger.error(f"Error on {dataset_name}: {e}")
            all_results[dataset_name] = {"error": str(e)}
    
    # Summary table
    logger.info(f"\n{'='*60}")
    logger.info("MULTI-DATASET SUMMARY")
    logger.info(f"{'='*60}")
    
    logger.info(f"\n{'Dataset':<20} {'Accuracy':<12} {'Sel.Acc':<12} {'Coverage':<12} {'Improv.':<12}")
    logger.info("-" * 68)
    
    for dataset_name, result in all_results.items():
        if "error" in result:
            logger.info(f"{dataset_name:<20} ERROR: {result['error'][:30]}")
        else:
            pr = result["prob_rag"]
            imp = result["improvements"]
            logger.info(
                f"{dataset_name:<20} {pr['accuracy']:.3f}{'':>8} "
                f"{pr['selective_accuracy']:.3f}{'':>8} "
                f"{pr['coverage']:.3f}{'':>8} "
                f"{imp['selective_accuracy']:+.1f}%"
            )
    
    # Save combined results
    with open(os.path.join(output_dir, "multi_dataset_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    return all_results


def main():
    """Main entry point for experiments."""
    parser = argparse.ArgumentParser(description="Run Prob-RAG experiments")
    
    parser.add_argument(
        "--experiment",
        choices=["single", "sweep", "multi"],
        default="single",
        help="Type of experiment to run"
    )
    parser.add_argument(
        "--dataset",
        default="synthetic",
        help="Dataset name (hotpotqa, musique, triviaqa, synthetic)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["synthetic"],
        help="Datasets for multi-dataset experiment"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples per experiment"
    )
    parser.add_argument(
        "--tau-low",
        type=float,
        default=0.3,
        help="Lower threshold for routing"
    )
    parser.add_argument(
        "--tau-high",
        type=float,
        default=0.7,
        help="Upper threshold for routing"
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Use real API (requires OPENAI_API_KEY)"
    )
    parser.add_argument(
        "--output-dir",
        default="./results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--preset",
        choices=["default", "conservative", "aggressive", "high_precision"],
        default=None,
        help="Use a preset configuration"
    )
    
    args = parser.parse_args()
    
    # Create config
    if args.preset:
        config = get_preset(args.preset)
    else:
        config = ProbRAGConfig(
            router=RouterConfig(
                tau_low=args.tau_low,
                tau_high=args.tau_high
            )
        )
    
    use_mock = not args.use_api
    
    if use_mock:
        logger.info("Running in MOCK mode (no API calls)")
    else:
        if not config.openai_api_key:
            logger.error("OPENAI_API_KEY required for API mode")
            return
        logger.info("Running with REAL API calls")
    
    # Run experiment
    if args.experiment == "single":
        run_single_experiment(
            config,
            args.dataset,
            num_samples=args.num_samples,
            use_mock=use_mock,
            output_dir=args.output_dir
        )
    elif args.experiment == "sweep":
        run_threshold_sweep(
            config,
            args.dataset,
            num_samples=args.num_samples,
            use_mock=use_mock,
            output_dir=args.output_dir
        )
    elif args.experiment == "multi":
        run_multi_dataset_experiment(
            config,
            args.datasets,
            num_samples=args.num_samples,
            use_mock=use_mock,
            output_dir=args.output_dir
        )
    
    logger.info("\n✅ Experiment complete!")


if __name__ == "__main__":
    main()
