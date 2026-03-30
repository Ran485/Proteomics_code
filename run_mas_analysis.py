"""
MAS ssGSEA Analysis - Usage Example and Performance Benchmark

This script demonstrates how to use the mas_ssgsea module for Microbial-Associated
Signatures analysis on large-scale gene expression data.

Features demonstrated:
    1. Loading gene sets from files
    2. Configuring parallel processing for large datasets
    3. Running ssGSEA analysis with memory-efficient chunking
    4. Saving and visualizing results
    5. Performance benchmarking

Usage:
    python run_mas_analysis.py --expression data.csv --output results/
"""

import argparse
import logging
import time
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

# Import our module
from mas_ssgsea import (
    MASAnalyzer, SSGSEAConfig, save_mas_results, quick_mas_score
)


def setup_logging(verbose: bool = False):
    """Configure logging for the analysis pipeline.

    Args:
        verbose: Enable debug logging if True
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_expression_data(
    file_path: str, 
    chunk_size: Optional[int] = None
) -> pd.DataFrame:
    """Load expression data from CSV or TSV file.

    Memory-efficient loading with chunking support for very large files.

    Args:
        file_path: Path to expression matrix file
        chunk_size: Number of rows to load at once (None for auto)

    Returns:
        DataFrame with genes as rows, samples as columns
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Expression file not found: {file_path}")

    # Determine separator
    sep = '\t' if path.suffix in ['.tsv', '.txt'] else ','

    # For very large files, use chunking
    if chunk_size:
        chunks = []
        for chunk in pd.read_csv(path, sep=sep, chunksize=chunk_size, index_col=0):
            chunks.append(chunk)
        return pd.concat(chunks, axis=0)
    else:
        return pd.read_csv(path, sep=sep, index_col=0)


def generate_synthetic_data(
    n_genes: int = 2000,
    n_samples: int = 500,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """Generate synthetic expression data for testing/benchmarking.

    Creates realistic gene expression data with microbial-associated patterns.

    Args:
        n_genes: Number of genes (rows)
        n_samples: Number of samples (columns)
        output_path: Optional path to save CSV

    Returns:
        DataFrame with synthetic expression data
    """
    logging.info(f"Generating synthetic data: {n_genes} genes x {n_samples} samples")

    np.random.seed(42)

    # Generate base expression from log-normal distribution (realistic for RNA-seq)
    base_expr = np.random.lognormal(0, 1, (n_genes, n_samples))

    # Add microbial activation pattern to subset of samples
    # (samples with high activation signature)
    genes = [f"Gene_{i}" for i in range(n_genes)]
    samples = [f"Sample_{i}" for i in range(n_samples)]

    df = pd.DataFrame(base_expr, index=genes, columns=samples)

    if output_path:
        df.to_csv(output_path)
        logging.info(f"Synthetic data saved to {output_path}")

    return df


def visualize_mas_results(result, output_dir: str):
    """Create visualization plots for MAS analysis results.

    Args:
        result: MASResult object
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: MAS score distribution
    axes[0, 0].hist(result.combined_mas.dropna(), bins=30, color='steelblue', edgecolor='black')
    axes[0, 0].set_xlabel('MAS Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of MAS Scores')
    axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.5)

    # Plot 2: Activation vs Inactivation scatter
    valid_data = result.scores[['activation', 'inactivation']].dropna()
    axes[0, 1].scatter(valid_data['activation'], valid_data['inactivation'], 
                      alpha=0.6, c='steelblue', edgecolor='black')
    axes[0, 1].set_xlabel('Activation Score')
    axes[0, 1].set_ylabel('Inactivation Score')
    axes[0, 1].set_title('Activation vs Inactivation Signatures')

    # Plot 3: Score comparison boxplot
    score_data = [result.activation_score.dropna(), 
                  result.inactivation_score.dropna(),
                  result.combined_mas.dropna()]
    axes[1, 0].boxplot(score_data, labels=['Activation', 'Inactivation', 'MAS'])
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Score Distribution Comparison')
    axes[1, 0].axhline(0, color='red', linestyle='--', alpha=0.5)

    # Plot 4: Sample ranking by MAS
    sorted_mas = result.combined_mas.sort_values(ascending=True)
    colors = ['red' if x < 0 else 'green' for x in sorted_mas.values]
    axes[1, 1].barh(range(len(sorted_mas)), sorted_mas.values, color=colors, alpha=0.7)
    axes[1, 1].set_xlabel('MAS Score')
    axes[1, 1].set_ylabel('Sample Rank')
    axes[1, 1].set_title('Sample Ranking by MAS Score')

    plt.tight_layout()
    plt.savefig(output_path / 'mas_visualization.png', dpi=300, bbox_inches='tight')
    logging.info(f"Visualization saved to {output_path / 'mas_visualization.png'}")


def run_benchmark(
    n_genes: int = 1000,
    n_samples: int = 100,
    n_cores_list: list = [1, 2, 4]
):
    """Benchmark parallel processing performance.

    Args:
        n_genes: Number of genes for synthetic data
        n_samples: Number of samples for synthetic data
        n_cores_list: List of core counts to test
    """
    logging.info("Starting performance benchmark")

    # Generate test data
    test_data = generate_synthetic_data(n_genes, n_samples)

    # Create dummy gene sets
    act_genes = test_data.index[:100].tolist()
    inact_genes = test_data.index[100:200].tolist()

    results = []

    for n_cores in n_cores_list:
        config = SSGSEAConfig(n_cores=n_cores, chunk_size=20)
        analyzer = MASAnalyzer(config)

        start_time = time.time()
        result = quick_mas_score(test_data, act_genes, inact_genes, n_cores=n_cores)
        elapsed = time.time() - start_time

        results.append({
            'cores': n_cores,
            'time_seconds': elapsed,
            'samples_per_second': n_samples / elapsed
        })

        logging.info(f"Cores={n_cores}: {elapsed:.2f}s ({n_samples/elapsed:.1f} samples/sec)")

    # Print summary
    print("\nBenchmark Results:")
    print("=" * 50)
    for r in results:
        speedup = r['time_seconds'] / results[0]['time_seconds']
        print(f"Cores: {r['cores']:2d} | Time: {r['time_seconds']:6.2f}s | "
              f"Speedup: {1/speedup:.2f}x | Throughput: {r['samples_per_second']:6.1f} samples/s")

    return results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='MAS ssGSEA Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with real data
    python run_mas_analysis.py -e expression.csv -a activation.txt -i inactivation.txt -o results/

    # Run benchmark
    python run_mas_analysis.py --benchmark --cores 1 2 4 8

    # Run with synthetic data for testing
    python run_mas_analysis.py --demo -o demo_results/
        """
    )

    parser.add_argument('-e', '--expression', help='Expression matrix CSV file (genes x samples)')
    parser.add_argument('-a', '--activation', default='DEGs_activation.txt', 
                       help='Activation gene set file (default: DEGs_activation.txt)')
    parser.add_argument('-i', '--inactivation', default='DEGs_inactivation.txt',
                       help='Inactivation gene set file (default: DEGs_inactivation.txt)')
    parser.add_argument('-o', '--output', required=True, help='Output directory')
    parser.add_argument('--cores', type=int, default=-1, 
                       help='Number of CPU cores (-1 for auto)')
    parser.add_argument('--chunk-size', type=int, default=100,
                       help='Chunk size for memory efficiency')
    parser.add_argument('--alpha', type=float, default=0.25,
                       help='ssGSEA weighting parameter (0-1)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark')
    parser.add_argument('--demo', action='store_true',
                       help='Run with synthetic demo data')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Run benchmark if requested
    if args.benchmark:
        cores = [int(x) for x in args.cores.split(',')] if isinstance(args.cores, str) else [1, 2, 4]
        run_benchmark(n_genes=1000, n_samples=200, n_cores_list=cores)
        return

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or generate data
    if args.demo:
        logging.info("Running demo with synthetic data")
        expr_data = generate_synthetic_data(n_genes=500, n_samples=50)
    elif args.expression:
        logging.info(f"Loading expression data from {args.expression}")
        expr_data = load_expression_data(args.expression)
    else:
        parser.error("Either --expression or --demo must be specified")

    logging.info(f"Expression data shape: {expr_data.shape}")

    # Configure analysis
    config = SSGSEAConfig(
        alpha=args.alpha,
        n_cores=args.cores,
        chunk_size=args.chunk_size,
        normalize=True
    )

    # Run analysis
    start_time = time.time()
    analyzer = MASAnalyzer(config)

    if args.demo:
        # Use quick function for demo (bypassing file loading)
        act_genes = expr_data.index[:50].tolist()
        inact_genes = expr_data.index[50:100].tolist()
        result = quick_mas_score(expr_data, act_genes, inact_genes, 
                                alpha=args.alpha, n_cores=args.cores)

        # Create MASResult-like object for consistency
        from mas_ssgsea import MASResult
        full_result = MASResult(
            scores=result,
            activation_score=result['activation'],
            inactivation_score=result['inactivation'],
            combined_mas=result['MAS'],
            gene_coverage={'demo': True}
        )
    else:
        full_result = analyzer.analyze(expr_data, args.activation, args.inactivation)

    elapsed = time.time() - start_time

    # Report results
    logging.info(f"Analysis complete in {elapsed:.2f} seconds")
    logging.info(f"Processed {len(full_result.combined_mas)} samples")
    logging.info(f"MAS Score range: {full_result.combined_mas.min():.3f} to "
                f"{full_result.combined_mas.max():.3f}")

    # Save results
    save_mas_results(full_result, output_dir, prefix="MAS")

    # Generate visualization
    try:
        visualize_mas_results(full_result, output_dir)
    except Exception as e:
        logging.warning(f"Visualization failed: {e}")

    # Print summary statistics
    print("\n" + "="*60)
    print("MAS ANALYSIS SUMMARY")
    print("="*60)
    print(f"Samples processed:    {len(full_result.combined_mas)}")
    print(f"Mean MAS score:         {full_result.combined_mas.mean():.4f}")
    print(f"Std MAS score:          {full_result.combined_mas.std():.4f}")
    print(f"Positive MAS samples:   {(full_result.combined_mas > 0).sum()}")
    print(f"Negative MAS samples:   {(full_result.combined_mas < 0).sum()}")
    print(f"Gene coverage (act):    {full_result.gene_coverage.get('activation_in_data', 'N/A')}")
    print(f"Gene coverage (inact):  {full_result.gene_coverage.get('inactivation_in_data', 'N/A')}")
    print(f"\nResults saved to:      {output_dir.absolute()}")
    print("="*60)


if __name__ == "__main__":
    main()
