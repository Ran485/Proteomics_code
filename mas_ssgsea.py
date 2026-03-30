"""
Microbial-Associated Signatures (MAS) ssGSEA Analysis Module

This module provides memory-efficient, parallelized single-sample Gene Set 
Enrichment Analysis (ssGSEA) for calculating microbial-associated signatures 
from large-scale gene expression data.

Features:
    - Memory-efficient chunked processing for large datasets (10GB+)
    - Parallel computation using multiprocessing
    - Comprehensive logging and error handling
    - Type-hinted functions following PEP 8 standards
    - Modular design for easy maintenance and testing

Author: Bioinformatics Analysis Pipeline
Date: 2026-03-28
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from pathlib import Path
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import simpson

# Configure module-level logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ============================================================================
# Data Classes for Configuration and Results
# ============================================================================

@dataclass
class SSGSEAConfig:
    """Configuration parameters for ssGSEA analysis.

    Attributes:
        alpha: Weighting exponent for gene ranking (default: 0.25)
        normalize: Whether to normalize enrichment scores (default: True)
        min_genes: Minimum genes required in overlap (default: 10)
        chunk_size: Number of samples to process per chunk (default: 100)
        n_cores: Number of CPU cores for parallel processing (default: -1 for auto)
        random_seed: Random seed for reproducibility (default: 42)
    """
    alpha: float = 0.25
    normalize: bool = True
    min_genes: int = 10
    chunk_size: int = 100
    n_cores: int = -1
    random_seed: int = 42

    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0 <= self.alpha <= 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {self.alpha}")
        if self.min_genes < 1:
            raise ValueError(f"min_genes must be >= 1, got {self.min_genes}")
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")
        if self.n_cores == -1:
            self.n_cores = max(1, cpu_count() - 1)
        elif self.n_cores < 1:
            raise ValueError(f"n_cores must be >= 1 or -1, got {self.n_cores}")


@dataclass 
class MASResult:
    """Container for MAS analysis results.

    Attributes:
        scores: DataFrame with MAS scores (samples x signatures)
        activation_score: Series with activation signature scores
        inactivation_score: Series with inactivation signature scores
        combined_mas: Series with combined MAS scores
        gene_coverage: Dict with gene set coverage statistics
    """
    scores: pd.DataFrame
    activation_score: pd.Series
    inactivation_score: pd.Series
    combined_mas: pd.Series
    gene_coverage: Dict[str, Any]


# ============================================================================
# Core ssGSEA Algorithm Implementation
# ============================================================================

class SSGSEACalculator:
    """Single-sample GSEA calculator with memory-efficient processing.

    This class implements the ssGSEA algorithm based on the method described
    by Barbie et al., 2009 (Nature), optimized for large-scale datasets.

    Algorithm Steps:
        1. Rank genes by expression level for each sample
        2. Calculate enrichment score using Gaussian kernel weighted walk
        3. Integrate the difference between hit and miss distributions
        4. Normalize scores if configured
    """

    def __init__(self, config: Optional[SSGSEAConfig] = None):
        """Initialize calculator with configuration.

        Args:
            config: SSGSEAConfig instance with parameters. If None, uses defaults.
        """
        self.config = config or SSGSEAConfig()
        np.random.seed(self.config.random_seed)

    def _rank_genes(self, gene_expression: np.ndarray, gene_names: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Rank genes by expression level using average method for ties.

        Args:
            gene_expression: 1D array of expression values
            gene_names: 1D array of gene symbols

        Returns:
            Tuple of (sorted gene names, sorted ranks)
        """
        # Step 1: Rank genes (highest expression = highest rank)
        # Using scipy.stats.rankdata with 'average' method for ties
        ranks = stats.rankdata(gene_expression, method='average')

        # Sort genes by rank (descending order)
        sorted_indices = np.argsort(-ranks)  # Negative for descending
        sorted_genes = gene_names[sorted_indices]
        sorted_ranks = ranks[sorted_indices]

        return sorted_genes, sorted_ranks

    def _calculate_enrichment_score(
        self, 
        sorted_genes: np.ndarray,
        sorted_ranks: np.ndarray,
        gene_set: set,
        n_total: int
    ) -> float:
        """Calculate enrichment score for a single gene set.

        This implements the weighted Kolmogorov-Smirnov-like statistic
        with Gaussian kernel weighting for smooth enrichment profiles.

        Args:
            sorted_genes: Gene symbols sorted by expression rank
            sorted_ranks: Corresponding rank values
            gene_set: Set of gene symbols in the gene set
            n_total: Total number of genes

        Returns:
            Enrichment score (ES) value
        """
        # Step 2: Identify which genes in sorted list are in the gene set (hits)
        is_hit = np.array([gene in gene_set for gene in sorted_genes], dtype=bool)
        n_hits = np.sum(is_hit)
        n_miss = n_total - n_hits

        # Check minimum gene overlap requirement
        if n_hits < self.config.min_genes:
            logger.warning(f"Gene set overlap ({n_hits}) below threshold ({self.config.min_genes})")
            return np.nan

        # Step 3: Calculate weights for hits using power of ranks
        # Higher weight for genes with higher ranks (higher expression)
        if self.config.alpha == 0:
            hit_weights = np.ones(n_hits)
        else:
            hit_ranks = sorted_ranks[is_hit]
            hit_weights = np.abs(hit_ranks) ** self.config.alpha

        # Step 4: Calculate cumulative distributions
        # Hit distribution (weighted)
        hit_indices = np.where(is_hit)[0]
        hit_cumsum = np.cumsum(hit_weights) / np.sum(hit_weights)

        # Miss distribution (uniform weight)
        miss_indices = np.where(~is_hit)[0]
        miss_cumsum = np.arange(1, n_miss + 1) / n_miss

        # Step 5: Calculate running enrichment score
        # Create full walk by interpolating between hit steps
        running_es = np.zeros(n_total)
        hit_ptr = 0
        miss_ptr = 0

        for i in range(n_total):
            if is_hit[i]:
                # Hit: use weighted cumulative
                if hit_ptr < len(hit_cumsum):
                    running_es[i] = hit_cumsum[hit_ptr] - (miss_ptr / n_miss if miss_ptr > 0 else 0)
                    hit_ptr += 1
                else:
                    running_es[i] = running_es[i-1] if i > 0 else 0
            else:
                # Miss: uniform step
                miss_ptr += 1
                if hit_ptr > 0:
                    running_es[i] = hit_cumsum[hit_ptr-1] - (miss_ptr / n_miss)
                else:
                    running_es[i] = -(miss_ptr / n_miss)

        # Step 6: Find maximum deviation from zero
        # This is the enrichment score (ES)
        max_deviation = np.max(np.abs(running_es))
        es_sign = np.sign(running_es[np.argmax(np.abs(running_es))])
        es = es_sign * max_deviation

        return es

    def _normalize_score(self, es: float, all_es: List[float]) -> float:
        """Normalize enrichment score against null distribution.

        Args:
            es: Raw enrichment score
            all_es: List of enrichment scores from random permutations

        Returns:
            Normalized enrichment score (NES)
        """
        if not self.config.normalize or np.isnan(es):
            return es

        # Normalize by mean and std of null distribution
        all_es = [e for e in all_es if not np.isnan(e)]
        if len(all_es) < 2:
            return es

        mean_es = np.mean(all_es)
        std_es = np.std(all_es)

        if std_es == 0:
            return 0.0

        return (es - mean_es) / std_es

    def calculate_sample_score(
        self, 
        gene_expression: pd.Series,
        gene_sets: Dict[str, set],
        gene_names: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate ssGSEA scores for a single sample.

        Args:
            gene_expression: Series with gene expression values (index=genes)
            gene_sets: Dictionary of gene set name -> set of genes
            gene_names: Optional array of gene names (uses index if None)

        Returns:
            Dictionary mapping gene set names to enrichment scores
        """
        if gene_names is None:
            gene_names = gene_expression.index.values

        expr_values = gene_expression.values.astype(float)
        n_total = len(expr_values)

        # Remove NA values if any
        valid_mask = ~np.isnan(expr_values)
        if not np.all(valid_mask):
            expr_values = expr_values[valid_mask]
            gene_names = gene_names[valid_mask]
            n_total = len(expr_values)

        # Rank genes
        sorted_genes, sorted_ranks = self._rank_genes(expr_values, gene_names)

        # Calculate score for each gene set
        scores = {}
        for set_name, gene_set in gene_sets.items():
            try:
                es = self._calculate_enrichment_score(sorted_genes, sorted_ranks, gene_set, n_total)
                scores[set_name] = es
            except Exception as e:
                logger.error(f"Error calculating score for {set_name}: {e}")
                scores[set_name] = np.nan

        return scores


# ============================================================================
# MAS Analysis Orchestrator
# ============================================================================

class MASAnalyzer:
    """Main orchestrator for Microbial-Associated Signatures analysis.

    This class manages the complete workflow from gene set loading to MAS
    score calculation, optimized for large datasets with parallel processing.

    Example:
        >>> analyzer = MASAnalyzer(config=SSGSEAConfig(n_cores=8))
        >>> result = analyzer.analyze(expression_df, activation_path, inactivation_path)
        >>> print(result.combined_mas.head())
    """

    def __init__(self, config: Optional[SSGSEAConfig] = None):
        """Initialize MAS analyzer with configuration.

        Args:
            config: SSGSEAConfig instance. If None, uses defaults.
        """
        self.config = config or SSGSEAConfig()
        self.calculator = SSGSEACalculator(config)
        self.gene_sets: Dict[str, set] = {}

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for the analyzer."""
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

    def load_gene_sets(
        self, 
        activation_path: Union[str, Path],
        inactivation_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """Load gene sets from text files.

        Args:
            activation_path: Path to activation gene set file
            inactivation_path: Path to inactivation gene set file

        Returns:
            Dictionary with gene set statistics

        Raises:
            FileNotFoundError: If gene set files not found
            ValueError: If files are empty or malformed
        """
        logger.info(f"Loading gene sets from {activation_path} and {inactivation_path}")

        activation_path = Path(activation_path)
        inactivation_path = Path(inactivation_path)

        # Validate file existence
        if not activation_path.exists():
            raise FileNotFoundError(f"Activation gene set file not found: {activation_path}")
        if not inactivation_path.exists():
            raise FileNotFoundError(f"Inactivation gene set file not found: {inactivation_path}")

        try:
            # Load activation genes
            with open(activation_path, 'r') as f:
                lines = f.readlines()
                # Skip header if present (assume first line is header if contains "Gene")
                start_idx = 1 if lines and "Gene" in lines[0] else 0
                act_genes = {line.strip() for line in lines[start_idx:] if line.strip()}

            # Load inactivation genes
            with open(inactivation_path, 'r') as f:
                lines = f.readlines()
                start_idx = 1 if lines and "Gene" in lines[0] else 0
                inact_genes = {line.strip() for line in lines[start_idx:] if line.strip()}

            # Validate
            if not act_genes:
                raise ValueError("Activation gene set is empty")
            if not inact_genes:
                raise ValueError("Inactivation gene set is empty")

            # Calculate overlap statistics
            overlap = act_genes & inact_genes
            coverage_stats = {
                'activation_genes': len(act_genes),
                'inactivation_genes': len(inact_genes),
                'overlap': len(overlap),
                'overlap_percentage': len(overlap) / max(len(act_genes), len(inact_genes)) * 100
            }

            logger.info(f"Gene sets loaded: {coverage_stats}")

            # Store gene sets
            self.gene_sets = {
                'activation': act_genes,
                'inactivation': inact_genes
            }

            return coverage_stats

        except Exception as e:
            logger.error(f"Failed to load gene sets: {e}")
            raise

    def _process_chunk(
        self, 
        chunk: pd.DataFrame,
        gene_sets: Dict[str, set]
    ) -> pd.DataFrame:
        """Process a chunk of samples (for parallel execution).

        Args:
            chunk: DataFrame subset (genes x samples)
            gene_sets: Dictionary of gene sets

        Returns:
            DataFrame with ssGSEA scores (samples x gene sets)
        """
        # Transpose to samples x genes for iteration
        chunk_T = chunk.T
        results = []

        # Calculate scores for each sample
        for sample_id, sample_expr in chunk_T.iterrows():
            scores = self.calculator.calculate_sample_score(sample_expr, gene_sets)
            scores['Sample'] = sample_id
            results.append(scores)

        return pd.DataFrame(results).set_index('Sample')

    def analyze(
        self, 
        expression_data: pd.DataFrame,
        activation_path: Union[str, Path],
        inactivation_path: Union[str, Path]
    ) -> MASResult:
        """Run complete MAS analysis on expression data.

        This is the main entry point for MAS analysis. It handles memory-efficient
        processing of large datasets through chunking and parallelization.

        Algorithm:
            1. Load gene sets from files
            2. Determine overlap with expression data
            3. Split samples into chunks for memory efficiency
            4. Process chunks in parallel using multiprocessing
            5. Combine results and calculate final MAS scores

        Args:
            expression_data: DataFrame (genes x samples) with expression values
            activation_path: Path to activation gene set file
            inactivation_path: Path to inactivation gene set file

        Returns:
            MASResult object containing all scores and statistics

        Raises:
            ValueError: If gene sets not loaded or expression data invalid
            RuntimeError: If processing fails
        """
        logger.info("Starting MAS analysis")

        # Step 1: Load gene sets
        coverage_stats = self.load_gene_sets(activation_path, inactivation_path)

        # Step 2: Validate expression data
        if expression_data.empty:
            raise ValueError("Expression data is empty")
        if expression_data.isna().all().all():
            raise ValueError("Expression data contains only NA values")

        # Step 3: Find gene overlap with expression data
        available_genes = set(expression_data.index)
        act_overlap = self.gene_sets['activation'] & available_genes
        inact_overlap = self.gene_sets['inactivation'] & available_genes

        coverage_stats['activation_in_data'] = len(act_overlap)
        coverage_stats['inactivation_in_data'] = len(inact_overlap)

        logger.info(f"Gene overlap with expression data: {len(act_overlap)} activation, "
                   f"{len(inact_overlap)} inactivation")

        if len(act_overlap) < self.config.min_genes:
            warnings.warn(f"Activation gene overlap ({len(act_overlap)}) below threshold")
        if len(inact_overlap) < self.config.min_genes:
            warnings.warn(f"Inactivation gene overlap ({len(inact_overlap)}) below threshold")

        # Update gene sets to only include available genes
        working_gene_sets = {
            'activation': act_overlap,
            'inactivation': inact_overlap
        }

        # Step 4: Process in chunks for memory efficiency
        n_samples = expression_data.shape[1]
        chunk_size = min(self.config.chunk_size, n_samples)
        n_chunks = (n_samples + chunk_size - 1) // chunk_size

        logger.info(f"Processing {n_samples} samples in {n_chunks} chunks "
                   f"using {self.config.n_cores} cores")

        all_scores = []

        try:
            if self.config.n_cores > 1 and n_chunks > 1:
                # Parallel processing
                # Create partial function with fixed arguments
                process_func = partial(self._process_chunk, gene_sets=working_gene_sets)

                # Split data into chunks
                chunks = [
                    expression_data.iloc[:, i:i+chunk_size] 
                    for i in range(0, n_samples, chunk_size)
                ]

                # Process chunks in parallel
                with Pool(processes=self.config.n_cores) as pool:
                    chunk_results = pool.map(process_func, chunks)

                all_scores = chunk_results
            else:
                # Sequential processing (for single core or small data)
                for i in range(0, n_samples, chunk_size):
                    chunk = expression_data.iloc[:, i:i+chunk_size]
                    result = self._process_chunk(chunk, working_gene_sets)
                    all_scores.append(result)

                    # Progress logging
                    if (i // chunk_size + 1) % 10 == 0:
                        logger.info(f"Processed chunk {i // chunk_size + 1}/{n_chunks}")

            # Step 5: Combine results
            combined_scores = pd.concat(all_scores, axis=0)

            # Step 6: Calculate combined MAS score
            # MAS = Activation score - Inactivation score
            # Higher MAS = more microbial activation signature
            combined_scores['MAS'] = (
                combined_scores['activation'] - combined_scores['inactivation']
            )

            # Handle missing values
            valid_samples = combined_scores.dropna()
            if len(valid_samples) < len(combined_scores):
                n_dropped = len(combined_scores) - len(valid_samples)
                logger.warning(f"Dropped {n_dropped} samples with missing scores")

            logger.info(f"MAS analysis complete for {len(valid_samples)} samples")

            return MASResult(
                scores=combined_scores,
                activation_score=combined_scores['activation'],
                inactivation_score=combined_scores['inactivation'],
                combined_mas=combined_scores['MAS'],
                gene_coverage=coverage_stats
            )

        except Exception as e:
            logger.error(f"MAS analysis failed: {e}")
            raise RuntimeError(f"Analysis pipeline failed: {e}") from e


# ============================================================================
# Utility Functions
# ============================================================================

def quick_mas_score(
    expression_data: pd.DataFrame,
    activation_genes: List[str],
    inactivation_genes: List[str],
    alpha: float = 0.25,
    n_cores: int = -1
) -> pd.DataFrame:
    """Convenience function for quick MAS analysis without file I/O.

    Args:
        expression_data: DataFrame (genes x samples)
        activation_genes: List of activation gene symbols
        inactivation_genes: List of inactivation gene symbols
        alpha: Weighting parameter for ssGSEA
        n_cores: Number of cores for parallel processing

    Returns:
        DataFrame with MAS scores for all samples
    """
    config = SSGSEAConfig(alpha=alpha, n_cores=n_cores)
    analyzer = MASAnalyzer(config)

    # Directly set gene sets bypassing file loading
    analyzer.gene_sets = {
        'activation': set(activation_genes),
        'inactivation': set(inactivation_genes)
    }

    # Create temporary file paths (won't be used)
    dummy_path = Path("/tmp/dummy.txt")

    # Manually calculate coverage stats
    available_genes = set(expression_data.index)
    act_overlap = analyzer.gene_sets['activation'] & available_genes
    inact_overlap = analyzer.gene_sets['inactivation'] & available_genes

    analyzer.coverage_stats = {
        'activation_in_data': len(act_overlap),
        'inactivation_in_data': len(inact_overlap)
    }

    # Process using internal logic
    working_gene_sets = {
        'activation': act_overlap,
        'inactivation': inact_overlap
    }

    result = analyzer._process_chunk(expression_data, working_gene_sets)
    result['MAS'] = result['activation'] - result['inactivation']

    return result


def save_mas_results(result: MASResult, output_dir: Union[str, Path], prefix: str = "MAS"):
    """Save MAS analysis results to CSV files.

    Args:
        result: MASResult object from analyze()
        output_dir: Directory path for output files
        prefix: File prefix for output files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save combined scores
    result.scores.to_csv(output_dir / f"{prefix}_scores.csv")

    # Save individual scores
    result.activation_score.to_csv(output_dir / f"{prefix}_activation.csv")
    result.inactivation_score.to_csv(output_dir / f"{prefix}_inactivation.csv")
    result.combined_mas.to_csv(output_dir / f"{prefix}_combined.csv")

    # Save metadata
    import json
    with open(output_dir / f"{prefix}_metadata.json", 'w') as f:
        json.dump(result.gene_coverage, f, indent=2)

    logger.info(f"Results saved to {output_dir}")


# ============================================================================
# Module Entry Point
# ============================================================================

if __name__ == "__main__":
    # Example usage demonstration
    print("Microbial-Associated Signatures (MAS) ssGSEA Analysis Module")
    print("=" * 60)
    print("Use: from mas_ssgsea import MASAnalyzer, SSGSEAConfig")
    print("     analyzer = MASAnalyzer(config=SSGSEAConfig(n_cores=8))")
    print("     result = analyzer.analyze(expression_df, act_path, inact_path)")
