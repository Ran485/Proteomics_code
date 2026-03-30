"""
Integrated MAS + GSVA Analysis Module

This module provides integrated single-sample Gene Set Enrichment Analysis (ssGSEA) 
and Gene Set Variation Analysis (GSVA) for microbial-associated signatures and 
standard pathway analysis (Hallmark, KEGG, Reactome).

Features:
    - ssGSEA for custom microbial gene sets (activation/inactivation)
    - GSVA for MSigDB pathways via GSEApy integration
    - Unified interface for both methods
    - Memory-efficient chunked processing for large datasets (10GB+)
    - Parallel computation support
    - Automatic gene set library management (Hallmark, KEGG, Reactome)

Requirements:
    - pandas, numpy, scipy
    - gseapy >= 1.1.0 (for GSVA functionality)

Author: Bioinformatics Analysis Pipeline
Date: 2026-03-28
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from pathlib import Path
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
from functools import partial
from enum import Enum
import tempfile
import numpy as np
import pandas as pd
from scipy import stats

# Import ssGSEA classes from existing module
import sys
import importlib.util
spec = importlib.util.spec_from_file_location("mas_ssgsea", "./mas_ssgsea.py")
mas_ssgsea_module = importlib.util.module_from_spec(spec)
sys.modules["mas_ssgsea"] = mas_ssgsea_module
spec.loader.exec_module(mas_ssgsea_module)

# Re-export ssGSEA classes
SSGSEAConfig = mas_ssgsea_module.SSGSEAConfig
SSGSEACalculator = mas_ssgsea_module.SSGSEACalculator
MASAnalyzer = mas_ssgsea_module.MASAnalyzer
MASResult = mas_ssgsea_module.MASResult
quick_mas_score = mas_ssgsea_module.quick_mas_score
save_mas_results = mas_ssgsea_module.save_mas_results

# Try to import gseapy for GSVA functionality
try:
    import gseapy
    from gseapy import gsva as gseapy_gsva, get_library_name
    GSEAPY_AVAILABLE = True
except ImportError:
    GSEAPY_AVAILABLE = False
    warnings.warn(
        "gseapy not installed. GSVA functionality will be disabled. "
        "Install with: pip install gseapy", 
        ImportWarning
    )

# Configure module-level logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ============================================================================
# Enums and Data Classes
# ============================================================================

class GeneSetCollection(Enum):
    """Enumeration of supported MSigDB gene set collections."""
    HALLMARK = "MSigDB_Hallmark_2020"
    KEGG = "KEGG_2021_Human"
    REACTOME = "Reactome_2022"
    BIOCARTA = "BioCarta_2016"
    GO_BP = "GO_Biological_Process_2021"
    GO_CC = "GO_Cellular_Component_2021"
    GO_MF = "GO_Molecular_Function_2021"


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
class GSVAConfig:
    """Configuration parameters for GSVA analysis via GSEApy.

    Attributes:
        kcdf: Kernel for CDF estimation ("Gaussian", "Poisson", or None)
        weight: Tau parameter for GSVA (default: 1.0)
        mx_diff: Use max difference vs max deviation approach (default: True)
        abs_rnk: Use absolute ranking (default: False)
        min_size: Minimum gene set size (default: 15)
        max_size: Maximum gene set size (default: 500)
        n_cores: Number of parallel threads (default: -1 for auto)
        random_seed: Random seed for reproducibility (default: 42)
        gene_sets: List of gene set collections to use (default: [HALLMARK])
    """
    kcdf: Optional[str] = "Gaussian"
    weight: float = 1.0
    mx_diff: bool = True
    abs_rnk: bool = False
    min_size: int = 15
    max_size: int = 500
    n_cores: int = -1
    random_seed: int = 42
    gene_sets: List[GeneSetCollection] = field(default_factory=lambda: [GeneSetCollection.HALLMARK])

    def __post_init__(self):
        """Validate GSVA configuration."""
        valid_kcdf = ["Gaussian", "Poisson", None, "none"]
        if self.kcdf not in valid_kcdf:
            raise ValueError(f"kcdf must be one of {valid_kcdf}, got {self.kcdf}")
        if self.weight <= 0:
            raise ValueError(f"weight must be positive, got {self.weight}")
        if self.min_size < 1:
            raise ValueError(f"min_size must be >= 1, got {self.min_size}")
        if self.max_size <= self.min_size:
            raise ValueError(f"max_size ({self.max_size}) must be > min_size ({self.min_size})")
        if self.n_cores == -1:
            self.n_cores = max(1, cpu_count() - 1)

        # Check gseapy availability
        if not GSEAPY_AVAILABLE and self.gene_sets:
            raise ImportError(
                "gseapy is required for GSVA analysis. "
                "Install with: pip install gseapy"
            )


@dataclass 
class MASResult:
    """Container for MAS analysis results."""
    scores: pd.DataFrame
    activation_score: pd.Series
    inactivation_score: pd.Series
    combined_mas: pd.Series
    gene_coverage: Dict[str, Any]


@dataclass
class GSVAResult:
    """Container for GSVA analysis results.

    Attributes:
        scores: DataFrame with GSVA scores (pathways x samples)
        gene_sets_used: List of gene set collections used
        pathway_coverage: Dict with pathway statistics per collection
        enrichment_df: Long-format DataFrame for visualization
    """
    scores: pd.DataFrame
    gene_sets_used: List[str]
    pathway_coverage: Dict[str, Any]
    enrichment_df: Optional[pd.DataFrame] = None


@dataclass
class CombinedResult:
    """Container for combined MAS + GSVA analysis results."""
    mas_result: MASResult
    gsva_result: Optional[GSVAResult] = None


# ============================================================================
# Gene Set Library Manager
# ============================================================================

class GeneSetLibraryManager:
    """Manager for MSigDB and other gene set libraries via GSEApy.

    This class handles downloading, caching, and retrieving gene set libraries
    from MSigDB (Hallmark, KEGG, Reactome, etc.) using GSEApy integration.

    Attributes:
        cache_dir: Directory for caching gene set files
        libraries: Dictionary mapping collection names to gene sets
    """

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        """Initialize the gene set library manager.

        Args:
            cache_dir: Directory to cache downloaded gene sets. 
                        If None, uses system temp directory.
        """
        if not GSEAPY_AVAILABLE:
            raise ImportError("gseapy is required for GeneSetLibraryManager")

        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "gseapy_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.libraries: Dict[str, Any] = {}
        self._available_libraries: Optional[List[str]] = None

        logger.info(f"GeneSetLibraryManager initialized with cache: {self.cache_dir}")

    def list_available_libraries(self) -> List[str]:
        """List all available gene set libraries from Enrichr.

        Returns:
            List of available library names

        Note:
            This requires internet connection to fetch from Enrichr.
        """
        if self._available_libraries is None:
            try:
                self._available_libraries = get_library_name()
                logger.info(f"Found {len(self._available_libraries)} available libraries")
            except Exception as e:
                logger.error(f"Failed to fetch library names: {e}")
                self._available_libraries = []
        return self._available_libraries

    def search_libraries(self, keyword: str) -> List[str]:
        """Search for libraries matching keyword.

        Args:
            keyword: Search term (case-insensitive)

        Returns:
            List of matching library names
        """
        all_libs = self.list_available_libraries()
        keyword_lower = keyword.lower()
        matches = [lib for lib in all_libs if keyword_lower in lib.lower()]
        return matches

    def get_gene_sets(self, collection: GeneSetCollection) -> Dict[str, List[str]]:
        """Retrieve gene sets for a specific collection.

        This method fetches the gene set library from Enrichr via GSEApy
        and returns it as a dictionary.

        Args:
            collection: GeneSetCollection enum value

        Returns:
            Dictionary mapping pathway names to lists of genes

        Raises:
            ValueError: If library not found
            ConnectionError: If unable to fetch from Enrichr
        """
        lib_name = collection.value

        # Check cache first
        cache_file = self.cache_dir / f"{lib_name}.json"
        if cache_file.exists():
            logger.info(f"Loading {lib_name} from cache")
            import json
            with open(cache_file, 'r') as f:
                return json.load(f)

        # Fetch from Enrichr via GSEApy
        logger.info(f"Fetching {lib_name} from Enrichr...")
        try:
            # Use gseapy.enrichr to get gene sets, or parse from GMT format
            # gseapy has internal mechanism to handle this when running gsva
            # We'll return the library name for GSEApy to handle internally
            # But for caching, let's try to get actual gene sets

            # Alternative: use gseapy.parser to download GMT
            from gseapy.parser import download_library
            gene_sets_dict = download_library(lib_name)

            # Cache for future use
            with open(cache_file, 'w') as f:
                json.dump(gene_sets_dict, f)

            logger.info(f"Downloaded {len(gene_sets_dict)} gene sets from {lib_name}")
            return gene_sets_dict

        except Exception as e:
            logger.error(f"Failed to fetch {lib_name}: {e}")
            raise ConnectionError(f"Could not retrieve {lib_name}: {e}")

    def validate_gene_set_coverage(
        self, 
        gene_sets: Dict[str, List[str]], 
        expression_genes: Set[str]
    ) -> Dict[str, Any]:
        """Validate gene set coverage against expression data.

        Args:
            gene_sets: Dictionary of gene sets
            expression_genes: Set of genes in expression matrix

        Returns:
            Dictionary with coverage statistics
        """
        stats = {
            'total_pathways': len(gene_sets),
            'total_genes_in_sets': sum(len(genes) for genes in gene_sets.values()),
            'unique_genes_in_sets': len(set().union(*gene_sets.values())),
            'overlap_with_expression': 0,
            'pathway_coverage': {}
        }

        all_set_genes = set().union(*gene_sets.values())
        overlap = all_set_genes & expression_genes
        stats['overlap_with_expression'] = len(overlap)
        stats['coverage_percentage'] = len(overlap) / len(all_set_genes) * 100 if all_set_genes else 0

        # Per-pathway stats
        for pathway, genes in gene_sets.items():
            pathway_overlap = set(genes) & expression_genes
            stats['pathway_coverage'][pathway] = {
                'total_genes': len(genes),
                'matched_genes': len(pathway_overlap),
                'coverage_pct': len(pathway_overlap) / len(genes) * 100 if genes else 0
            }

        return stats


# ============================================================================
# GSVA Analyzer using GSEApy
# ============================================================================

class GSVAAnalyzer:
    """GSVA analysis wrapper using GSEApy backend.

    This class provides a unified interface to GSVA analysis with support
    for multiple gene set collections (Hallmark, KEGG, Reactome) and
    automatic gene set management.

    Performance Notes:
        - GSEApy's GSVA implementation is Rust-accelerated (since v1.1.0)
        - Parallel processing is handled internally by GSEApy
        - Memory usage scales with expression matrix size, not gene set count
        - For very large matrices (>10GB), consider chunking samples
    """

    def __init__(self, config: Optional[GSVAConfig] = None):
        """Initialize GSVA analyzer.

        Args:
            config: GSVAConfig instance. If None, uses defaults (Hallmark only).
        """
        if not GSEAPY_AVAILABLE:
            raise ImportError(
                "GSEApy is required for GSVA analysis. "
                "Install with: pip install gseapy"
            )

        self.config = config or GSVAConfig()
        self.library_manager = GeneSetLibraryManager()

        # Setup logging
        self._setup_logging()

        # Validate gene sets are specified
        if not self.config.gene_sets:
            raise ValueError("At least one gene set collection must be specified")

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

    def _prepare_gene_sets_input(
        self, 
        collections: List[GeneSetCollection]
    ) -> Union[str, List[str]]:
        """Prepare gene_sets parameter for GSEApy.

        Args:
            collections: List of GeneSetCollection enums

        Returns:
            Gene sets in format accepted by gseapy.gsva():
            - Library name(s) as string or list of strings
        """
        lib_names = [col.value for col in collections]

        # If single collection, return as string; else return list
        if len(lib_names) == 1:
            return lib_names[0]
        return lib_names

    def analyze(
        self, 
        expression_data: pd.DataFrame,
        gene_sets: Optional[List[GeneSetCollection]] = None
    ) -> GSVAResult:
        """Run GSVA analysis on expression data.

        This method computes GSVA enrichment scores for the specified
        gene set collections using GSEApy's optimized implementation.

        Algorithm Steps:
            1. Validate input data format (genes x samples)
            2. Prepare gene sets from MSigDB via GSEApy
            3. Run GSVA with specified parameters (kcdf, weight, etc.)
            4. Parse results into structured DataFrame
            5. Calculate coverage statistics

        Args:
            expression_data: DataFrame with genes as rows, samples as columns
            gene_sets: Override default gene sets (optional)

        Returns:
            GSVAResult object containing enrichment scores and metadata

        Raises:
            ValueError: If input data is invalid
            RuntimeError: If GSVA computation fails
            ImportError: If gseapy is not installed
        """
        # Validate input
        if expression_data.empty:
            raise ValueError("Expression data is empty")
        if expression_data.isna().all().all():
            raise ValueError("Expression data contains only NA values")

        # Use configured gene sets or override
        collections = gene_sets or self.config.gene_sets
        if not collections:
            raise ValueError("No gene sets specified for analysis")

        logger.info(f"Starting GSVA analysis with {len(collections)} collection(s)")
        logger.info(f"Expression data: {expression_data.shape[0]} genes x "
                   f"{expression_data.shape[1]} samples")

        # Prepare gene sets input
        gs_input = self._prepare_gene_sets_input(collections)
        logger.info(f"Using gene sets: {gs_input}")

        try:
            # Run GSVA via GSEApy
            # Note: GSEApy handles parallelization internally
            result_obj = gseapy_gsva(
                data=expression_data,
                gene_sets=gs_input,
                kcdf=self.config.kcdf,
                weight=self.config.weight,
                mx_diff=self.config.mx_diff,
                abs_rnk=self.config.abs_rnk,
                min_size=self.config.min_size,
                max_size=self.config.max_size,
                threads=self.config.n_cores,
                seed=self.config.random_seed,
                verbose=logger.level <= logging.INFO
            )

            # Extract results
            # result_obj.res2d contains the main results as DataFrame
            scores_df = result_obj.res2d

            # GSEApy returns results in different possible formats depending on version
            # Handle both old format (index=pathways, columns=samples) and new format
            if scores_df is None or scores_df.empty:
                raise RuntimeError("GSVA returned empty results")

            # Convert to pathway x sample matrix if needed
            # GSEApy returns long-form or wide-form depending on version
            if 'Term' in scores_df.columns or 'term' in scores_df.columns:
                # Long format - pivot to wide
                term_col = 'Term' if 'Term' in scores_df.columns else 'term'
                sample_cols = [c for c in scores_df.columns if c not in [term_col, 'Name']]

                if len(sample_cols) == len(expression_data.columns):
                    # Already in good format
                    scores_matrix = scores_df.set_index(term_col)[sample_cols]
                else:
                    # Need to infer structure
                    scores_matrix = scores_df
            else:
                scores_matrix = scores_df

            # Validate dimensions
            if scores_matrix.shape[1] != expression_data.shape[1]:
                logger.warning(f"Score matrix columns ({scores_matrix.shape[1]}) "
                              f"don't match samples ({expression_data.shape[1]})")

            # Calculate coverage statistics
            pathway_stats = {
                'collections_used': [c.value for c in collections],
                'total_pathways_scored': len(scores_matrix),
                'samples_scored': scores_matrix.shape[1],
                'gsva_parameters': {
                    'kcdf': self.config.kcdf,
                    'weight': self.config.weight,
                    'mx_diff': self.config.mx_diff,
                    'min_size': self.config.min_size,
                    'max_size': self.config.max_size
                }
            }

            # Create long-format DataFrame for visualization
            enrichment_long = scores_matrix.reset_index().melt(
                id_vars=[scores_matrix.index.name or 'index'],
                var_name='Sample',
                value_name='Enrichment_Score'
            )

            logger.info(f"GSVA analysis complete: {len(scores_matrix)} pathways x "
                       f"{scores_matrix.shape[1]} samples")

            return GSVAResult(
                scores=scores_matrix,
                gene_sets_used=[c.value for c in collections],
                pathway_coverage=pathway_stats,
                enrichment_df=enrichment_long
            )

        except Exception as e:
            logger.error(f"GSVA analysis failed: {e}")
            raise RuntimeError(f"GSVA computation failed: {e}") from e

    def analyze_multiple_collections_separately(
        self, 
        expression_data: pd.DataFrame,
        collections: Optional[List[GeneSetCollection]] = None
    ) -> Dict[str, GSVAResult]:
        """Analyze each gene set collection separately.

        This is useful when you want separate results for Hallmark, 
        KEGG, and Reactome rather than combined scores.

        Args:
            expression_data: DataFrame with genes as rows, samples as columns
            collections: List of collections to analyze (default: from config)

        Returns:
            Dictionary mapping collection names to GSVAResult objects
        """
        collections = collections or self.config.gene_sets
        results = {}

        for collection in collections:
            logger.info(f"Analyzing collection: {collection.value}")

            # Create temporary config with single collection
            temp_config = GSVAConfig(
                kcdf=self.config.kcdf,
                weight=self.config.weight,
                mx_diff=self.config.mx_diff,
                abs_rnk=self.config.abs_rnk,
                min_size=self.config.min_size,
                max_size=self.config.max_size,
                n_cores=self.config.n_cores,
                random_seed=self.config.random_seed,
                gene_sets=[collection]
            )

            temp_analyzer = GSVAAnalyzer(temp_config)
            result = temp_analyzer.analyze(expression_data)
            results[collection.value] = result

        return results


# ============================================================================
# Previous ssGSEA code remains unchanged (SSGSEACalculator, MASAnalyzer, etc.)
# ============================================================================

# [Include the previous ssGSEA classes here - truncated for brevity but would be included]
# SSGSEACalculator class...
# MASAnalyzer class...
# quick_mas_score function...
# save_mas_results function...


# ============================================================================
# Unified Analyzer Interface
# ============================================================================

class IntegratedPathwayAnalyzer:
    """Unified interface for both MAS (ssGSEA) and GSVA pathway analysis.

    This class provides a single entry point for analyzing expression data
    with both custom microbial signatures (MAS) and standard pathway databases
    (Hallmark, KEGG, Reactome via GSVA).

    Usage Examples:
        >>> # MAS only
        >>> analyzer = IntegratedPathwayAnalyzer()
        >>> result = analyzer.run_mas(expression_df, "act.txt", "inact.txt")

        >>> # GSVA only (Hallmark + KEGG)
        >>> analyzer = IntegratedPathwayAnalyzer(
        ...     gsva_config=GSVAConfig(gene_sets=[GeneSetCollection.HALLMARK, GeneSetCollection.KEGG])
        ... )
        >>> result = analyzer.run_gsva(expression_df)

        >>> # Both MAS and GSVA
        >>> result = analyzer.run_both(expression_df, "act.txt", "inact.txt")
    """

    def __init__(
        self,
        mas_config: Optional[SSGSEAConfig] = None,
        gsva_config: Optional[GSVAConfig] = None
    ):
        """Initialize integrated analyzer.

        Args:
            mas_config: Configuration for MAS/ssGSEA analysis (optional)
            gsva_config: Configuration for GSVA analysis (optional)
        """
        self.mas_config = mas_config or SSGSEAConfig()
        self.gsva_config = gsva_config

        self.mas_analyzer: Optional[MASAnalyzer] = None
        self.gsva_analyzer: Optional[GSVAAnalyzer] = None

        # Initialize MAS analyzer if config provided
        if self.mas_config:
            self.mas_analyzer = MASAnalyzer(self.mas_config)

        # Initialize GSVA analyzer if config provided and gseapy available
        if self.gsva_config and GSEAPY_AVAILABLE:
            self.gsva_analyzer = GSVAAnalyzer(self.gsva_config)

    def run_mas(
        self, 
        expression_data: pd.DataFrame,
        activation_path: Union[str, Path],
        inactivation_path: Union[str, Path]
    ) -> MASResult:
        """Run MAS analysis only.

        Args:
            expression_data: Expression matrix (genes x samples)
            activation_path: Path to activation gene set file
            inactivation_path: Path to inactivation gene set file

        Returns:
            MASResult with microbial-associated signatures
        """
        if not self.mas_analyzer:
            self.mas_analyzer = MASAnalyzer(self.mas_config)

        return self.mas_analyzer.analyze(expression_data, activation_path, inactivation_path)

    def run_gsva(
        self, 
        expression_data: pd.DataFrame,
        gene_sets: Optional[List[GeneSetCollection]] = None
    ) -> GSVAResult:
        """Run GSVA analysis only.

        Args:
            expression_data: Expression matrix (genes x samples)
            gene_sets: Override default gene sets (optional)

        Returns:
            GSVAResult with pathway enrichment scores

        Raises:
            ImportError: If gseapy not installed
            ValueError: If no GSVA config available
        """
        if not GSEAPY_AVAILABLE:
            raise ImportError("gseapy is required for GSVA analysis")

        if not self.gsva_analyzer:
            if not self.gsva_config:
                # Create default config with Hallmark
                self.gsva_config = GSVAConfig(gene_sets=[GeneSetCollection.HALLMARK])
            self.gsva_analyzer = GSVAAnalyzer(self.gsva_config)

        return self.gsva_analyzer.analyze(expression_data, gene_sets)

    def run_both(
        self, 
        expression_data: pd.DataFrame,
        activation_path: Union[str, Path],
        inactivation_path: Union[str, Path],
        gsva_gene_sets: Optional[List[GeneSetCollection]] = None
    ) -> CombinedResult:
        """Run both MAS and GSVA analyses.

        This method runs both analyses sequentially and returns combined results.
        Note: GSVA requires gseapy to be installed.

        Args:
            expression_data: Expression matrix (genes x samples)
            activation_path: Path to activation gene set file
            inactivation_path: Path to inactivation gene set file
            gsva_gene_sets: Gene sets for GSVA (optional, uses default if None)

        Returns:
            CombinedResult with both MAS and GSVA results
        """
        # Run MAS
        logger.info("=" * 60)
        logger.info("Starting Integrated Pathway Analysis (MAS + GSVA)")
        logger.info("=" * 60)

        mas_result = self.run_mas(expression_data, activation_path, inactivation_path)

        # Run GSVA if available
        gsva_result = None
        if GSEAPY_AVAILABLE:
            try:
                gsva_result = self.run_gsva(expression_data, gsva_gene_sets)
            except Exception as e:
                logger.warning(f"GSVA analysis failed: {e}. Returning MAS results only.")
        else:
            logger.warning("gseapy not available. Skipping GSVA analysis.")

        return CombinedResult(mas_result=mas_result, gsva_result=gsva_result)


# ============================================================================
# Utility Functions
# ============================================================================

def run_integrated_analysis(
    expression_data: pd.DataFrame,
    activation_path: Union[str, Path],
    inactivation_path: Union[str, Path],
    gsva_collections: Optional[List[str]] = None,
    n_cores: int = -1,
    chunk_size: int = 100
) -> CombinedResult:
    """Convenience function for one-line integrated analysis.

    This function automatically configures and runs both MAS and GSVA
    analyses with sensible defaults.

    Args:
        expression_data: Expression matrix (genes x samples)
        activation_path: Path to activation gene set file
        inactivation_path: Path to inactivation gene set file
        gsva_collections: List of collection names ("HALLMARK", "KEGG", "REACTOME")
                       or None to skip GSVA
        n_cores: Number of CPU cores (-1 for auto)
        chunk_size: Chunk size for memory efficiency

    Returns:
        CombinedResult with both analyses

    Example:
        >>> result = run_integrated_analysis(
        ...     expr_df,
        ...     "DEGs_activation.txt",
        ...     "DEGs_inactivation.txt",
        ...     gsva_collections=["HALLMARK", "KEGG"],
        ...     n_cores=8
        ... )
    """
    # Parse GSVA collections
    gsva_config = None
    if gsva_collections and GSEAPY_AVAILABLE:
        collection_map = {
            "HALLMARK": GeneSetCollection.HALLMARK,
            "KEGG": GeneSetCollection.KEGG,
            "REACTOME": GeneSetCollection.REACTOME,
            "BIOCARTA": GeneSetCollection.BIOCARTA,
            "GO_BP": GeneSetCollection.GO_BP,
            "GO_CC": GeneSetCollection.GO_CC,
            "GO_MF": GeneSetCollection.GO_MF
        }

        collections = []
        for col in gsva_collections:
            col_upper = col.upper()
            if col_upper in collection_map:
                collections.append(collection_map[col_upper])
            else:
                logger.warning(f"Unknown collection: {col}. Skipping.")

        if collections:
            gsva_config = GSVAConfig(
                gene_sets=collections,
                n_cores=n_cores,
                kcdf="Gaussian"
            )

    # Create configs
    mas_config = SSGSEAConfig(n_cores=n_cores, chunk_size=chunk_size)

    # Run analysis
    analyzer = IntegratedPathwayAnalyzer(
        mas_config=mas_config,
        gsva_config=gsva_config
    )

    return analyzer.run_both(
        expression_data,
        activation_path,
        inactivation_path
    )


# ============================================================================
# Module Entry Point
# ============================================================================

if __name__ == "__main__":
    print("Integrated MAS + GSVA Analysis Module")
    print("=" * 60)
    print("Available functions:")
    print("  - run_integrated_analysis()  : One-line MAS + GSVA")
    print("  - IntegratedPathwayAnalyzer : Unified interface class")
    print("  - MASAnalyzer               : ssGSEA for microbial signatures")
    print("  - GSVAAnalyzer              : GSVA for pathway analysis")
    print("  - GeneSetLibraryManager     : MSigDB gene set management")
    print("")
    print("Available Gene Set Collections:")
    for col in GeneSetCollection:
        print(f"  - {col.name}: {col.value}")
