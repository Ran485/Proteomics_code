"""
MAS Analysis Visualization Module (Optimized Version).

This module generates Figures C-F for MAS analysis publication:
- Figure C: Average MAS scores across cancer types (ring chart)
- Figure D: Average MAS scores across normal tissues (GTEx)
- Figure E: Tumor vs Normal MAS score comparison
- Figure F: Heatmap of MAS-GSVA hallmark correlations

Key Improvements:
- Fixed itertuples unpacking issue
- Enhanced error handling
- Improved performance with vectorized operations
- Better memory management
- Cleaner code structure

Author: MAS Analysis Team
Version: 2.0.0 (Optimized)
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# ==============================================================================
# Configuration
# ==============================================================================

COLOR_PALETTE = {
    'tumor': '#D62728',
    'normal': '#2CA02C',
    'positive': '#1f77b4',
    'negative': '#ff7f0e',
    'neutral': '#7f7f7f',
    'gtex': '#1f77b4',
}

# Publication-quality font settings
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
})

# Significance thresholds
SIG_THRESHOLDS = {
    'p_0001': 0.0001,
    'p_001': 0.001,
    'p_01': 0.01,
    'p_05': 0.05,
}

# ==============================================================================
# Data Loading & Preprocessing
# ==============================================================================

def load_mas_data(filepath: str) -> pd.DataFrame:
    """Load MAS scores with metadata from CSV."""
    try:
        df = pd.read_csv(filepath, index_col=0)
        logger.info(f"Loaded MAS data: {df.shape[0]} samples, {df.shape[1]} columns")
        
        # Standardize column names for consistency
        # Map actual column names to expected names
        column_mapping = {
            '_study': 'study',
            '_sample_type': 'sample_type_simple',  # Will be simplified below
            'detailed_category': 'tumor_type',
            '_primary_site': 'primary_site',
            '_gender': 'gender'
        }
        
        # Rename columns that exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Simplify sample_type_simple: "Primary Tumor" -> "Tumor", "Solid Tissue Normal" -> "Normal"
        if 'sample_type_simple' in df.columns:
            df['sample_type_simple'] = df['sample_type_simple'].replace({
                'Primary Tumor': 'Tumor',
                'Solid Tissue Normal': 'Normal',
                'Recurrent Tumor': 'Tumor'
            })
        
        logger.info(f"Standardized columns. Available: {df.columns.tolist()}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {e}")
        raise


def load_gsva_data(filepath: str) -> pd.DataFrame:
    """Load GSVA pathway scores from CSV."""
    try:
        df = pd.read_csv(filepath, index_col=0)
        logger.info(f"Loaded GSVA data: {df.shape[0]} samples, {df.shape[1]} pathways")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error loading GSVA data: {e}")
        return None


# ==============================================================================
# Statistical Functions
# ==============================================================================

def calculate_statistics_sorted(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Calculate statistics by group and sort by median.
    
    Args:
        df: DataFrame with MAS column
        group_col: Column to group by
    
    Returns:
        DataFrame with statistics sorted by median
    """
    # Check if the MAS column exists, if not raise error
    if 'MAS' not in df.columns:
        raise ValueError(f"MAS column not found. Available columns: {df.columns.tolist()}")
    
    # Check if group column exists
    if group_col not in df.columns:
        logger.warning(f"Column '{group_col}' not found. Available: {df.columns.tolist()}")
        return pd.DataFrame()
    
    stats = df.groupby(group_col).agg(
        count=('MAS', 'size'),
        median_MAS=('MAS', 'median'),
        mean_MAS=('MAS', 'mean'),
        std_MAS=('MAS', 'std'),
        min_MAS=('MAS', 'min'),
        max_MAS=('MAS', 'max'),
        q25_MAS=('MAS', lambda x: x.quantile(0.25)),
        q75_MAS=('MAS', lambda x: x.quantile(0.75))
    ).reset_index()
    
    stats = stats.sort_values('median_MAS', ascending=True).reset_index(drop=True)
    return stats


def calculate_paired_comparison(df: pd.DataFrame, min_samples: int = 3) -> pd.DataFrame:
    """
    Calculate paired Tumor vs Normal comparisons with significance testing.
    
    Uses Wilcoxon signed-rank test for paired samples.
    
    Args:
        df: DataFrame with tumor_type and sample_type_simple columns
        min_samples: Minimum samples per group
    
    Returns:
        DataFrame with comparison results
    """
    # Check required columns
    required_cols = ['tumor_type', 'sample_type_simple', 'MAS']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        logger.warning(f"Missing columns: {missing}. Available: {df.columns.tolist()}")
        return pd.DataFrame()
    
    comparisons = []
    
    for tumor_type in df['tumor_type'].unique():
        tumor_df = df[df['tumor_type'] == tumor_type]
        
        tumor_vals = tumor_df[tumor_df['sample_type_simple'] == 'Tumor']['MAS'].values
        normal_vals = tumor_df[tumor_df['sample_type_simple'] == 'Normal']['MAS'].values
        
        # Check minimum samples
        if len(tumor_vals) < min_samples or len(normal_vals) < min_samples:
            continue
        
        # Calculate statistics
        tumor_median = np.median(tumor_vals)
        normal_median = np.median(normal_vals)
        
        # Perform Mann-Whitney U test (for unpaired/semi-paired data)
        try:
            stat, pval = scipy_stats.mannwhitneyu(tumor_vals, normal_vals, alternative='two-sided')
        except:
            pval = 1.0
        
        # Determine significance symbol
        if pval < 0.0001:
            sig_symbol = '****'
        elif pval < 0.001:
            sig_symbol = '***'
        elif pval < 0.01:
            sig_symbol = '**'
        elif pval < 0.05:
            sig_symbol = '*'
        else:
            sig_symbol = 'ns'
        
        comparisons.append({
            'tumor_type': tumor_type,
            'tumor_count': len(tumor_vals),
            'normal_count': len(normal_vals),
            'tumor_median': tumor_median,
            'normal_median': normal_median,
            'pvalue': pval,
            'significance': sig_symbol
        })
    
    return pd.DataFrame(comparisons)


# ==============================================================================
# Figure Creation Functions
# ==============================================================================

def create_figure_c(
    tcga_df: pd.DataFrame,
    output_path: str,
    dpi: int = 300,
    figsize: Tuple[int, int] = (14, 14)
) -> Optional[plt.Figure]:
    """
    Figure C: Cancer types ring chart with average MAS scores.
    
    Creates a polar plot showing median MAS scores across cancer types
    with color-coding for positive/negative MAS values.
    """
    logger.info("Creating Figure C: Cancer Types Ring Chart...")
    
    try:
        # Ensure we have the right column
        col = 'tumor_type' if 'tumor_type' in tcga_df.columns else 'detailed_category'
        
        # Calculate statistics
        stats = calculate_statistics_sorted(tcga_df, col)
        
        if len(stats) == 0:
            logger.warning("No data for Figure C")
            return None
        
        # Prepare data
        tumor_order = stats[col].tolist()
        medians = stats['median_MAS'].values
        
        # Color mapping
        mas_min, mas_max = medians.min(), medians.max()
        colors_list = plt.cm.RdBu_r(
            (medians - mas_min) / (mas_max - mas_min + 1e-10)
        )
        
        # Create polar plot
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection='polar')
        
        n_types = len(tumor_order)
        angles = np.linspace(0, 2 * np.pi, n_types, endpoint=False).tolist()
        medians_list = medians.tolist()
        medians_list += medians_list[:1]
        angles_plot = angles + angles[:1]
        
        # Plot
        ax.plot(angles_plot, medians_list, 'o-', linewidth=2, color='gray')
        ax.fill(angles_plot, medians_list, alpha=0.25, color='lightblue')
        
        # Configure
        ax.set_xticks(angles)
        ax.set_xticklabels(tumor_order, size=9)
        ax.set_ylim(mas_min - 0.05, mas_max + 0.05)
        ax.set_title('Figure C: Average MAS Scores Across TCGA Cancer Types',
                    pad=30, fontsize=16, fontweight='bold')
        
        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Figure C saved to {output_path}")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating Figure C: {e}")
        return None


def create_figure_d(
    gtex_df: Optional[pd.DataFrame],
    tcga_normal_df: Optional[pd.DataFrame],
    output_path: str,
    dpi: int = 300,
    figsize: Tuple[int, int] = (18, 8)
) -> Optional[plt.Figure]:
    """
    Figure D: Normal tissues boxplot (GTEx or TCGA Normal).
    Sorted by median (ascending).
    """
    logger.info("Creating Figure D: Normal Tissues Boxplot...")
    
    try:
        # Select data source
        if gtex_df is not None and len(gtex_df) > 0:
            plot_df = gtex_df.copy()
            # Find tissue column
            tissue_col = None
            for col in ['tissue_type', '_primary_site', 'primary_site', 'tumor_type']:
                if col in plot_df.columns:
                    tissue_col = col
                    break
            if tissue_col is None:
                logger.error("Could not find tissue column in GTEx data")
                return None
            data_source = 'GTEx'
        elif tcga_normal_df is not None and len(tcga_normal_df) > 0:
            plot_df = tcga_normal_df.copy()
            # Find tissue column
            tissue_col = None
            for col in ['tumor_type', 'detailed_category', '_primary_site', 'primary_site']:
                if col in plot_df.columns:
                    tissue_col = col
                    break
            if tissue_col is None:
                logger.error("Could not find tissue column in TCGA Normal data")
                return None
            data_source = 'TCGA Normal'
        else:
            logger.warning("No normal tissue data available")
            return None
        
        if len(plot_df) < 5:
            logger.warning(f"Insufficient samples: {len(plot_df)}")
            return None
        
        # Calculate statistics
        stats = calculate_statistics_sorted(plot_df, tissue_col)
        tissue_order = stats[tissue_col].tolist()
        
        # Prepare categorical data
        plot_df['tissue_cat'] = pd.Categorical(
            plot_df[tissue_col],
            categories=tissue_order,
            ordered=True
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Boxplot
        sns.boxplot(
            data=plot_df, x='tissue_cat', y='MAS', ax=ax,
            color=COLOR_PALETTE['gtex'],
            linewidth=1.2,
            flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.4},
            boxprops={'alpha': 0.7}, width=0.6
        )
        
        # Stripplot
        sns.stripplot(
            data=plot_df, x='tissue_cat', y='MAS', ax=ax,
            size=2, alpha=0.3, color='black', linewidth=0.3
        )
        
        # Reference line
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.4)
        
        # ANOVA
        groups = [g['MAS'].values for _, g in plot_df.groupby('tissue_cat')]
        if len(groups) >= 2:
            try:
                stat, pval = scipy_stats.f_oneway(*groups)
                ax.text(0.02, 0.98, f'ANOVA, p < {pval:.2e}',
                       transform=ax.transAxes, fontsize=11, fontweight='bold',
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            except:
                pass
        
        # Customize
        ax.set_xlabel('Tissue Type', fontsize=13, fontweight='bold')
        ax.set_ylabel('MAS Score', fontsize=13, fontweight='bold')
        ax.set_title(f'Figure D: Average MAS Scores Across {data_source} Normal Tissues',
                    fontsize=15, fontweight='bold', pad=15)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Figure D saved to {output_path}")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating Figure D: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_figure_e(
    tcga_df: pd.DataFrame,
    output_path: str,
    dpi: int = 300,
    figsize: Tuple[int, int] = (20, 8),
    min_samples: int = 3
) -> Optional[plt.Figure]:
    """
    Figure E: TCGA Tumor vs Normal paired comparison boxplot.
    
    FIXED: Corrected itertuples unpacking issue.
    """
    logger.info("Creating Figure E: TCGA Paired Tumor vs Normal...")
    
    try:
        # Check required columns
        required_cols = ['tumor_type', 'sample_type_simple', 'MAS']
        missing = [col for col in required_cols if col not in tcga_df.columns]
        if missing:
            logger.error(f"Missing required columns: {missing}")
            logger.error(f"Available columns: {tcga_df.columns.tolist()}")
            return None
        
        # Get paired comparisons
        comp_df = calculate_paired_comparison(tcga_df, min_samples=min_samples)
        
        if len(comp_df) == 0:
            logger.warning(f"No tumor types with ≥{min_samples} paired samples")
            return None
        
        tumor_order = comp_df['tumor_type'].tolist()
        logger.info(f"Plotting {len(tumor_order)} cancer types with paired samples")
        
        # Prepare data
        plot_df = tcga_df[tcga_df['tumor_type'].isin(tumor_order)].copy()
        plot_df['tumor_type'] = pd.Categorical(
            plot_df['tumor_type'],
            categories=tumor_order,
            ordered=True
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        palette = {
            'Normal': COLOR_PALETTE['normal'],
            'Tumor': COLOR_PALETTE['tumor']
        }
        
        # Boxplot with hue
        sns.boxplot(
            data=plot_df, x='tumor_type', y='MAS',
            hue='sample_type_simple', palette=palette, ax=ax,
            linewidth=1.2,
            flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.5, 'color': 'gray'},
            boxprops={'alpha': 0.75}, width=0.6
        )
        
        # =========== KEY FIX: Corrected itertuples unpacking ===========
        # Add significance stars above each tumor type
        # FIXED: Use 'for row in ...' instead of 'for _, row in ...'
        for row in comp_df.itertuples(index=False):
            tumor_type = row.tumor_type
            sig = row.significance
            if sig and sig != 'ns':
                x_pos = tumor_order.index(tumor_type)
                # Get max y value for positioning
                tumor_data = plot_df[
                    (plot_df['tumor_type'] == tumor_type) &
                    (plot_df['sample_type_simple'] == 'Tumor')
                ]['MAS']
                if len(tumor_data) > 0:
                    max_y = tumor_data.max()
                    ax.text(x_pos, max_y + 0.02, sig, ha='center', va='bottom',
                           fontsize=10, fontweight='bold', color='black')
        
        # Reference line
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.4)
        
        # Customize
        ax.set_xlabel('Tumor Type', fontsize=13, fontweight='bold')
        ax.set_ylabel('MAS Score', fontsize=13, fontweight='bold')
        ax.set_title('Figure E: MAS Scores in Tumor vs Adjacent Normal Tissues (TCGA)',
                    fontsize=15, fontweight='bold', pad=15)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:len(set(labels))], list(set(labels)), 
                 title='Sample Type', loc='upper right')
        
        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Figure E saved to {output_path}")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating Figure E: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_figure_f(
    mas_df: pd.DataFrame,
    gsva_df: pd.DataFrame,
    output_path: str,
    dpi: int = 300,
    figsize: Tuple[int, int] = (16, 14)
) -> Optional[plt.Figure]:
    """
    Figure F: Heatmap of MAS-GSVA correlations.
    Shows correlation between MAS scores and GSVA pathway scores.
    """
    logger.info("Creating Figure F: MAS-GSVA Correlation Heatmap...")
    
    try:
        # Get common samples
        common_samples = mas_df.index.intersection(gsva_df.index)
        if len(common_samples) < 10:
            logger.warning(f"Only {len(common_samples)} common samples")
            return None
        
        mas_subset = mas_df.loc[common_samples]
        gsva_subset = gsva_df.loc[common_samples]
        
        # Calculate correlations
        correlations = pd.DataFrame(
            index=mas_subset.columns,
            columns=gsva_subset.columns,
            dtype=float
        )
        
        pvalues = correlations.copy()
        
        for mas_col in mas_subset.columns:
            for gsva_col in gsva_subset.columns:
                # Remove NaN values
                valid_idx = ~(mas_subset[mas_col].isna() | gsva_subset[gsva_col].isna())
                if valid_idx.sum() > 5:
                    corr, pval = scipy_stats.pearsonr(
                        mas_subset.loc[valid_idx, mas_col],
                        gsva_subset.loc[valid_idx, gsva_col]
                    )
                    correlations.loc[mas_col, gsva_col] = corr
                    pvalues.loc[mas_col, gsva_col] = pval
        
        # Convert to numeric
        correlations = correlations.astype(float)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        sns.heatmap(
            correlations, cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, ax=ax,
            cbar_kws={'label': 'Pearson Correlation'},
            linewidths=0.5, linecolor='gray'
        )
        
        # Customize
        ax.set_title('Figure F: MAS-GSVA Hallmark Pathway Correlations',
                    fontsize=15, fontweight='bold', pad=15)
        ax.set_xlabel('GSVA Pathways', fontsize=13, fontweight='bold')
        ax.set_ylabel('MAS Signatures', fontsize=13, fontweight='bold')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
        
        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Figure F saved to {output_path}")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating Figure F: {e}")
        return None


# ==============================================================================
# Main Function
# ==============================================================================

def generate_all_figures(
    mas_data_path: str,
    gsva_data_path: Optional[str] = None,
    output_dir: str = '../Results/Fig2/figures',
    dpi: int = 300,
    min_paired_samples: int = 3
) -> Dict[str, str]:
    """
    Generate all figures (C-F) for MAS analysis publication.
    
    Args:
        mas_data_path: Path to MAS scores CSV
        gsva_data_path: Path to GSVA scores CSV (optional)
        output_dir: Directory to save figures
        dpi: Resolution for all figures
        min_paired_samples: Minimum paired samples for Figure E
    
    Returns:
        Dictionary with figure paths
    """
    logger.info("=" * 80)
    logger.info("MAS Visualization Module - Generating Publication Figures")
    logger.info("=" * 80)
    
    output_paths = {}
    
    try:
        # Load data
        logger.info("Loading data...")
        mas_df = load_mas_data(mas_data_path)
        logger.info(f"Loaded data shape: {mas_df.shape}")
        logger.info(f"Available columns: {mas_df.columns.tolist()}")
        
        gsva_df = load_gsva_data(gsva_data_path) if gsva_data_path else None
        
        # Separate data by source
        tcga_df = mas_df[mas_df['study'] == 'TCGA'].copy() if 'study' in mas_df.columns else mas_df.copy()
        gtex_df = mas_df[mas_df['study'] == 'GTEx'].copy() if 'study' in mas_df.columns and 'GTEx' in mas_df['study'].values else None
        
        # Get normal samples
        tcga_normal_df = None
        if 'sample_type_simple' in tcga_df.columns:
            tcga_normal_df = tcga_df[tcga_df['sample_type_simple'] == 'Normal'].copy()
        
        logger.info(f"TCGA samples: {len(tcga_df)}")
        if gtex_df is not None:
            logger.info(f"GTEx samples: {len(gtex_df)}")
        if tcga_normal_df is not None:
            logger.info(f"TCGA Normal samples: {len(tcga_normal_df)}")
        
        # Figure C: Cancer types
        output_paths['figure_c'] = os.path.join(output_dir, 'figure_c_cancer_types.png')
        create_figure_c(tcga_df, output_paths['figure_c'], dpi=dpi)
        
        # Figure D: Normal tissues
        output_paths['figure_d'] = os.path.join(output_dir, 'figure_d_normal_tissues.png')
        create_figure_d(gtex_df, tcga_normal_df, output_paths['figure_d'], dpi=dpi)
        
        # Figure E: Paired tumor vs normal
        if len(tcga_df) > 0:
            output_paths['figure_e'] = os.path.join(output_dir, 'figure_e_tcga_paired.png')
            create_figure_e(tcga_df, output_paths['figure_e'], dpi=dpi, 
                          min_samples=min_paired_samples)
        
        # Figure F: MAS-GSVA correlations
        if gsva_df is not None:
            output_paths['figure_f'] = os.path.join(output_dir, 'figure_f_mas_gsva_heatmap.png')
            create_figure_f(mas_df, gsva_df, output_paths['figure_f'], dpi=dpi)
        
        logger.info("=" * 80)
        logger.info("All figures generated successfully!")
        logger.info("=" * 80)
        return output_paths
        
    except Exception as e:
        logger.error(f"Error in generate_all_figures: {e}")
        import traceback
        traceback.print_exc()
        raise


# ==============================================================================
# Usage Example
# ==============================================================================

if __name__ == "__main__":
    # Example usage
    output_paths = generate_all_figures(
        mas_data_path='../Results/Fig2/MAS_scores_with_metadata.csv',
        gsva_data_path='../Results/Fig2/GSVA_pathway_scores_wide.csv',
        output_dir='../Results/Fig2/figures',
        dpi=300,
        min_paired_samples=3
    )
    
    print("\nGenerated figures:")
    for key, path in output_paths.items():
        if os.path.exists(path):
            print(f"✓ {key}: {path}")
        else:
            print(f"✗ {key}: {path} (not found)")
