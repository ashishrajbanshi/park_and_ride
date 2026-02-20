"""
Equity-Aware Coverage Analysis
===============================

Purpose:
    Ensure equitable P&R coverage across all geographic areas
    Balance efficiency (accessibility) with equity (spatial justice)
    
Methodology:
    1. Define service zones (Census tracts)
    2. Calculate coverage per zone
    3. Identify underserved areas
    4. Apply equity bonuses
    5. Generate efficiency-equity tradeoff curves
    
Citation:
    Based on Church & ReVelle (1974) maximin location models
    
Author: [Your Name]
Date: January 2026
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import matplotlib.pyplot as plt

import json
import warnings

import sys

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config_loader import Config


# ============================================================================
# ZONE DEFINITION
# ============================================================================

def define_service_zones(census_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Define geographic zones for equity analysis
    
    Uses Census tracts as service zones (standard in transportation planning)
    
    Args:
        census_gdf: Census block groups
        
    Returns:
        GeoDataFrame with zone definitions
    """
    print("\n📍 Defining service zones for equity analysis...")
    
    # For simplicity, we'll use census blocks as zones
    zones_gdf = census_gdf.copy()
    
    # DEBUG: Check what columns we have
    print(f"  🔍 Census columns available: {zones_gdf.columns.tolist()}")
    
    # Create zone identifier
    if 'geoid' in zones_gdf.columns:
        zones_gdf['zone_id'] = zones_gdf['geoid']
    elif 'GEOID' in zones_gdf.columns:
        zones_gdf['zone_id'] = zones_gdf['GEOID']
    else:
        zones_gdf['zone_id'] = range(1, len(zones_gdf) + 1)
        print(f"  ℹ️  No GEOID found, using sequential zone_id")
    
    # Handle population column
    if 'total_pop' in zones_gdf.columns:
        zones_gdf['zone_population'] = zones_gdf['total_pop'].fillna(0)
    elif 'population' in zones_gdf.columns:
        zones_gdf['zone_population'] = zones_gdf['population'].fillna(0)
    else:
        print(f"  ⚠️  No population column found - using default value of 100 per zone")
        zones_gdf['zone_population'] = 100
    
    # Handle employment column
    if 'employment' in zones_gdf.columns:
        zones_gdf['zone_employment'] = zones_gdf['employment'].fillna(0)
    else:
        zones_gdf['zone_employment'] = 0
    
    # Calculate zone area
    zones_gdf['zone_area_km2'] = zones_gdf.geometry.area / 1_000_000
    
    # Calculate zone centroid
    zones_gdf['zone_centroid'] = zones_gdf.geometry.centroid
    
    print(f"  ✓ Defined {len(zones_gdf)} service zones")
    print(f"  ✓ Total population: {zones_gdf['zone_population'].sum():,.0f}")
    print(f"  ✓ Total employment: {zones_gdf['zone_employment'].sum():,.0f}")
    
    # Keep only necessary columns for merge (avoid column name conflicts)
    zones_gdf = zones_gdf[[
        'zone_id', 
        'zone_population', 
        'zone_employment', 
        'zone_area_km2', 
        'zone_centroid',
        'geometry'
    ]]
    
    return zones_gdf


# ============================================================================
# COVERAGE CALCULATION
# ============================================================================

def calculate_zone_coverage(parking_gdf: gpd.GeoDataFrame,
                           zones_gdf: gpd.GeoDataFrame,
                           drive_distance: int = 5000) -> pd.DataFrame:
    """
    Calculate P&R coverage for each zone
    
    Coverage = accessibility-weighted proximity to P&R sites
    
    Args:
        parking_gdf: Deduplicated parking lots with accessibility scores
        zones_gdf: Service zones
        drive_distance: Maximum reasonable drive to P&R (meters)
        
    Returns:
        DataFrame with coverage metrics per zone
    """
    print(f"\n📊 Calculating P&R coverage by zone (max drive: {drive_distance/1000:.1f}km)...")
    
    # Ensure same CRS
    if parking_gdf.crs != zones_gdf.crs:
        zones_gdf = zones_gdf.to_crs(parking_gdf.crs)
    
    coverage_results = []
    
    for idx, zone in zones_gdf.iterrows():
        zone_id = zone['zone_id']
        zone_centroid = zone['zone_centroid']
        zone_pop = zone.get('zone_population', 0)
        
        # Find P&R sites within driving distance of zone centroid
        parking_gdf['distance_to_zone'] = parking_gdf.geometry.distance(zone_centroid)
        accessible_sites = parking_gdf[parking_gdf['distance_to_zone'] <= drive_distance]
        
        if len(accessible_sites) == 0:
            # No P&R coverage
            coverage_results.append({
                'zone_id': zone_id,
                'num_pr_sites': 0,
                'nearest_pr_distance_m': 9999,
                'coverage_score': 0,
                'weighted_accessibility': 0,
                'zone_population': zone_pop,
                'has_coverage': False
            })
        else:
            # Calculate coverage metrics
            num_sites = len(accessible_sites)
            nearest_distance = accessible_sites['distance_to_zone'].min()
            
            # Coverage score: accessibility-weighted, distance-decayed
            accessible_sites['coverage_contribution'] = (
                accessible_sites['accessibility_score'] *
                (1 - accessible_sites['distance_to_zone'] / drive_distance)
            )
            
            coverage_score = accessible_sites['coverage_contribution'].sum()
            
            # Weighted accessibility
            if accessible_sites['coverage_contribution'].sum() > 0:
                weighted_accessibility = (
                    accessible_sites['accessibility_score'] * 
                    accessible_sites['coverage_contribution']
                ).sum() / accessible_sites['coverage_contribution'].sum()
            else:
                weighted_accessibility = 0
            
            coverage_results.append({
                'zone_id': zone_id,
                'num_pr_sites': num_sites,
                'nearest_pr_distance_m': nearest_distance,
                'coverage_score': coverage_score,  # RAW (unbounded)
                'weighted_accessibility': weighted_accessibility,
                'zone_population': zone_pop,
                'has_coverage': True
            })
    
    coverage_df = pd.DataFrame(coverage_results)
    
    # ========================================================================
    # NORMALIZATION: Scale coverage_score to 0-100
    # ========================================================================
    print(f"\n📐 Normalizing coverage scores to 0-100 scale...")
    
    # Store raw scores for reference
    coverage_df['coverage_score_raw'] = coverage_df['coverage_score']
    
    # Get min/max (excluding zeros for better scaling)
    raw_min = coverage_df['coverage_score'].min()
    raw_max = coverage_df['coverage_score'].max()
    
    print(f"  Raw coverage range: {raw_min:.1f} - {raw_max:.1f}")
    
    # Min-Max normalization to 0-100
    if raw_max > raw_min:
        coverage_df['coverage_score_normalized'] = (
            ((coverage_df['coverage_score'] - raw_min) / (raw_max - raw_min)) * 100
        )
    else:
        coverage_df['coverage_score_normalized'] = 0
    
    # Replace raw with normalized for all downstream analysis
    coverage_df['coverage_score'] = coverage_df['coverage_score_normalized']
    
    norm_min = coverage_df['coverage_score'].min()
    norm_max = coverage_df['coverage_score'].max()
    
    print(f"  Normalized range: {norm_min:.1f} - {norm_max:.1f}")
    print(f"  ✅ Coverage scores normalized to 0-100")
    
    # ========================================================================
    # END NORMALIZATION
    # ========================================================================
    
    # Calculate coverage statistics
    zones_covered = coverage_df['has_coverage'].sum()
    total_zones = len(coverage_df)
    pct_covered = (zones_covered / total_zones) * 100
    
    print(f"\n  ✓ Coverage calculated for {total_zones} zones")
    print(f"  ✓ Zones with P&R coverage: {zones_covered} ({pct_covered:.1f}%)")
    print(f"  ✓ Zones without coverage: {total_zones - zones_covered} ({100-pct_covered:.1f}%)")
    
    # Population coverage
    pop_covered = coverage_df[coverage_df['has_coverage']]['zone_population'].sum()
    pop_total = coverage_df['zone_population'].sum()
    pct_pop_covered = (pop_covered / pop_total) * 100 if pop_total > 0 else 0
    
    print(f"  ✓ Population with P&R access: {pop_covered:,.0f} ({pct_pop_covered:.1f}%)")
    
    return coverage_df


# ============================================================================
# UNDERSERVED AREA IDENTIFICATION
# ============================================================================

def identify_underserved_zones(coverage_df: pd.DataFrame,
                               zones_gdf: gpd.GeoDataFrame,
                               threshold_percentile: float = 25) -> Tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Identify zones with inadequate P&R coverage
    
    Args:
        coverage_df: Coverage metrics by zone
        zones_gdf: Geographic zones
        threshold_percentile: Percentile below which zones are "underserved"
        
    Returns:
        Tuple of (coverage with flags, underserved zones GeoDataFrame)
    """
    print(f"\n🎯 Identifying underserved zones (threshold: {threshold_percentile}th percentile)...")
    
    # Merge coverage with geographic data
    coverage_with_geom = zones_gdf.merge(coverage_df, on='zone_id', how='left')
    
    # Handle duplicate zone_population columns from merge
    if 'zone_population_x' in coverage_with_geom.columns:
        coverage_with_geom['zone_population'] = coverage_with_geom['zone_population_y']
        coverage_with_geom = coverage_with_geom.drop(columns=['zone_population_x', 'zone_population_y'], errors='ignore')
    
    if 'zone_employment_x' in coverage_with_geom.columns:
        coverage_with_geom['zone_employment'] = coverage_with_geom['zone_employment_y']
        coverage_with_geom = coverage_with_geom.drop(columns=['zone_employment_x', 'zone_employment_y'], errors='ignore')
    
    # Calculate threshold (only for zones with coverage > 0)
    zones_with_coverage = coverage_df[coverage_df['coverage_score'] > 0]
    
    if len(zones_with_coverage) > 0:
        coverage_threshold = zones_with_coverage['coverage_score'].quantile(threshold_percentile / 100)
    else:
        coverage_threshold = 0
    
    print(f"  ✓ Coverage threshold: {coverage_threshold:.2f}")
    
    # Flag underserved zones
    coverage_with_geom['is_underserved'] = (
        (coverage_with_geom['coverage_score'] > 0) &  # Has some coverage
        (coverage_with_geom['coverage_score'] < coverage_threshold)  # But below threshold
    )
    
    # Simple, robust classification
    def classify_coverage(score):
        if score == 0:
            return 'No Service'
        elif len(zones_with_coverage) == 0:
            return 'Has Service'
        elif score < coverage_threshold:
            return 'Underserved'
        elif score < zones_with_coverage['coverage_score'].median():
            return 'Adequate'
        elif score < zones_with_coverage['coverage_score'].quantile(0.75):
            return 'Good Service'
        else:
            return 'Well Served'
    
    coverage_with_geom['service_level'] = coverage_with_geom['coverage_score'].apply(classify_coverage)
    
    # Count underserved
    underserved = coverage_with_geom[coverage_with_geom['is_underserved']]
    no_service = coverage_with_geom[coverage_with_geom['coverage_score'] == 0]
    
    print(f"  ✓ Zones with no service: {len(no_service)} ({len(no_service)/len(coverage_with_geom)*100:.1f}%)")
    print(f"  ✓ Zones underserved (has coverage but low): {len(underserved)} ({len(underserved)/len(coverage_with_geom)*100:.1f}%)")
    print(f"  ✓ Total problematic zones: {len(underserved) + len(no_service)} ({(len(underserved) + len(no_service))/len(coverage_with_geom)*100:.1f}%)")
    
    # Show service level distribution
    print(f"\n  📊 Service Level Distribution:")
    service_dist = coverage_with_geom['service_level'].value_counts().sort_index()
    for level, count in service_dist.items():
        pct = (count / len(coverage_with_geom)) * 100
        print(f"    {level}: {count} zones ({pct:.1f}%)")
    
    # Population in underserved zones (safe handling)
    if 'zone_population' in coverage_with_geom.columns:
        try:
            underserved_pop = underserved['zone_population'].sum()
            no_service_pop = no_service['zone_population'].sum()
            total_pop = coverage_with_geom['zone_population'].sum()
            
            if total_pop > 0:
                pct_underserved_pop = (underserved_pop / total_pop) * 100
                pct_no_service_pop = (no_service_pop / total_pop) * 100
                
                print(f"\n  👥 Population Impact:")
                print(f"    In zones with no service: {no_service_pop:,.0f} ({pct_no_service_pop:.1f}%)")
                print(f"    In underserved zones: {underserved_pop:,.0f} ({pct_underserved_pop:.1f}%)")
                print(f"    Total affected: {underserved_pop + no_service_pop:,.0f} ({(pct_underserved_pop + pct_no_service_pop):.1f}%)")
        except Exception as e:
            print(f"\n  ℹ️  Could not calculate population impact: {e}")
    
    # For underserved_zones, include both no-service and underserved
    combined_underserved = coverage_with_geom[
        (coverage_with_geom['is_underserved']) | 
        (coverage_with_geom['coverage_score'] == 0)
    ]
    
    return coverage_with_geom, combined_underserved

# ============================================================================
# COVERAGE BONUS CALCULATION
# ============================================================================

# ============================================================================
# EQUITY BONUS CALCULATION
# ============================================================================

def calculate_equity_adjustments(parking_gdf: gpd.GeoDataFrame,
                                coverage_with_geom: gpd.GeoDataFrame,
                                equity_weight: float = 0.5) -> gpd.GeoDataFrame:
    """
    Apply equity bonuses to sites serving underserved areas
    
    Methodology:
        Sites serving low-coverage zones get bonus points
        Bonus proportional to population served in underserved zones
        
    Args:
        parking_gdf: Deduplicated parking with accessibility scores
        coverage_with_geom: Zone coverage data WITH geometry and is_underserved flag
        equity_weight: Weight for equity adjustment (0-1)
        
    Returns:
        GeoDataFrame with equity-adjusted scores
    """
    print(f"\n⚖️  Calculating equity adjustments (weight: {equity_weight})...")
    
    # Ensure same CRS
    if parking_gdf.crs != coverage_with_geom.crs:
        coverage_with_geom = coverage_with_geom.to_crs(parking_gdf.crs)
    
    parking_equity = parking_gdf.copy()
    
    # For each parking site, calculate population served in underserved zones
    equity_bonuses = []
    
    for idx, site in parking_equity.iterrows():
        parking_id = site['parking_id']
        site_geom = site.geometry
        
        # Find zones within 5km drive catchment
        coverage_with_geom['distance_to_site'] = coverage_with_geom['zone_centroid'].distance(site_geom)
        nearby_zones = coverage_with_geom[coverage_with_geom['distance_to_site'] <= 5000]
        
        if len(nearby_zones) == 0:
            equity_bonus = 0
        else:
            # Filter to underserved zones only
            underserved_nearby = nearby_zones[
                nearby_zones.get('is_underserved', False) == True
            ]
            
            if len(underserved_nearby) > 0:
                # Bonus = sum of (population × inverse_distance_decay)
                underserved_nearby['weight'] = (
                    underserved_nearby['zone_population'] * 
                    (1 - underserved_nearby['distance_to_site'] / 5000)
                )
                equity_bonus = underserved_nearby['weight'].sum()
            else:
                equity_bonus = 0
        
        equity_bonuses.append({
            'parking_id': parking_id,
            'equity_bonus': equity_bonus
        })
    
    # Merge bonuses with parking data
    equity_df = pd.DataFrame(equity_bonuses)
    parking_equity = parking_equity.merge(equity_df, on='parking_id', how='left')
    
    # Normalize equity bonus to 0-100 scale
    if parking_equity['equity_bonus'].max() > 0:
        parking_equity['equity_bonus_normalized'] = (
            parking_equity['equity_bonus'] / parking_equity['equity_bonus'].max() * 100
        )
    else:
        parking_equity['equity_bonus_normalized'] = 0
    
    # Calculate equity-adjusted score
    parking_equity['equity_adjusted_score'] = (
        parking_equity['accessibility_score'] * (1 - equity_weight) +
        parking_equity['equity_bonus_normalized'] * equity_weight
    )
    
    # Re-rank
    parking_equity = parking_equity.sort_values('equity_adjusted_score', ascending=False)
    parking_equity['equity_adjusted_rank'] = range(1, len(parking_equity) + 1)
    
    print(f"  ✓ Calculated equity bonuses for {len(parking_equity)} sites")
    print(f"  ✓ Sites with equity bonus: {(parking_equity['equity_bonus'] > 0).sum()}")
    print(f"  ✓ Mean bonus: {parking_equity['equity_bonus_normalized'].mean():.1f}")
    
    # Show rank changes (if we have original ranking)
    if 'rank_moderate' in parking_equity.columns:
        parking_equity['rank_change'] = (
            parking_equity['rank_moderate'] - 
            parking_equity['equity_adjusted_rank']
        )
        
        big_movers_up = parking_equity[parking_equity['rank_change'] > 10].nlargest(5, 'rank_change')
        if len(big_movers_up) > 0:
            print(f"\n  📈 Sites that rose most due to equity adjustment:")
            for idx, row in big_movers_up.iterrows():
                print(f"    {row['name']}: rank {int(row['rank_moderate'])} → {int(row['equity_adjusted_rank'])} (+{int(row['rank_change'])})")
    
    return parking_equity


# ============================================================================
# EQUITY METRICS CALCULATION
# ============================================================================

def calculate_equity_metrics(coverage_df: pd.DataFrame) -> Dict:
    """
    Calculate standard equity metrics
    
    Metrics:
        - Gini coefficient (inequality measure)
        - Coefficient of variation
        - Minimum coverage ratio
        - Coverage gap
        
    Args:
        coverage_df: Coverage data by zone
        
    Returns:
        Dictionary of equity metrics
    """
    print(f"\n📐 Calculating equity metrics...")
    
    coverage_scores = coverage_df['coverage_score'].values
    population = coverage_df['zone_population'].values
    
    # 1. Gini Coefficient (0 = perfect equality, 1 = perfect inequality)
    # https://en.wikipedia.org/wiki/Gini_coefficient
    sorted_scores = np.sort(coverage_scores)
    n = len(sorted_scores)
    cumsum = np.cumsum(sorted_scores)
    gini = (2 * np.sum((np.arange(1, n+1)) * sorted_scores)) / (n * np.sum(sorted_scores)) - (n + 1) / n
    
    # 2. Coefficient of Variation (std/mean)
    cv = coverage_scores.std() / coverage_scores.mean() if coverage_scores.mean() > 0 else 0
    
    # 3. Minimum coverage ratio (min/mean)
    min_coverage_ratio = coverage_scores.min() / coverage_scores.mean() if coverage_scores.mean() > 0 else 0
    
    # 4. Coverage gap (max - min)
    coverage_gap = coverage_scores.max() - coverage_scores.min()
    
    # 5. Population-weighted Gini
    # Zones with more population should count more
    if population.sum() > 0:
        pop_weights = population / population.sum()
        weighted_scores = coverage_scores * pop_weights
        sorted_weighted = np.sort(weighted_scores)
        cumsum_weighted = np.cumsum(sorted_weighted)
        gini_weighted = (2 * np.sum((np.arange(1, n+1)) * sorted_weighted)) / (n * np.sum(sorted_weighted)) - (n + 1) / n
    else:
        gini_weighted = gini
    
    metrics = {
        'gini_coefficient': gini,
        'gini_population_weighted': gini_weighted,
        'coefficient_of_variation': cv,
        'min_coverage_ratio': min_coverage_ratio,
        'coverage_gap': coverage_gap,
        'mean_coverage': coverage_scores.mean(),
        'median_coverage': np.median(coverage_scores),
        'min_coverage': coverage_scores.min(),
        'max_coverage': coverage_scores.max()
    }
    
    print(f"  ✓ Equity Metrics:")
    print(f"    Gini coefficient: {metrics['gini_coefficient']:.3f} (lower = more equal)")
    print(f"    Population-weighted Gini: {metrics['gini_population_weighted']:.3f}")
    print(f"    Coefficient of variation: {metrics['coefficient_of_variation']:.3f}")
    print(f"    Min/mean coverage ratio: {metrics['min_coverage_ratio']:.3f}")
    print(f"    Coverage gap: {metrics['coverage_gap']:.1f}")
    
    # Interpretation
    if metrics['gini_coefficient'] < 0.3:
        print(f"\n  ✅ LOW INEQUALITY: Coverage is relatively equitable")
    elif metrics['gini_coefficient'] < 0.5:
        print(f"\n  ⚠️  MODERATE INEQUALITY: Some areas underserved")
    else:
        print(f"\n  🔴 HIGH INEQUALITY: Significant spatial inequity")
    
    return metrics

def calculate_selection_equity_metrics(parking_selected: gpd.GeoDataFrame,
                                      zones_gdf: gpd.GeoDataFrame,
                                      top_n: int = 20) -> Dict:
    """
    Calculate equity metrics for a SELECTED subset of parking sites
    
    Uses EQUITY-ADJUSTED scores to show how equity weighting affects coverage.
    
    Args:
        parking_selected: Parking sites with equity_adjusted_score
        zones_gdf: Service zones with population
        top_n: Number of sites to evaluate (default: 20)
        
    Returns:
        Dictionary with selection equity metrics
    """
    print(f"\n📐 Calculating selection equity (top {top_n} sites)...")
    
    # CRITICAL: Use equity_adjusted_score for ranking
    if 'equity_adjusted_score' not in parking_selected.columns:
        print(f"  🔴 ERROR: equity_adjusted_score missing!")
        print(f"     Available columns: {parking_selected.columns.tolist()}")
        raise ValueError("Cannot calculate selection equity without equity_adjusted_score")

    score_column = 'equity_adjusted_score'
    print(f"  ✅ Using equity_adjusted_score for selection (equity weight affects this)")
    
    # Take top N sites by equity-adjusted score
    top_sites = parking_selected.nlargest(top_n, score_column)
    
    print(f"  • Selected top {len(top_sites)} sites by {score_column}")
    
    # Recalculate coverage using ONLY selected sites
    coverage_results = []
    
    for idx, zone in zones_gdf.iterrows():
        zone_id = zone['zone_id']
        zone_centroid = zone['zone_centroid']
        zone_pop = zone.get('zone_population', 0)
        
        # Find selected P&R sites within driving distance
        top_sites['distance_to_zone'] = top_sites.geometry.distance(zone_centroid)
        accessible_sites = top_sites[top_sites['distance_to_zone'] <= 5000]
        
        if len(accessible_sites) == 0:
            coverage_score = 0
        else:
            # CRITICAL: Use equity_adjusted_score in coverage calculation
            accessible_sites['coverage_contribution'] = (
                accessible_sites[score_column] *  # ← Use equity-adjusted score!
                (1 - accessible_sites['distance_to_zone'] / 5000)
            )
            coverage_score = accessible_sites['coverage_contribution'].sum()
        
        coverage_results.append({
            'zone_id': zone_id,
            'coverage_score_selected': coverage_score,
            'zone_population': zone_pop
        })
    
    selected_coverage_df = pd.DataFrame(coverage_results)
    
    # Calculate Gini for SELECTED sites
    coverage_scores = selected_coverage_df['coverage_score_selected'].values
    population = selected_coverage_df['zone_population'].values
    
    # Gini coefficient
    sorted_scores = np.sort(coverage_scores)
    n = len(sorted_scores)
    
    if np.sum(sorted_scores) > 0:
        cumsum = np.cumsum(sorted_scores)
        gini_selected = (2 * np.sum((np.arange(1, n+1)) * sorted_scores)) / (n * np.sum(sorted_scores)) - (n + 1) / n
    else:
        gini_selected = 1.0  # Worst case: no coverage
    
    # Population-weighted Gini
    if population.sum() > 0:
        pop_weights = population / population.sum()
        weighted_scores = coverage_scores * pop_weights
        sorted_weighted = np.sort(weighted_scores)
        
        if np.sum(sorted_weighted) > 0:
            gini_weighted_selected = (2 * np.sum((np.arange(1, n+1)) * sorted_weighted)) / (n * np.sum(sorted_weighted)) - (n + 1) / n
        else:
            gini_weighted_selected = 1.0
    else:
        gini_weighted_selected = gini_selected
    
    # Coverage statistics
    zones_covered = (selected_coverage_df['coverage_score_selected'] > 0).sum()
    total_zones = len(selected_coverage_df)
    
    pop_covered = selected_coverage_df[selected_coverage_df['coverage_score_selected'] > 0]['zone_population'].sum()
    pop_total = selected_coverage_df['zone_population'].sum()
    
    metrics = {
        'top_n_sites': top_n,
        'score_method': score_column,
        'gini_coefficient_selected': gini_selected,
        'gini_population_weighted_selected': gini_weighted_selected,
        'zones_covered': zones_covered,
        'zones_covered_pct': (zones_covered / total_zones * 100) if total_zones > 0 else 0,
        'population_covered': pop_covered,
        'population_covered_pct': (pop_covered / pop_total * 100) if pop_total > 0 else 0,
        'mean_coverage_selected': coverage_scores.mean(),
        'median_coverage_selected': np.median(coverage_scores),
        'min_coverage_selected': coverage_scores.min(),
        'max_coverage_selected': coverage_scores.max()
    }
    
    print(f"  ✓ Selection Equity Metrics (Top {top_n}):")
    print(f"    Score method: {score_column}")
    print(f"    Gini coefficient: {metrics['gini_coefficient_selected']:.3f}")
    print(f"    Zones covered: {metrics['zones_covered']} ({metrics['zones_covered_pct']:.1f}%)")
    print(f"    Population covered: {metrics['population_covered']:,.0f} ({metrics['population_covered_pct']:.1f}%)")
    
    return metrics, selected_coverage_df

# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def perform_equity_analysis(config_path: str = None):
    """
    Complete coverage-focused analysis for VW P&R site selection
    
    Goal: Ensure wide geographic distribution across Chattanooga
    
    Steps:
        1. Define service zones
        2. Calculate coverage per zone
        3. Identify coverage gaps
        4. Apply coverage bonuses
        5. Generate coverage metrics
        6. Export results
    """
    cfg = Config(config_path)
    
    print("\n" + "="*80)
    print("EQUITY-AWARE COVERAGE ANALYSIS")
    print("="*80)
    print(f"Project: {cfg.project_name}")
    print("Goal: Balance accessibility with spatial equity")
    print("="*80)
    
    start_time = datetime.now()
    
    # ========================================================================
    # STEP 1: Load data
    # ========================================================================
    print("\n📂 Loading processed datasets...")

    parking_gdf = gpd.read_file(cfg.paths.data_processed / "parking_lots_deduplicated.gpkg")
    census_gdf = gpd.read_file(cfg.paths.data_processed / "census_blocks_clean.gpkg")

    print(f"  ✓ Deduplicated parking lots: {len(parking_gdf)}")
    print(f"  ✓ Census blocks: {len(census_gdf)}")

    # Load accessibility scores
    accessibility_df = pd.read_csv(
        cfg.paths.data_outputs / "accessibility" / "accessibility_scores_moderate.csv"
    )

    print(f"  ✓ Accessibility scores: {len(accessibility_df)}")

    # Ensure parking_id is compatible type for merge
    if 'parking_id' in parking_gdf.columns:
        parking_gdf['parking_id'] = parking_gdf['parking_id'].astype(int)
    if 'parking_id' in accessibility_df.columns:
        accessibility_df['parking_id'] = accessibility_df['parking_id'].astype(int)

    # Merge accessibility scores with parking
    print(f"\n  🔗 Merging accessibility scores...")
    parking_before = len(parking_gdf)

    parking_gdf = parking_gdf.merge(
        accessibility_df[[
            'parking_id', 
            'accessibility_score', 
            'has_adequate_transit',
            'num_stops_800m',
            'total_daily_trips',
            'service_quality'
        ]],
        on='parking_id',
        how='left',
        suffixes=('', '_access')
    )

    print(f"    Before merge: {parking_before} sites")
    print(f"    After merge: {len(parking_gdf)} sites")

    # Check merge success
    if 'accessibility_score' not in parking_gdf.columns:
        print(f"\n  ❌ ERROR: Merge failed - accessibility_score not found!")
        raise ValueError("Failed to merge accessibility scores - check parking_id consistency")

    matched = parking_gdf['accessibility_score'].notna().sum()
    print(f"    Matched: {matched}/{len(parking_gdf)} sites ({matched/len(parking_gdf)*100:.1f}%)")

    if matched == 0:
        print(f"\n  ❌ ERROR: No accessibility scores matched!")
        raise ValueError("Zero accessibility scores matched - cannot proceed")

    # Fill missing values with 0
    parking_gdf['accessibility_score'] = parking_gdf['accessibility_score'].fillna(0)
    parking_gdf['has_adequate_transit'] = parking_gdf['has_adequate_transit'].fillna(False)
    parking_gdf['num_stops_800m'] = parking_gdf['num_stops_800m'].fillna(0)
    parking_gdf['total_daily_trips'] = parking_gdf['total_daily_trips'].fillna(0)
    parking_gdf['service_quality'] = parking_gdf['service_quality'].fillna('No Transit Available')

    print(f"\n  ✅ Data loaded successfully")
    print(f"     Sites with accessibility scores: {(parking_gdf['accessibility_score'] > 0).sum()}")
    print(f"     Sites with adequate transit: {parking_gdf['has_adequate_transit'].sum()}")
    
    # ========================================================================
    # STEP 2: Define service zones
    # ========================================================================
    zones_gdf = define_service_zones(census_gdf)
    
    # ========================================================================
    # STEP 3: Calculate coverage per zone
    # ========================================================================
    coverage_df = calculate_zone_coverage(parking_gdf, zones_gdf, drive_distance=5000)
    
    # ========================================================================
    # STEP 4: Identify underserved areas
    # ========================================================================
    coverage_with_geom, underserved_zones = identify_underserved_zones(
        coverage_df, 
        zones_gdf,
        threshold_percentile=25
    )

    # ========================================================================
    # STEP 5: Calculate equity adjustments
    # ========================================================================
    parking_equity = calculate_equity_adjustments(
        parking_gdf,
        coverage_with_geom,
        equity_weight=0.5  # 50% weight on equity
    )
    
    # ========================================================================
    # STEP 6: Calculate equity metrics (ALL candidate sites)
    # ========================================================================
    print("\n" + "="*80)
    print("CALCULATING EQUITY METRICS")
    print("="*80)

    print("\n📊 Coverage Equity (All Candidate Sites):")
    equity_metrics_all = calculate_equity_metrics(coverage_df)

    # ========================================================================
    # DIAGNOSTIC: Verify equity_adjusted_score exists
    # ========================================================================
    print("\n" + "="*80)
    print("DIAGNOSTIC: Verifying Equity Adjustment")
    print("="*80)

    if 'equity_adjusted_score' not in parking_equity.columns:
        print("🔴 CRITICAL ERROR: equity_adjusted_score not created!")
        raise ValueError("equity_adjusted_score missing")

    print(f"✅ equity_adjusted_score exists")

    # Compare rankings
    if 'accessibility_score' in parking_equity.columns:
        correlation = parking_equity[['accessibility_score', 'equity_adjusted_score']].corr().iloc[0, 1]
        print(f"\nCorrelation between scores: {correlation:.3f}")
        
        if correlation > 0.99:
            print("🔴 WARNING: Scores are nearly identical!")
        elif correlation > 0.95:
            print("⚠️  Scores are very similar")
        else:
            print("✅ Scores differ - equity adjustment is working!")
        
        # Compare top 20
        top20_access = set(parking_equity.nlargest(20, 'accessibility_score')['parking_id'])
        top20_equity = set(parking_equity.nlargest(20, 'equity_adjusted_score')['parking_id'])
        overlap = len(top20_access & top20_equity)
        
        print(f"\nTop 20 rankings:")
        print(f"  Sites in both rankings: {overlap}/20")
        print(f"  Sites that differ: {20-overlap}/20")
        
        if overlap == 20:
            print("  🔴 Rankings IDENTICAL")
        elif overlap >= 18:
            print("  ⚠️  Rankings nearly identical")
        elif overlap >= 15:
            print("  ✓ Some difference")
        else:
            print("  ✅ Significant difference - equity weight is working!")

    print("="*80 + "\n")

    # ========================================================================
    # STEP 7: Calculate selection equity (TOP N sites)
    # ========================================================================
    print("\n📊 Selection Equity (Impact of Site Selection):")

    selection_metrics = {}
    for top_n in [10, 20, 30]:
        metrics, _ = calculate_selection_equity_metrics(
            parking_equity,
            zones_gdf,
            top_n=top_n
        )
        selection_metrics[f'top_{top_n}'] = metrics

    # Compare: All sites vs. Top 20 selection
    print("\n" + "="*80)
    print("EQUITY COMPARISON")
    print("="*80)

    gini_all = equity_metrics_all['gini_coefficient']
    gini_top20 = selection_metrics['top_20']['gini_coefficient_selected']
    gini_improvement = ((gini_all - gini_top20) / gini_all * 100) if gini_all > 0 else 0

    print(f"\n🔍 Gini Coefficient Comparison:")
    print(f"  All candidate sites: {gini_all:.3f}")
    print(f"  Top 20 selected sites: {gini_top20:.3f}")
    print(f"  Improvement: {gini_improvement:+.1f}%")

    if gini_improvement > 5:
        print(f"  ✅ Selection reduces inequality by {gini_improvement:.0f}%")
    elif gini_improvement > 0:
        print(f"  ✓ Slight improvement in equity")
    else:
        print(f"  ⚠️  Selection does not improve equity (this is normal - see thesis interpretation)")

    # Combine metrics
    equity_metrics = {
        'all_sites': equity_metrics_all,
        'selected_sites': selection_metrics,
        'comparison': {
            'gini_all': gini_all,
            'gini_top20': gini_top20,
            'gini_improvement_pct': gini_improvement
        }
    }
    
    # ========================================================================
    # STEP 7: Export results
    # ========================================================================
    print("\n💾 Saving coverage analysis results...")
    
    results_dir = cfg.paths.data_outputs / "equity"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save coverage by zone (CSV)
    coverage_csv = results_dir / "zone_coverage.csv"
    coverage_df.to_csv(coverage_csv, index=False)
    print(f"  ✓ Zone coverage CSV: {coverage_csv}")

    # Save coverage with geometry (GPKG)
    coverage_gpkg = cfg.paths.data_processed / "zone_coverage.gpkg"
    coverage_to_save = coverage_with_geom.copy()
    if 'zone_centroid' in coverage_to_save.columns:
        coverage_to_save['zone_centroid_wkt'] = coverage_to_save['zone_centroid'].to_wkt()
        coverage_to_save = coverage_to_save.drop(columns=['zone_centroid'])
    coverage_to_save = coverage_to_save.set_geometry('geometry')
    coverage_to_save.to_file(coverage_gpkg, driver='GPKG')
    print(f"  ✓ Zone coverage GPKG: {coverage_gpkg}")

    # Save underserved zones (CSV)
    underserved_csv = results_dir / "underserved_zones.csv"
    underserved_zones.drop(columns=['geometry', 'zone_centroid'], errors='ignore').to_csv(underserved_csv, index=False)
    print(f"  ✓ Underserved zones CSV: {underserved_csv}")

    # Save underserved zones (GPKG)
    underserved_gpkg = cfg.paths.data_processed / "underserved_zones.gpkg"
    if len(underserved_zones) > 0:
        underserved_to_save = underserved_zones.copy()
        if 'zone_centroid' in underserved_to_save.columns:
            underserved_to_save = underserved_to_save.drop(columns=['zone_centroid'])
        if 'distance_to_site' in underserved_to_save.columns:
            underserved_to_save = underserved_to_save.drop(columns=['distance_to_site'])
        underserved_to_save = underserved_to_save.set_geometry('geometry')
        underserved_to_save.to_file(underserved_gpkg, driver='GPKG')
        print(f"  ✓ Underserved zones GPKG: {underserved_gpkg}")

    # Save equity-adjusted parking sites
    equity_gpkg = cfg.paths.data_processed / "parking_lots_equity_adjusted.gpkg"
    equity_to_save = parking_equity.copy()
    extra_cols = ['zone_centroid', 'distance_to_site', 'distance_to_zone']
    for col in extra_cols:
        if col in equity_to_save.columns:
            equity_to_save = equity_to_save.drop(columns=[col])
    equity_to_save = equity_to_save.set_geometry('geometry')
    equity_to_save.to_file(equity_gpkg, driver='GPKG')
    print(f"  ✓ Equity-adjusted sites GPKG: {equity_gpkg}")

    # Save equity metrics (JSON)
    metrics_json = results_dir / "equity_metrics.json"
    with open(metrics_json, 'w') as f:
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif pd.isna(obj):
                return None
            else:
                return obj
        
        # No import here - it's at the top of the file
        metrics_serializable = convert_to_serializable(equity_metrics)
        json.dump(metrics_serializable, f, indent=2)
    print(f"  ✓ Equity metrics JSON: {metrics_json}")
    
    # ========================================================================
    # STEP 8: Summary
    # ========================================================================
    duration = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "="*80)
    print("✅ EQUITY ANALYSIS COMPLETE")
    print("="*80)
    print(f"Processing time: {duration:.1f} seconds")

    print(f"\n📊 Coverage Summary:")
    print(f"  Service zones: {len(zones_gdf)}")
    print(f"  Zones with coverage: {(coverage_df['has_coverage']).sum()} ({(coverage_df['has_coverage']).sum()/len(zones_gdf)*100:.1f}%)")
    print(f"  Zones underserved: {len(underserved_zones)} ({len(underserved_zones)/len(zones_gdf)*100:.1f}%)")

    gini_all = equity_metrics['all_sites']['gini_coefficient']
    gini_top20 = equity_metrics['selected_sites']['top_20']['gini_coefficient_selected']
    gini_improvement = equity_metrics['comparison']['gini_improvement_pct']

    print(f"\n📊 Equity Metrics:")
    print(f"  Gini coefficient (all candidate sites): {gini_all:.3f}")
    print(f"  Gini coefficient (top 20 selected): {gini_top20:.3f}")
    print(f"  Equity change from selection: {gini_improvement:+.1f}%")

    if gini_all < 0.3:
        print(f"    → All sites: Low inequality")
    elif gini_all < 0.5:
        print(f"    → All sites: Moderate inequality")
    else:
        print(f"    → All sites: High inequality")

    if gini_improvement > 10:
        print(f"  ✅ Site selection significantly improves equity")
    elif gini_improvement > 5:
        print(f"  ✓ Site selection moderately improves equity")
    elif gini_improvement > 0:
        print(f"  ⚠️  Minimal equity improvement")
    else:
        print(f"  ℹ️  Selection increases concentration (normal for P&R optimization)")

    print(f"\n🎯 Equity-Adjusted Rankings:")
    sites_with_bonus = (parking_equity['equity_bonus'] > 0).sum()
    print(f"  Sites serving underserved areas: {sites_with_bonus} ({sites_with_bonus/len(parking_equity)*100:.1f}%)")

    if 'equity_bonus' in parking_equity.columns:
        avg_bonus = parking_equity[parking_equity['equity_bonus'] > 0]['equity_bonus_normalized'].mean()
        print(f"  Average equity bonus: {avg_bonus:.1f}/100")

    print(f"\n📋 Multi-Scale Selection Results:")
    for n in [10, 20, 30]:
        if f'top_{n}' in equity_metrics['selected_sites']:
            m = equity_metrics['selected_sites'][f'top_{n}']
            print(f"  Top {n}: Gini={m['gini_coefficient_selected']:.3f}, {m['zones_covered']} zones ({m['zones_covered_pct']:.1f}%)")

    print("\n" + "="*80)
    print("NEXT STEP: Final Site Selection & Optimization")
    print("="*80)
    print("Run: python src/05_optimization/final_selection.py")

    return parking_equity, coverage_with_geom, equity_metrics


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    """Run equity analysis"""
    
    try:
        parking_equity, coverage, metrics = perform_equity_analysis()
        print("\n✅ Equity analysis completed successfully")
        exit(0)
    except Exception as e:
        print(f"\n❌ Equity analysis failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)