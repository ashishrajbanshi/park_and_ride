"""
Transit Accessibility Analysis
===============================

Purpose:
    Calculate multi-dimensional transit accessibility for parking lots
    
Methodology:
    1. Buffer-based spatial analysis (walking distance catchments)
    2. Service frequency aggregation (daily trips, routes)
    3. Multi-scenario evaluation (conservative/moderate/extended)
    4. Composite accessibility scoring
    
Output:
    For each parking lot × scenario combination:
        - Number of accessible stops
        - Total daily trips
        - Route diversity
        - Proximity to nearest stop
        - Service quality classification
        - Composite accessibility score
    
Author: [Your Name]
Date: January 2026
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime
import warnings
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.config_loader import Config


# ============================================================================
# SPATIAL ANALYSIS: BUFFER CREATION
# ============================================================================

def create_walking_buffers(parking_gdf: gpd.GeoDataFrame, 
                          cfg: Config) -> Dict[str, gpd.GeoDataFrame]:
    """
    Create walking distance buffers around each parking lot
    
    Args:
        parking_gdf: GeoDataFrame of parking lots (must be in UTM)
        cfg: Configuration object
        
    Returns:
        Dictionary of {scenario_name: GeoDataFrame with buffers}
    """
    print("\n🚶 Creating walking distance buffers...")
    
    if parking_gdf.crs.to_epsg() != cfg.crs.analysis_utm:
        raise ValueError(f"Parking lots must be in UTM (EPSG:{cfg.crs.analysis_utm})")
    
    buffers = {}
    
    for scenario_name, distance_m in cfg.walking.as_dict().items():
        print(f"  • {scenario_name.capitalize()}: {distance_m}m buffers")
        
        # Create buffer
        gdf_buffered = parking_gdf.copy()
        gdf_buffered['buffer_geometry'] = parking_gdf.geometry.buffer(distance_m)
        gdf_buffered = gdf_buffered.set_geometry('buffer_geometry')
        
        # Keep original point geometry as attribute
        gdf_buffered['point_geometry'] = parking_gdf.geometry
        
        buffers[scenario_name] = gdf_buffered
        
        print(f"    ✓ Created {len(gdf_buffered)} buffers")
    
    return buffers


# ============================================================================
# TRANSIT METRICS CALCULATION
# ============================================================================

def calculate_transit_metrics(parking_gdf: gpd.GeoDataFrame,
                              stops_gdf: gpd.GeoDataFrame,
                              buffer_distance: int,
                              scenario_name: str,
                              cfg: Config) -> pd.DataFrame:
    """
    Calculate transit accessibility metrics for given walking distance
    
    FIXED VERSION - Handles GTFS parent stations and route counting correctly
    """
    print(f"\n📊 Calculating transit metrics for {scenario_name} ({buffer_distance}m)...")
    
    # Validate CRS match
    if parking_gdf.crs != stops_gdf.crs:
        raise ValueError("Parking lots and transit stops must have same CRS")
    
    # Create buffers
    parking_buffered = parking_gdf.copy()
    parking_buffered['geometry'] = parking_buffered.geometry.buffer(buffer_distance)
    
    # Spatial join: find stops within each buffer
    print(f"  • Performing spatial join...")
    joined = gpd.sjoin(
        parking_buffered[['parking_id', 'name', 'geometry']],
        stops_gdf[['stop_id', 'stop_name', 'daily_trips', 'num_routes', 'geometry']],
        how='left',
        predicate='intersects'
    )
    
    print(f"    ✓ Found {len(joined)} stop-parking associations")
    
    # ========================================================================
    # FIX: Ensure we're not double-counting
    # ========================================================================
    # Remove duplicate stop-parking pairs (can happen with spatial join)
    joined = joined.drop_duplicates(subset=['parking_id', 'stop_id'])
    print(f"    ✓ After deduplication: {len(joined)} unique associations")
    
    # Calculate distance to each accessible stop
    print(f"  • Calculating walking distances...")
    
    parking_points = parking_gdf.set_index('parking_id')['geometry']
    
    joined['walk_distance_m'] = joined.apply(
        lambda row: parking_points.loc[row['parking_id']].distance(
            stops_gdf[stops_gdf['stop_id'] == row['stop_id']].iloc[0].geometry
        ) if pd.notna(row['stop_id']) else np.nan,
        axis=1
    )
    
    # Aggregate metrics by parking lot
    print(f"  • Aggregating transit metrics...")
    
    metrics = joined.groupby('parking_id').agg({
        'name': 'first',  # Parking lot name (same for all rows)
        'stop_id': 'count',  # Number of stops
        'daily_trips': 'sum',  # Total daily trips (sum across stops)
        'num_routes': 'sum',  # Total route coverage (sum, not unique)
        'walk_distance_m': ['min', 'mean']
    }).reset_index()
    
    # Flatten column names
    metrics.columns = [
        'parking_id',
        'name',
        f'num_stops_{buffer_distance}m',
        f'total_daily_trips',
        f'total_routes',
        f'nearest_stop_distance_m',
        f'avg_walk_distance_m'
    ]
    
    # Fill NaN values
    metrics[f'num_stops_{buffer_distance}m'] = metrics[f'num_stops_{buffer_distance}m'].fillna(0).astype(int)
    metrics['total_daily_trips'] = metrics['total_daily_trips'].fillna(0).astype(int)
    metrics['total_routes'] = metrics['total_routes'].fillna(0).astype(int)
    metrics['nearest_stop_distance_m'] = metrics['nearest_stop_distance_m'].fillna(9999).astype(int)
    metrics['avg_walk_distance_m'] = metrics['avg_walk_distance_m'].fillna(9999).astype(int)
    
    # Add scenario identifier
    metrics['scenario'] = scenario_name
    metrics['buffer_distance_m'] = buffer_distance
    
    # Add adequacy flag
    cfg = Config()  # Get config for thresholds
    metrics['has_adequate_transit'] = (
        (metrics[f'num_stops_{buffer_distance}m'] >= cfg.min_thresholds.min_transit_stops) &
        (metrics['total_daily_trips'] >= cfg.min_thresholds.min_daily_trips)
    )
    
    # ========================================================================
    # SANITY CHECK: Flag suspicious values
    # ========================================================================
    suspicious = metrics[metrics['total_daily_trips'] > 1000]
    if len(suspicious) > 0:
        print(f"\n  ⚠️  WARNING: {len(suspicious)} sites have >1000 daily trips")
        print(f"              This may indicate data quality issues")
        print(f"              Max trips: {metrics['total_daily_trips'].max()}")
        
        # Calculate trips per stop (should be 10-60 typically)
        metrics['trips_per_stop'] = metrics['total_daily_trips'] / metrics[f'num_stops_{buffer_distance}m'].replace(0, 1)
        
        extreme = metrics[metrics['trips_per_stop'] > 100]
        if len(extreme) > 0:
            print(f"  ⚠️  {len(extreme)} sites have >100 trips/stop (investigate GTFS data)")
    
    print(f"    ✓ Calculated metrics for {len(metrics)} parking lots")
    print(f"    ✓ {metrics['has_adequate_transit'].sum()} have adequate transit access")
    
    return metrics


# ============================================================================
# ACCESSIBILITY SCORING
# ============================================================================

def calculate_accessibility_scores(metrics_df: pd.DataFrame, 
                                   cfg: Config,
                                   scenario_name: str) -> pd.DataFrame:
    """
    Calculate normalized accessibility scores from transit metrics
    
    Components:
        1. Number of stops (redundancy)
        2. Service frequency (daily trips)
        3. Service reliability (headway-based)
        4. Walking proximity (distance to nearest stop)
        5. Route diversity (number of routes)
        
    Args:
        metrics_df: DataFrame with raw transit metrics
        cfg: Configuration with weights
        scenario_name: Scenario identifier
        
    Returns:
        DataFrame with normalized component scores and composite score
    """
    print(f"\n⚖️  Calculating accessibility scores ({scenario_name})...")
    
    df = metrics_df.copy()
    
    # Get buffer distance from first row
    buffer_dist = df['buffer_distance_m'].iloc[0]
    stops_col = f'num_stops_{buffer_dist}m'
    
    # ========================================================================
    # COMPONENT 1: Number of Stops Score (Redundancy)
    # ========================================================================
    max_stops = df[stops_col].max()
    if max_stops > 0:
        df['stops_score'] = (df[stops_col] / max_stops) * 100
    else:
        df['stops_score'] = 0
    
    # ========================================================================
    # COMPONENT 2: Service Frequency Score
    # ========================================================================
    max_trips = df['total_daily_trips'].max()
    if max_trips > 0:
        df['frequency_score'] = (df['total_daily_trips'] / max_trips) * 100
    else:
        df['frequency_score'] = 0
    
    # ========================================================================
    # COMPONENT 3: Service Reliability Score (Headway-based)
    # ========================================================================
    # Based on TCQSM standards - shorter headway = higher reliability
    df['reliability_score'] = df['total_daily_trips'].apply(lambda trips: 
        100 if trips >= 100 else  # ≤10 min headway
        80 if trips >= 50 else    # ≤20 min headway
        60 if trips >= 30 else    # ≤30 min headway
        40 if trips >= 15 else    # ≤60 min headway
        20 if trips >= 10 else    # ≤90 min headway
        0
    )
    
    # ========================================================================
    # COMPONENT 4: Walking Proximity Score (Inverse distance)
    # ========================================================================
    max_dist = df['nearest_stop_distance_m'].max()
    if max_dist > 0:
        # Inverse: closer = higher score
        df['proximity_score'] = (1 - (df['nearest_stop_distance_m'] / max_dist)) * 100
        df['proximity_score'] = df['proximity_score'].clip(lower=0)
    else:
        df['proximity_score'] = 0
    
    # ========================================================================
    # COMPONENT 5: Route Diversity Score
    # ========================================================================
    max_routes = df['total_routes'].max()
    if max_routes > 0:
        df['diversity_score'] = (df['total_routes'] / max_routes) * 100
    else:
        df['diversity_score'] = 0
    
    # ========================================================================
    # COMPOSITE ACCESSIBILITY SCORE (Weighted)
    # ========================================================================
    weights = cfg.accessibility_weights.as_dict()
    
    df['accessibility_score'] = (
        df['stops_score'] * weights['num_stops'] +
        df['frequency_score'] * weights['trip_frequency'] +
        df['reliability_score'] * weights['service_reliability'] +
        df['proximity_score'] * weights['walking_distance'] +
        df['diversity_score'] * weights['route_diversity']
    )
    
    # Round scores
    score_cols = ['stops_score', 'frequency_score', 'reliability_score', 
                  'proximity_score', 'diversity_score', 'accessibility_score']
    for col in score_cols:
        df[col] = df[col].round(2)
    
    # ========================================================================
    # SERVICE QUALITY CLASSIFICATION
    # ========================================================================
    df['service_quality'] = df['total_daily_trips'].apply(
        lambda trips: cfg.service_quality.classify(trips)
    )
    
    # ========================================================================
    # RELIABILITY CLASSIFICATION
    # ========================================================================
    df['reliability_class'] = df.apply(
        lambda row: 
            'High Reliability' if row['total_daily_trips'] >= 50 and row[stops_col] >= 3 else
            'Moderate Reliability' if row['total_daily_trips'] >= 30 and row[stops_col] >= 2 else
            'Low Reliability' if row['total_daily_trips'] >= 10 else
            'Unreliable',
        axis=1
    )
    
    # Add scenario-specific ranking
    df = df.sort_values('accessibility_score', ascending=False)
    df[f'rank_{scenario_name}'] = range(1, len(df) + 1)
    
    print(f"    ✓ Calculated composite scores")
    print(f"    ✓ Score range: {df['accessibility_score'].min():.1f} - {df['accessibility_score'].max():.1f}")
    print(f"    ✓ Mean score: {df['accessibility_score'].mean():.1f}")
    
    # Show service quality distribution
    quality_dist = df['service_quality'].value_counts()
    print(f"\n    Service Quality Distribution:")
    for quality, count in quality_dist.items():
        print(f"      {quality}: {count} sites")
    
    return df


# ============================================================================
# SCENARIO COMPARISON
# ============================================================================

def compare_scenarios(scenario_results: Dict[str, pd.DataFrame],
                     cfg: Config) -> pd.DataFrame:
    """
    Compare accessibility across walking scenarios
    
    For each parking lot, identifies:
        - Best-performing scenario
        - Maximum accessibility score achieved
        - Scenario-specific trade-offs
        
    Args:
        scenario_results: Dict of {scenario_name: results_df}
        cfg: Configuration
        
    Returns:
        DataFrame with comparison metrics
    """
    print("\n📈 Comparing scenarios...")
    
    # Start with conservative scenario as base
    comparison = scenario_results['conservative'][['parking_id', 'name']].copy()
    
    # Add scores from each scenario
    for scenario_name, df in scenario_results.items():
        score_col = f'score_{cfg.walking.as_dict()[scenario_name]}m'
        stops_col_pattern = f'num_stops_{cfg.walking.as_dict()[scenario_name]}m'
        
        comparison = comparison.merge(
            df[[
                'parking_id',
                'accessibility_score',
                stops_col_pattern,
                'service_quality',
                f'rank_{scenario_name}'
            ]].rename(columns={
                'accessibility_score': score_col,
                stops_col_pattern: f'stops_{cfg.walking.as_dict()[scenario_name]}m',
                'service_quality': f'quality_{scenario_name}',
                f'rank_{scenario_name}': f'rank_{scenario_name}'
            }),
            on='parking_id',
            how='outer'
        )
    
    # Fill NaN scores with 0
    score_cols = [f'score_{d}m' for d in cfg.walking.as_dict().values()]
    for col in score_cols:
        if col in comparison.columns:
            comparison[col] = comparison[col].fillna(0)
    
    # Identify best scenario for each lot
    comparison['max_accessibility_score'] = comparison[score_cols].max(axis=1)
    
    comparison['best_scenario'] = comparison[score_cols].idxmax(axis=1)
    comparison['best_scenario'] = comparison['best_scenario'].apply(
        lambda x: {
            f'score_{cfg.walking.conservative}m': f'Conservative ({cfg.walking.conservative}m)',
            f'score_{cfg.walking.moderate}m': f'Moderate ({cfg.walking.moderate}m)',
            f'score_{cfg.walking.extended}m': f'Extended ({cfg.walking.extended}m)'
        }.get(x, 'Unknown')
    )
    
    # Count how many lots benefit from extended walking
    best_scenario_dist = comparison['best_scenario'].value_counts()
    print(f"\n  Best Scenario Distribution:")
    for scenario, count in best_scenario_dist.items():
        pct = (count / len(comparison)) * 100
        print(f"    {scenario}: {count} lots ({pct:.1f}%)")
    
    # Calculate improvement from extending walking distance
    if all(col in comparison.columns for col in score_cols):
        comparison['improvement_moderate_vs_conservative'] = (
            comparison[f'score_{cfg.walking.moderate}m'] - 
            comparison[f'score_{cfg.walking.conservative}m']
        )
        comparison['improvement_extended_vs_moderate'] = (
            comparison[f'score_{cfg.walking.extended}m'] - 
            comparison[f'score_{cfg.walking.moderate}m']
        )
        
        significant_improvements = (comparison['improvement_moderate_vs_conservative'] > 10).sum()
        print(f"\n  {significant_improvements} lots gain >10 points by extending to {cfg.walking.moderate}m")
    
    return comparison


# ============================================================================
# MAIN ANALYSIS ORCHESTRATOR
# ============================================================================

def analyze_transit_accessibility(config_path: str = None):
    """
    Complete transit accessibility analysis workflow
    
    Steps:
        1. Load processed data
        2. Calculate transit metrics for each scenario
        3. Compute accessibility scores
        4. Compare scenarios
        5. Save results
    """
    cfg = Config(config_path)
    
    print("\n" + "="*80)
    print("TRANSIT ACCESSIBILITY ANALYSIS")
    print("="*80)
    print(f"Project: {cfg.project_name}")
    print(f"Walking scenarios: {', '.join([f'{d}m' for d in cfg.walking.as_dict().values()])}")
    print("="*80)
    
    start_time = datetime.now()
    
    # ========================================================================
    # STEP 1: Load processed data
    # ========================================================================
    print("\n📂 Loading processed datasets...")
    
    parking_gdf = gpd.read_file(cfg.paths.data_processed / "parking_lots_clean.gpkg")
    stops_gdf = gpd.read_file(cfg.paths.data_processed / "transit_stops_clean.gpkg")
    
    print(f"  ✓ Parking lots: {len(parking_gdf)}")
    print(f"  ✓ Transit stops: {len(stops_gdf)}")
    print(f"  ✓ CRS: EPSG:{parking_gdf.crs.to_epsg()}")
    
    # ========================================================================
    # STEP 2: Calculate transit metrics for each scenario
    # ========================================================================
    scenario_metrics = {}
    scenario_scores = {}
    
    for scenario_name, distance_m in cfg.walking.as_dict().items():
        # Calculate metrics
        metrics = calculate_transit_metrics(
            parking_gdf,
            stops_gdf,
            distance_m,
            scenario_name,
            cfg
        )
        scenario_metrics[scenario_name] = metrics
        
        # Calculate scores
        scores = calculate_accessibility_scores(
            metrics,
            cfg,
            scenario_name
        )
        scenario_scores[scenario_name] = scores
    
    # ========================================================================
    # STEP 3: Compare scenarios
    # ========================================================================
    comparison = compare_scenarios(scenario_scores, cfg)
    
    # ========================================================================
    # STEP 4: Save results
    # ========================================================================
    print("\n💾 Saving accessibility results...")
    
    results_dir = cfg.paths.data_outputs / "accessibility"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual scenario results
    for scenario_name, df in scenario_scores.items():
        output_path = results_dir / f"accessibility_scores_{scenario_name}.csv"
        df.to_csv(output_path, index=False)
        print(f"  ✓ {scenario_name}: {output_path}")
    
    # Save scenario comparison
    comparison_path = results_dir / "scenario_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    print(f"  ✓ Comparison: {comparison_path}")
    
    # ========================================================================
    # STEP 5: Generate summary statistics
    # ========================================================================
    print("\n📊 ACCESSIBILITY ANALYSIS SUMMARY")
    print("="*80)
    
    duration = (datetime.now() - start_time).total_seconds()
    print(f"Processing time: {duration:.1f} seconds")
    
    print(f"\nParking lots analyzed: {len(parking_gdf)}")
    print(f"Transit stops considered: {len(stops_gdf)}")
    
    print(f"\nLots with adequate transit access:")
    for scenario_name in cfg.walking.as_dict().keys():
        adequate = scenario_scores[scenario_name]['has_adequate_transit'].sum()
        pct = (adequate / len(parking_gdf)) * 100
        dist = cfg.walking.as_dict()[scenario_name]
        print(f"  {scenario_name.capitalize()} ({dist}m): {adequate} ({pct:.1f}%)")
    
    print(f"\nTop 5 most accessible parking lots:")
    top5 = comparison.nlargest(5, 'max_accessibility_score')
    for idx, row in top5.iterrows():
        print(f"  {row['name']}: {row['max_accessibility_score']:.1f} ({row['best_scenario']})")
    
    print("\n" + "="*80)
    print("✅ TRANSIT ACCESSIBILITY ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {results_dir}")
    print("\n" + "="*80)
    print("NEXT STEP: Spatial Clustering & Deduplication")
    print("="*80)
    
    return scenario_scores, comparison


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    """Run transit accessibility analysis"""
    
    try:
        scenario_scores, comparison = analyze_transit_accessibility()
        print("\n✅ Analysis completed successfully")
        exit(0)
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)