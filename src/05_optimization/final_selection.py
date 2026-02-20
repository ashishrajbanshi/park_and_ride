import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config_loader import Config


# ============================================================================
# VW-SPECIFIC CONSTRAINTS
# ============================================================================

def apply_vw_distance_constraints(parking_gdf: gpd.GeoDataFrame,
                                 cfg: Config) -> gpd.GeoDataFrame:
    """
    Calculate distance from VW plant and apply scoring
    
    Distance zones:
        0-5km: Exclusion (too close, no modal shift)
        8-15km: Optimal (sweet spot)
        15-25km: Acceptable (diminishing returns)
        >25km: Poor (too far)
    
    Args:
        parking_gdf: Parking sites with equity-adjusted scores
        cfg: Configuration
        
    Returns:
        GeoDataFrame with VW distance scoring
    """
    print("\n📍 Calculating distance from VW plant...")
    plant = cfg.vw_plant


    # VW plant location
    vw_lat = plant.latitude
    vw_lon = plant.longitude

    
    # Create VW point geometry (in WGS84)
    from shapely.geometry import Point
    vw_point_wgs84 = Point(vw_lon, vw_lat)
    
    # Convert to UTM for distance calculation
    vw_gdf = gpd.GeoDataFrame(
        {'name': ['VW Plant']},
        geometry=[vw_point_wgs84],
        crs='EPSG:4326'
    ).to_crs(parking_gdf.crs)
    
    vw_point_utm = vw_gdf.geometry.iloc[0]
    
    # Calculate distance
    parking_gdf['distance_to_vw_m'] = parking_gdf.geometry.distance(vw_point_utm)
    parking_gdf['distance_to_vw_km'] = parking_gdf['distance_to_vw_m'] / 1000
    
    print(f"  ✓ Distance calculated for {len(parking_gdf)} sites")
    print(f"  ✓ Range: {parking_gdf['distance_to_vw_km'].min():.1f} - {parking_gdf['distance_to_vw_km'].max():.1f} km")
    
    # Apply distance scoring
    def score_vw_distance(dist_km):
        """Score based on VW distance zones"""
        if dist_km < 5:
            # Too close - linear penalty
            return (dist_km / 5) * 50  # 0-50 score
        elif 8 <= dist_km <= 15:
            # Optimal zone
            return 100
        elif 5 <= dist_km < 8:
            # Approaching optimal
            return 50 + ((dist_km - 5) / 3) * 50  # 50-100 score
        elif 15 < dist_km <= 25:
            # Acceptable but diminishing
            return 100 - ((dist_km - 15) / 10) * 30  # 100-70 score
        else:
            # Too far
            return max(0, 70 - ((dist_km - 25) / 10) * 20)  # <70 score
    
    parking_gdf['vw_distance_score'] = parking_gdf['distance_to_vw_km'].apply(score_vw_distance)
    
    # Classify zones
    def classify_vw_zone(dist_km):
        if dist_km < 5:
            return 'Exclusion (<5km)'
        elif dist_km < 8:
            return 'Suboptimal (5-8km)'
        elif dist_km <= 15:
            return 'Optimal (8-15km)'
        elif dist_km <= 25:
            return 'Acceptable (15-25km)'
        else:
            return 'Poor (>25km)'
    
    parking_gdf['vw_distance_zone'] = parking_gdf['distance_to_vw_km'].apply(classify_vw_zone)
    
    # Show distribution
    print(f"\n  📊 Distance Zone Distribution:")
    zone_dist = parking_gdf['vw_distance_zone'].value_counts().sort_index()
    for zone, count in zone_dist.items():
        pct = (count / len(parking_gdf)) * 100
        print(f"    {zone}: {count} ({pct:.1f}%)")
    
    return parking_gdf


# ============================================================================
# COMPOSITE FINAL SCORING
# ============================================================================

def calculate_final_scores(parking_gdf: gpd.GeoDataFrame,
                          cfg: Config,
                          equity_influence: float = 0.5) -> gpd.GeoDataFrame:
    """
    Calculate final composite scores combining all factors
    
    Components:
        - Equity-adjusted accessibility (from Step 4)
        - VW distance optimality
        - Transit service quality
        - Parking capacity
        
    Args:
        parking_gdf: Sites with all metrics
        cfg: Configuration
        equity_influence: 0-1, how much equity affects final score
        
    Returns:
        GeoDataFrame with final scores
    """
    print(f"\n🎯 Calculating final composite scores (equity influence: {equity_influence})...")
    
    # Normalize capacity (0-100)
    if 'capacity' in parking_gdf.columns:
        max_capacity = parking_gdf['capacity'].max()
        parking_gdf['capacity_score'] = (parking_gdf['capacity'] / max_capacity * 100) if max_capacity > 0 else 50
    else:
        parking_gdf['capacity_score'] = 50
    
    # Normalize transit trips (already have accessibility_score, but add trips for emphasis)
    if 'total_daily_trips' in parking_gdf.columns:
        max_trips = parking_gdf['total_daily_trips'].max()
        parking_gdf['transit_score'] = (parking_gdf['total_daily_trips'] / max_trips * 100) if max_trips > 0 else 0
    else:
        parking_gdf['transit_score'] = parking_gdf.get('accessibility_score', 0)
    
    # Final composite score
    # Weights: Equity-adjusted accessibility (50%), VW distance (25%), Transit (15%), Capacity (10%)
    parking_gdf['final_score'] = (
        parking_gdf['equity_adjusted_score'] * 0.50 +
        parking_gdf['vw_distance_score'] * 0.25 +
        parking_gdf['transit_score'] * 0.15 +
        parking_gdf['capacity_score'] * 0.10
    )
    
    # Rank by final score
    parking_gdf = parking_gdf.sort_values('final_score', ascending=False)
    parking_gdf['final_rank'] = range(1, len(parking_gdf) + 1)
    
    print(f"  ✓ Final scores calculated")
    print(f"  ✓ Score range: {parking_gdf['final_score'].min():.1f} - {parking_gdf['final_score'].max():.1f}")
    print(f"  ✓ Mean score: {parking_gdf['final_score'].mean():.1f}")
    
    return parking_gdf


# ============================================================================
# TIER CLASSIFICATION
# ============================================================================

def assign_implementation_tiers(parking_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Classify sites into implementation tiers
    
    Tier 1 (Immediate): Top sites, high accessibility, in optimal VW zone
    Tier 2 (Short-term): Good sites, may need minor improvements
    Tier 3 (Long-term): Equity sites, require transit expansion
    
    Args:
        parking_gdf: Sites with final scores
        
    Returns:
        GeoDataFrame with tier assignments
    """
    print("\n🏆 Assigning implementation tiers...")
    
    def assign_tier(row):
        """Tier assignment logic"""
        # Tier 1: High score + optimal VW zone + adequate transit
        if (row['final_score'] >= 70 and 
            row.get('vw_distance_zone', '') == 'Optimal (8-15km)' and
            row.get('has_adequate_transit', False)):
            return 'Tier 1: Immediate'
        
        # Tier 2: Good score + acceptable location
        elif (row['final_score'] >= 50 and
              row.get('vw_distance_zone', '') in ['Optimal (8-15km)', 'Acceptable (15-25km)', 'Suboptimal (5-8km)']):
            return 'Tier 2: Short-term'
        
        # Tier 3: Equity sites or requiring improvements
        else:
            return 'Tier 3: Long-term'
    
    parking_gdf['implementation_tier'] = parking_gdf.apply(assign_tier, axis=1)
    
    # Show distribution
    tier_dist = parking_gdf['implementation_tier'].value_counts().sort_index()
    print(f"\n  📊 Implementation Tier Distribution:")
    for tier, count in tier_dist.items():
        pct = (count / len(parking_gdf)) * 100
        avg_score = parking_gdf[parking_gdf['implementation_tier'] == tier]['final_score'].mean()
        print(f"    {tier}: {count} sites ({pct:.1f}%), avg score: {avg_score:.1f}")
    
    return parking_gdf


# ============================================================================
# TOP N SELECTION
# ============================================================================

def select_top_sites(parking_gdf: gpd.GeoDataFrame,
                    top_n_list: List[int] = [10, 20, 30]) -> Dict[int, gpd.GeoDataFrame]:
    """
    Select top N sites for different scenarios
    
    Args:
        parking_gdf: All sites with final scores
        top_n_list: List of top N values to generate
        
    Returns:
        Dictionary of {N: GeoDataFrame of top N sites}
    """
    print(f"\n🎯 Selecting top sites...")
    
    selections = {}
    
    for n in top_n_list:
        top_n = parking_gdf.nlargest(n, 'final_score').copy()
        top_n['selection_group'] = f'Top {n}'
        selections[n] = top_n
        
        # Calculate summary stats
        total_capacity = top_n['capacity'].sum() if 'capacity' in top_n.columns else 0
        avg_score = top_n['final_score'].mean()
        with_transit = top_n.get('has_adequate_transit', pd.Series([False]*len(top_n))).sum()
        
        print(f"\n  Top {n} sites:")
        print(f"    Total capacity: {total_capacity:,} spaces")
        print(f"    Average score: {avg_score:.1f}")
        print(f"    With adequate transit: {with_transit}/{n}")
        
        # Tier breakdown
        if 'implementation_tier' in top_n.columns:
            tier_breakdown = top_n['implementation_tier'].value_counts()
            for tier, count in tier_breakdown.items():
                print(f"      {tier}: {count} sites")
    
    return selections



# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_final_selections(parking_final: gpd.GeoDataFrame,
                           top_selections: Dict[int, gpd.GeoDataFrame],
                           cfg: Config):
    """
    Export all final outputs for ArcGIS and reporting
    
    Creates:
        - Top N site selections (GPKG + CSV)
        - Implementation tiers (GPKG + CSV)
        - Summary report (JSON)
    """
    print("\n💾 Exporting final selections...")
    
    results_dir = cfg.paths.data_outputs / "final_selection"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # 1. Save complete final dataset
    # ========================================================================
    final_gpkg = cfg.paths.data_processed / "parking_lots_final_scored.gpkg"
    
    # Clean for export
    final_export = parking_final.copy()
    extra_cols = ['zone_centroid', 'distance_to_site', 'weight']
    for col in extra_cols:
        if col in final_export.columns:
            final_export = final_export.drop(columns=[col])
    
    final_export.to_file(final_gpkg, driver='GPKG')
    print(f"  ✓ Final scored sites: {final_gpkg}")
    
    # CSV version
    final_csv = results_dir / "final_scores_all_sites.csv"
    final_export.drop(columns=['geometry'], errors='ignore').to_csv(final_csv, index=False)
    print(f"  ✓ Final scores CSV: {final_csv}")
    
    # ========================================================================
    # 2. Save top N selections
    # ========================================================================
    for n, top_n_gdf in top_selections.items():
        # GPKG for GIS
        top_n_gpkg = cfg.paths.data_processed / f"parkride_top{n}.gpkg"
        top_n_clean = top_n_gdf.copy()
        for col in extra_cols:
            if col in top_n_clean.columns:
                top_n_clean = top_n_clean.drop(columns=[col])
        
        top_n_clean.to_file(top_n_gpkg, driver='GPKG')
        print(f"  ✓ Top {n} sites (GPKG): {top_n_gpkg}")
        
        # CSV for easy viewing
        top_n_csv = results_dir / f"top{n}_sites.csv"
        top_n_clean.drop(columns=['geometry'], errors='ignore').to_csv(top_n_csv, index=False)
        print(f"  ✓ Top {n} sites (CSV): {top_n_csv}")
    
    # ========================================================================
    # 3. Save implementation tiers
    # ========================================================================
    tiers_gpkg = cfg.paths.data_processed / "parkride_implementation_tiers.gpkg"
    parking_final.to_file(tiers_gpkg, driver='GPKG')
    print(f"  ✓ Implementation tiers: {tiers_gpkg}")

    
    # ========================================================================
    # 5. Generate summary report
    # ========================================================================
    summary = {
        'analysis_date': datetime.now().isoformat(),
        'total_sites_analyzed': len(parking_final),
        'top_selections': {
            f'top_{n}': {
                'num_sites': len(gdf),
                'total_capacity': int(gdf.get('capacity', pd.Series([50]*len(gdf))).sum()),
                'avg_final_score': float(gdf['final_score'].mean()),
                'score_range': [float(gdf['final_score'].min()), float(gdf['final_score'].max())]
            }
            for n, gdf in top_selections.items()
        },
        'implementation_tiers': parking_final['implementation_tier'].value_counts().to_dict(),
    }
    
    summary_json = results_dir / "selection_summary.json"
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Summary report: {summary_json}")


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def perform_final_selection(config_path: str = None):
    
    cfg = Config(config_path)
    
    print("\n" + "="*80)
    print("FINAL SITE SELECTION & OPTIMIZATION")
    print("="*80)
    print(f"Project: {cfg.project_name}")
    print("="*80)
    
    start_time = datetime.now()
    
    # ========================================================================
    # STEP 1: Load equity-adjusted sites
    # ========================================================================
    print("\n📂 Loading equity-adjusted sites...")
    
    parking_gdf = gpd.read_file(cfg.paths.data_processed / "parking_lots_equity_adjusted.gpkg")
    
    print(f"  ✓ Loaded {len(parking_gdf)} sites")
    print(f"  ✓ With equity-adjusted scores: {(parking_gdf['equity_adjusted_score'] > 0).sum()}")
    
    # ========================================================================
    # STEP 2: Apply VW distance constraints
    # ========================================================================
    parking_gdf = apply_vw_distance_constraints(parking_gdf, cfg)
    
    # ========================================================================
    # STEP 3: Calculate final scores
    # ========================================================================
    parking_gdf = calculate_final_scores(parking_gdf, cfg, equity_influence=0.5)
    
    # ========================================================================
    # STEP 4: Assign implementation tiers
    # ========================================================================
    parking_gdf = assign_implementation_tiers(parking_gdf)
    
    # ========================================================================
    # STEP 5: Select top N sites
    # ========================================================================
    top_selections = select_top_sites(parking_gdf, top_n_list=[10, 20, 30])
    
    
    
    # ========================================================================
    # STEP 8: Final summary
    # ========================================================================
    duration = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "="*80)
    print("✅ FINAL SELECTION COMPLETE")
    print("="*80)
    print(f"Processing time: {duration:.1f} seconds")
    
    print(f"\n📊 Final Statistics:")
    print(f"  Total sites analyzed: {len(parking_gdf)}")
    print(f"  Sites in optimal VW zone: {(parking_gdf['vw_distance_zone'] == 'Optimal (8-15km)').sum()}")
    print(f"  Sites with adequate transit: {parking_gdf.get('has_adequate_transit', pd.Series([False]*len(parking_gdf))).sum()}")
    
    print(f"\n🏆 Top 10 Recommended Sites:")
    top10 = parking_gdf.nlargest(10, 'final_score')
    for idx, row in top10.iterrows():
        print(f"  {int(row['final_rank'])}. {row['name'][:50]}")
        print(f"     Score: {row['final_score']:.1f} | {row['implementation_tier']} | {row['vw_distance_zone']}")
    
    print("\n" + "="*80)
    print("📁 OUTPUTS READY FOR ARCGIS PRO")
    print("="*80)
    print(f"  Processed data: {cfg.paths.data_processed}")
    print(f"  Final results: {cfg.paths.data_outputs / 'final_selection'}")
    
    print("\n" + "="*80)
    print("NEXT: Load data in ArcGIS Pro for mapping")
    print("="*80)
    
    return parking_gdf, top_selections

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    """Run final site selection"""
    
    try:
        parking_final, top_selections = perform_final_selection()
        print("\n✅ Final selection completed successfully")
        exit(0)
    except Exception as e:
        print(f"\n❌ Final selection failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)