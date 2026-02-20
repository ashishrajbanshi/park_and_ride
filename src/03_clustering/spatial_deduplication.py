"""
Spatial Clustering & Deduplication
===================================

Purpose:
    Identify and resolve overlapping Park & Ride candidates
    
Methodology:
    1. Detect spatial clusters (DBSCAN + catchment overlap)
    2. Calculate directional access patterns
    3. Evaluate unique coverage contribution
    4. Select optimal subset from each cluster
    
Key Innovation:
    Preserves directional diversity while eliminating redundancy
    
Author: [Your Name]
Date: January 2026
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
from datetime import datetime
from sklearn.cluster import DBSCAN
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import warnings
import sys

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config_loader import Config


# ============================================================================
# SPATIAL CLUSTER DETECTION
# ============================================================================

def detect_spatial_clusters(parking_gdf: gpd.GeoDataFrame,
                           max_distance_m: int = 400) -> gpd.GeoDataFrame:
    """
    Detect clusters of nearby parking lots using DBSCAN
    
    Args:
        parking_gdf: Parking lots with geometry (must be in UTM)
        max_distance_m: Maximum distance to consider as same cluster
        
    Returns:
        GeoDataFrame with cluster_id column added
    """
    print(f"\n🔍 Detecting spatial clusters (max distance: {max_distance_m}m)...")
    
    # Extract coordinates
    coords = np.array([[geom.x, geom.y] for geom in parking_gdf.geometry])
    
    # DBSCAN clustering
    # eps = max distance between points in same cluster
    # min_samples = 2 (at least 2 sites to form a cluster)
    clustering = DBSCAN(eps=max_distance_m, min_samples=2, metric='euclidean')
    
    parking_gdf['cluster_id'] = clustering.fit_predict(coords)
    
    # cluster_id = -1 means "noise" (not in any cluster = isolated site)
    num_clusters = len(set(parking_gdf['cluster_id'])) - (1 if -1 in parking_gdf['cluster_id'].values else 0)
    num_isolated = (parking_gdf['cluster_id'] == -1).sum()
    num_clustered = len(parking_gdf) - num_isolated
    
    print(f"  ✓ Found {num_clusters} clusters")
    print(f"  ✓ Clustered sites: {num_clustered}")
    print(f"  ✓ Isolated sites: {num_isolated}")
    
    # Show cluster size distribution
    if num_clusters > 0:
        cluster_sizes = parking_gdf[parking_gdf['cluster_id'] >= 0]['cluster_id'].value_counts().sort_values(ascending=False)
        print(f"\n  📊 Cluster size distribution:")
        print(f"     Largest cluster: {cluster_sizes.max()} sites")
        print(f"     Median cluster size: {cluster_sizes.median():.0f} sites")
        print(f"     Clusters with 5+ sites: {(cluster_sizes >= 5).sum()}")
        print(f"     Clusters with 10+ sites: {(cluster_sizes >= 10).sum()}")
    
    return parking_gdf


# ============================================================================
# CATCHMENT OVERLAP CALCULATION
# ============================================================================

def calculate_catchment_overlap(parking_gdf: gpd.GeoDataFrame,
                                buffer_distance: int = 800) -> pd.DataFrame:
    """
    Calculate pairwise catchment overlap using Jaccard Index
    
    Args:
        parking_gdf: Parking lots with geometry
        buffer_distance: Walking distance for catchment (meters)
        
    Returns:
        DataFrame with pairwise overlap scores
    """
    print(f"\n📐 Calculating catchment overlap ({buffer_distance}m buffers)...")
    
    # Create catchment buffers
    parking_gdf['catchment'] = parking_gdf.geometry.buffer(buffer_distance)
    
    overlaps = []
    
    # Only calculate for sites in same cluster (optimization)
    for cluster_id in parking_gdf['cluster_id'].unique():
        if cluster_id == -1:  # Skip isolated sites
            continue
        
        cluster_sites = parking_gdf[parking_gdf['cluster_id'] == cluster_id]
        
        if len(cluster_sites) < 2:
            continue
        
        # Pairwise overlap calculation within cluster
        for i, row_i in cluster_sites.iterrows():
            for j, row_j in cluster_sites.iterrows():
                if i >= j:  # Skip self and duplicates
                    continue
                
                # Jaccard Index: Intersection / Union
                intersection = row_i['catchment'].intersection(row_j['catchment']).area
                union = row_i['catchment'].union(row_j['catchment']).area
                
                if union > 0:
                    overlap_ratio = intersection / union
                else:
                    overlap_ratio = 0
                
                overlaps.append({
                    'parking_id_1': row_i['parking_id'],
                    'parking_id_2': row_j['parking_id'],
                    'cluster_id': cluster_id,
                    'overlap_ratio': overlap_ratio,
                    'distance_m': row_i.geometry.distance(row_j.geometry)
                })
    
    if overlaps:
        overlap_df = pd.DataFrame(overlaps)
        
        print(f"  ✓ Calculated {len(overlap_df)} pairwise overlaps")
        print(f"\n  📊 Overlap statistics:")
        print(f"     Mean overlap: {overlap_df['overlap_ratio'].mean():.2f}")
        print(f"     Median overlap: {overlap_df['overlap_ratio'].median():.2f}")
        print(f"     High overlap (>0.7): {(overlap_df['overlap_ratio'] > 0.7).sum()} pairs")
        print(f"     Extreme overlap (>0.9): {(overlap_df['overlap_ratio'] > 0.9).sum()} pairs")
        
        return overlap_df
    else:
        print(f"  ℹ️  No overlapping pairs found")
        return pd.DataFrame()


# ============================================================================
# DIRECTIONAL ACCESS ANALYSIS
# ============================================================================

def calculate_directional_access(parking_gdf: gpd.GeoDataFrame,
                                 census_gdf: gpd.GeoDataFrame,
                                 buffer_distance: int = 5000) -> gpd.GeoDataFrame:
    """
    Analyze which directions (compass quadrants) each parking lot serves
    
    Methodology:
        1. Create 5km buffer around each parking lot (drive catchment)
        2. Divide buffer into 8 directional quadrants (N, NE, E, SE, S, SW, W, NW)
        3. Calculate population in each quadrant
        4. Identify unique population served only by this lot
        
    Args:
        parking_gdf: Parking lots with cluster_id
        census_gdf: Census blocks with population
        buffer_distance: Drive catchment distance (meters)
        
    Returns:
        GeoDataFrame with directional access metrics
    """
    print(f"\n🧭 Analyzing directional access patterns...")
    
    # Ensure same CRS
    if parking_gdf.crs != census_gdf.crs:
        census_gdf = census_gdf.to_crs(parking_gdf.crs)
    
    # Define 8 compass directions (in degrees, clockwise from North)
    directions = {
        'N': (337.5, 22.5),    # North
        'NE': (22.5, 67.5),    # Northeast
        'E': (67.5, 112.5),    # East
        'SE': (112.5, 157.5),  # Southeast
        'S': (157.5, 202.5),   # South
        'SW': (202.5, 247.5),  # Southwest
        'W': (247.5, 292.5),   # West
        'NW': (292.5, 337.5)   # Northwest
    }
    
    def create_directional_wedge(center_point: Point, 
                                 direction_name: str,
                                 radius: float) -> Polygon:
        """Create a wedge-shaped polygon for a compass direction"""
        angle_start, angle_end = directions[direction_name]
        
        # Handle wraparound for North (337.5-360 and 0-22.5)
        if direction_name == 'N':
            angles = list(range(int(angle_start), 360, 5)) + list(range(0, int(angle_end), 5))
        else:
            angles = range(int(angle_start), int(angle_end), 5)
        
        # Create wedge points
        points = [center_point]
        for angle in angles:
            rad = np.radians(angle)
            x = center_point.x + radius * np.sin(rad)
            y = center_point.y + radius * np.cos(rad)
            points.append(Point(x, y))
        points.append(center_point)
        
        return Polygon(points)
    
    # Calculate directional access for each parking lot
    directional_data = []
    
    for idx, parking_row in parking_gdf.iterrows():
        parking_id = parking_row['parking_id']
        center = parking_row.geometry
        
        # Population served in each direction
        dir_populations = {}
        
        for dir_name in directions.keys():
            # Create directional wedge
            wedge = create_directional_wedge(center, dir_name, buffer_distance)
            
            # Find census blocks in this wedge
            census_in_wedge = census_gdf[census_gdf.geometry.intersects(wedge)]
            
            # Sum population
            if 'total_pop' in census_in_wedge.columns:
                pop = census_in_wedge['total_pop'].sum()
            else:
                pop = 0
            
            dir_populations[f'pop_{dir_name}'] = pop
        
        # Total population served
        total_pop = sum(dir_populations.values())
        
        # Dominant direction (highest population)
        if total_pop > 0:
            dominant_dir = max(dir_populations, key=dir_populations.get)
            dominant_dir = dominant_dir.replace('pop_', '')
        else:
            dominant_dir = 'Unknown'
        
        directional_data.append({
            'parking_id': parking_id,
            **dir_populations,
            'total_pop_served': total_pop,
            'dominant_direction': dominant_dir
        })
    
    # Merge with parking data
    dir_df = pd.DataFrame(directional_data)
    parking_gdf = parking_gdf.merge(dir_df, on='parking_id', how='left')
    
    print(f"  ✓ Calculated directional access for {len(parking_gdf)} sites")
    
    # Show dominant direction distribution
    if 'dominant_direction' in parking_gdf.columns:
        dir_dist = parking_gdf['dominant_direction'].value_counts()
        print(f"\n  📊 Dominant direction distribution:")
        for direction, count in dir_dist.head(8).items():
            print(f"     {direction}: {count} sites")
    
    return parking_gdf


# ============================================================================
# UNIQUE COVERAGE CALCULATION
# ============================================================================

def calculate_unique_coverage(parking_gdf: gpd.GeoDataFrame,
                              overlap_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Calculate how much unique coverage each site provides within its cluster
    
    Methodology:
        For sites in same cluster with high overlap:
        - Site with more unique directional coverage scores higher
        - Uniqueness = population served ONLY by this site in each direction
        
    Args:
        parking_gdf: Parking lots with directional access data
        overlap_df: Pairwise overlap scores
        
    Returns:
        GeoDataFrame with uniqueness scores
    """
    print(f"\n🎯 Calculating unique coverage contribution...")
    
    # For each cluster, calculate uniqueness
    uniqueness_scores = []
    
    for cluster_id in parking_gdf['cluster_id'].unique():
        if cluster_id == -1:  # Isolated sites are 100% unique
            isolated_sites = parking_gdf[parking_gdf['cluster_id'] == -1]['parking_id']
            for pid in isolated_sites:
                uniqueness_scores.append({
                    'parking_id': pid,
                    'uniqueness_score': 100.0,  # Fully unique
                    'explanation': 'Isolated site (no nearby competitors)'
                })
            continue
        
        # Sites in this cluster
        cluster_sites = parking_gdf[parking_gdf['cluster_id'] == cluster_id]
        
        if len(cluster_sites) < 2:
            continue
        
        # For each site in cluster, calculate uniqueness
        for idx, site in cluster_sites.iterrows():
            parking_id = site['parking_id']
            
            # Find overlapping sites
            overlaps = overlap_df[
                ((overlap_df['parking_id_1'] == parking_id) | 
                 (overlap_df['parking_id_2'] == parking_id)) &
                (overlap_df['overlap_ratio'] > 0.5)  # Significant overlap
            ]
            
            if len(overlaps) == 0:
                # No significant overlaps within cluster
                uniqueness = 80.0
                explanation = 'Low overlap with cluster neighbors'
            else:
                # Calculate directional uniqueness
                # Sites serving different directions are more unique
                
                # Get dominant direction of this site
                site_direction = site.get('dominant_direction', 'Unknown')
                
                # Get directions of overlapping sites
                overlapping_ids = set()
                for _, overlap_row in overlaps.iterrows():
                    other_id = (overlap_row['parking_id_2'] 
                               if overlap_row['parking_id_1'] == parking_id 
                               else overlap_row['parking_id_1'])
                    overlapping_ids.add(other_id)
                
                # Check if any overlapping site has same dominant direction
                same_direction = False
                for other_id in overlapping_ids:
                    other_site = parking_gdf[parking_gdf['parking_id'] == other_id]
                    if len(other_site) > 0:
                        other_direction = other_site.iloc[0].get('dominant_direction', 'Unknown')
                        if other_direction == site_direction:
                            same_direction = True
                            break
                
                if same_direction:
                    # Competing for same direction → low uniqueness
                    avg_overlap = overlaps['overlap_ratio'].mean()
                    uniqueness = (1 - avg_overlap) * 100  # Invert overlap
                    explanation = f'High overlap ({avg_overlap:.1%}) with same-direction sites'
                else:
                    # Different directions → high uniqueness
                    uniqueness = 70.0
                    explanation = 'Serves different direction than overlapping sites'
            
            uniqueness_scores.append({
                'parking_id': parking_id,
                'uniqueness_score': uniqueness,
                'explanation': explanation
            })
    
    # Merge with parking data
    uniqueness_df = pd.DataFrame(uniqueness_scores)
    parking_gdf = parking_gdf.merge(uniqueness_df, on='parking_id', how='left')
    
    # Fill NaN for sites not in clusters
    parking_gdf['uniqueness_score'] = parking_gdf['uniqueness_score'].fillna(100.0)
    parking_gdf['explanation'] = parking_gdf['explanation'].fillna('No clustering analysis needed')
    
    print(f"  ✓ Calculated uniqueness for {len(uniqueness_scores)} sites")
    print(f"\n  📊 Uniqueness distribution:")
    print(f"     Mean: {parking_gdf['uniqueness_score'].mean():.1f}")
    print(f"     High uniqueness (>70): {(parking_gdf['uniqueness_score'] > 70).sum()} sites")
    print(f"     Low uniqueness (<30): {(parking_gdf['uniqueness_score'] < 30).sum()} sites")
    
    return parking_gdf


# ============================================================================
# CLUSTER-BASED SELECTION
# ============================================================================

def select_optimal_sites(parking_gdf: gpd.GeoDataFrame,
                        accessibility_scores: pd.DataFrame,
                        max_sites_per_cluster: int = 5,
                        directional_weight: float = 0.3) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """
    Select optimal subset of parking lots from each cluster
    
    Selection Criteria:
        1. Accessibility score (primary)
        2. Directional uniqueness (secondary)
        3. Capacity (tie-breaker)
        
    Args:
        parking_gdf: Parking lots with cluster and uniqueness data
        accessibility_scores: Scores from Step 2
        max_sites_per_cluster: Maximum sites to keep per cluster
        directional_weight: Weight for directional uniqueness (0-1)
        
    Returns:
        Tuple of (selected sites GeoDataFrame, deduplication report DataFrame)
    """
    print(f"\n🎯 Selecting optimal sites from clusters...")
    print(f"   Max sites per cluster: {max_sites_per_cluster}")
    print(f"   Directional uniqueness weight: {directional_weight}")
    
    # Merge accessibility scores
    parking_with_scores = parking_gdf.merge(
        accessibility_scores[['parking_id', 'accessibility_score', 'has_adequate_transit']],
        on='parking_id',
        how='left'
    )
    
    # Calculate adjusted score (accessibility + directional bonus)
    parking_with_scores['adjusted_score'] = (
        parking_with_scores['accessibility_score'] * 
        (1 + directional_weight * (parking_with_scores['uniqueness_score'] / 100))
    )
    
    selected_sites = []
    dedup_report = []
    
    # Process isolated sites (always keep)
    isolated = parking_with_scores[parking_with_scores['cluster_id'] == -1]
    selected_sites.append(isolated)
    
    print(f"\n  ✓ Keeping {len(isolated)} isolated sites (no clustering)")
    
    # Process each cluster
    clusters = parking_with_scores[parking_with_scores['cluster_id'] >= 0]['cluster_id'].unique()
    
    for cluster_id in sorted(clusters):
        cluster_sites = parking_with_scores[parking_with_scores['cluster_id'] == cluster_id].copy()
        
        # Sort by adjusted score
        cluster_sites = cluster_sites.sort_values('adjusted_score', ascending=False)
        
        # Select top N sites
        n_to_select = min(max_sites_per_cluster, len(cluster_sites))
        selected = cluster_sites.head(n_to_select)
        excluded = cluster_sites.iloc[n_to_select:]
        
        selected_sites.append(selected)
        
        # Document deduplication
        dedup_report.append({
            'cluster_id': cluster_id,
            'cluster_size': len(cluster_sites),
            'sites_selected': n_to_select,
            'sites_excluded': len(excluded),
            'mean_overlap': cluster_sites.get('uniqueness_score', pd.Series([100])).mean(),
            'selected_ids': list(selected['parking_id'].values),
            'excluded_ids': list(excluded['parking_id'].values) if len(excluded) > 0 else []
        })
        
        print(f"\n  Cluster {cluster_id}: {len(cluster_sites)} sites")
        print(f"    → Selected: {n_to_select}")
        print(f"    → Excluded: {len(excluded)}")
        if len(selected) > 0:
            print(f"    → Top site: {selected.iloc[0]['name']} (score: {selected.iloc[0]['adjusted_score']:.1f})")
    
    # Combine selected sites
    selected_gdf = pd.concat(selected_sites, ignore_index=True)
    
    # Create deduplication report
    dedup_df = pd.DataFrame(dedup_report)
    
    # Summary
    total_original = len(parking_with_scores)
    total_selected = len(selected_gdf)
    total_excluded = total_original - total_selected
    
    print(f"\n{'='*80}")
    print(f"DEDUPLICATION SUMMARY")
    print(f"{'='*80}")
    print(f"Original sites: {total_original}")
    print(f"Selected sites: {total_selected}")
    print(f"Excluded sites: {total_excluded} ({total_excluded/total_original*100:.1f}%)")
    print(f"Reduction: {total_excluded/total_original*100:.1f}%")
    
    return selected_gdf, dedup_df


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def perform_spatial_deduplication(config_path: str = None):
    """
    Complete spatial clustering and deduplication workflow
    
    Steps:
        1. Load data (parking + census + accessibility)
        2. Detect spatial clusters
        3. Calculate catchment overlap
        4. Analyze directional access
        5. Calculate unique coverage
        6. Select optimal sites
        7. Export results
    """
    cfg = Config(config_path)
    
    print("\n" + "="*80)
    print("SPATIAL CLUSTERING & DEDUPLICATION")
    print("="*80)
    print(f"Project: {cfg.project_name}")
    print("="*80)
    
    start_time = datetime.now()
    
    # ========================================================================
    # STEP 1: Load data
    # ========================================================================
    print("\n📂 Loading processed datasets...")
    
    parking_gdf = gpd.read_file(cfg.paths.data_processed / "parking_lots_clean.gpkg")
    census_gdf = gpd.read_file(cfg.paths.data_processed / "census_blocks_clean.gpkg")
    
    # Load accessibility scores (moderate scenario as primary)
    accessibility_df = pd.read_csv(
        cfg.paths.data_outputs / "accessibility" / "accessibility_scores_moderate.csv"
    )
    
    print(f"  ✓ Parking lots: {len(parking_gdf)}")
    print(f"  ✓ Census blocks: {len(census_gdf)}")
    print(f"  ✓ Accessibility scores: {len(accessibility_df)}")
    
    # ========================================================================
    # STEP 2: Detect spatial clusters
    # ========================================================================
    parking_gdf = detect_spatial_clusters(parking_gdf, max_distance_m=400)
    
    # ========================================================================
    # STEP 3: Calculate catchment overlap
    # ========================================================================
    overlap_df = calculate_catchment_overlap(parking_gdf, buffer_distance=800)
    
    # ========================================================================
    # STEP 4: Analyze directional access
    # ========================================================================
    parking_gdf = calculate_directional_access(
        parking_gdf, 
        census_gdf,
        buffer_distance=5000  # 5km drive catchment
    )
    
    # ========================================================================
    # STEP 5: Calculate unique coverage
    # ========================================================================
    parking_gdf = calculate_unique_coverage(parking_gdf, overlap_df)
    
    # ========================================================================
    # STEP 6: Select optimal sites
    # ========================================================================
    selected_gdf, dedup_report = select_optimal_sites(
        parking_gdf,
        accessibility_df,
        max_sites_per_cluster=5,
        directional_weight=0.3
    )
    
    # ========================================================================
    # STEP 7: Export results
    # ========================================================================
    print("\n💾 Saving deduplication results...")
    
    results_dir = cfg.paths.data_outputs / "clustering"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save selected sites
    output_path = cfg.paths.data_processed / "parking_lots_deduplicated.gpkg"
    
    # Drop extra geometry columns to keep only the main 'geometry' column for file export
    selected_export = selected_gdf.copy()
    # Drop the catchment column which is also a geometry column
    selected_export = selected_export.drop(columns=['catchment'], errors='ignore')
    
    selected_export.to_file(output_path, driver='GPKG')
    print(f"  ✓ Selected sites: {output_path}")
    
    # Save full clustering analysis
    clustering_csv = results_dir / "clustering_analysis.csv"
    parking_gdf.drop(columns=['geometry', 'catchment'], errors='ignore').to_csv(
        clustering_csv, index=False
    )
    print(f"  ✓ Clustering analysis: {clustering_csv}")
    
    # Save overlap matrix
    if len(overlap_df) > 0:
        overlap_csv = results_dir / "catchment_overlap.csv"
        overlap_df.to_csv(overlap_csv, index=False)
        print(f"  ✓ Overlap matrix: {overlap_csv}")
    
    # Save deduplication report
    dedup_csv = results_dir / "deduplication_report.csv"
    dedup_report.to_csv(dedup_csv, index=False)
    print(f"  ✓ Deduplication report: {dedup_csv}")
    
    # ========================================================================
    # STEP 8: Generate summary
    # ========================================================================
    duration = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "="*80)
    print("✅ SPATIAL DEDUPLICATION COMPLETE")
    print("="*80)
    print(f"Processing time: {duration:.1f} seconds")
    
    print(f"\n📊 Final Statistics:")
    print(f"  Original parking lots: {len(parking_gdf)}")
    print(f"  Spatial clusters detected: {(parking_gdf['cluster_id'] >= 0).any()}")
    print(f"  Sites selected: {len(selected_gdf)}")
    print(f"  Sites excluded: {len(parking_gdf) - len(selected_gdf)}")
    print(f"  Reduction rate: {(1 - len(selected_gdf)/len(parking_gdf))*100:.1f}%")
    
    if len(overlap_df) > 0:
        high_overlap = (overlap_df['overlap_ratio'] > 0.7).sum()
        print(f"\n  High-overlap pairs eliminated: {high_overlap}")
    
    print("\n" + "="*80)
    print("NEXT STEP: Equity Analysis & Coverage Optimization")
    print("="*80)
    print("Run: python src/04_equity/coverage_analysis.py")
    
    return selected_gdf, dedup_report, overlap_df


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    """Run spatial clustering and deduplication"""
    
    try:
        selected_gdf, dedup_report, overlap_df = perform_spatial_deduplication()
        print("\n✅ Deduplication completed successfully")
        exit(0)
    except Exception as e:
        print(f"\n❌ Deduplication failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)