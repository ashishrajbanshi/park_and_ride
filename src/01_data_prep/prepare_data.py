import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
import warnings
import sys

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config_loader import Config


# ============================================================================
# FIELD MAPPINGS
# ============================================================================

# Census Bureau field codes → Standardized names
CENSUS_FIELD_MAPPING = {
    # Core demographic fields
    'B01003_001E': 'total_pop',
    'Employment': 'employment',
    'B19301_001E': 'per_capita_income',
    'B19001_001E': 'median_hh_income',
    'B08134_001E': 'transport_mode',
    'B08303_001E': 'travel_time',
    'B25044_001E': 'vehicles_available',
    'B25001_001E': 'housing_units',
    'workFromHome': 'work_from_home',
    
    # Identifiers
    'GEOID': 'geoid',
    'NAME': 'name',
    'ALAND': 'land_area',
    
    # Geometry (preserve as-is)
    'geometry': 'geometry'
}

# Parking field variations → Standardized names
PARKING_FIELD_MAPPING = {
    'capacity': 'capacity',
    'Capacity': 'capacity',
    'spaces': 'capacity',
    'parking_spaces': 'capacity',
    'total_spaces': 'capacity',
    
    'type': 'type',
    'Type': 'type',
    'parking_type': 'type',
    
    'name': 'name',
    'Name': 'name',
    
    'id': 'parking_id',
    'ID': 'parking_id',
    'osm_id': 'parking_id',
    
    'geometry': 'geometry'
}


# ============================================================================
# PROCESSING METADATA TRACKER
# ============================================================================

class ProcessingMetadata:
    """Track all transformations applied to data"""
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.start_time = datetime.now()
        self.transformations = []
        self.statistics = {
            'raw_record_count': 0,
            'final_record_count': 0,
            'fields_renamed': [],
            'fields_created': [],
            'records_removed': 0,
            'values_imputed': {}
        }
    
    def log_transformation(self, operation: str, details: str, count: int = 0):
        """Log a transformation operation"""
        self.transformations.append({
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'details': details,
            'affected_count': count
        })
        print(f"  ✓ {operation}: {details}")
    
    def log_imputation(self, field: str, method: str, count: int):
        """Log missing value imputation"""
        self.statistics['values_imputed'][field] = {
            'method': method,
            'count': count
        }
        self.log_transformation(
            'imputation',
            f"{field}: {count} values imputed using {method}",
            count
        )
    
    def to_dict(self) -> dict:
        """Export metadata as dictionary"""
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            return obj
        
        data = {
            'dataset': self.dataset_name,
            'processing_date': datetime.now().isoformat(),
            'duration_seconds': (datetime.now() - self.start_time).total_seconds(),
            'statistics': self.statistics,
            'transformations': self.transformations
        }
        return convert_numpy_types(data)
    
    def save_json(self, output_path: Path):
        """Save metadata to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"💾 Metadata saved: {output_path}")


# ============================================================================
# CENSUS BLOCK PREPARATION
# ============================================================================

def prepare_census_blocks(cfg: Config) -> gpd.GeoDataFrame:
    """
    Prepare census block group data for analysis
    
    Steps:
        1. Load raw data
        2. Standardize field names
        3. Fix invalid geometries
        4. Reproject to UTM
        5. Calculate derived fields (densities, etc.)
        6. Handle missing values
        7. Save processed version
    
    Returns:
        GeoDataFrame with clean, standardized census data
    """
    print("\n" + "="*80)
    print("PREPARING CENSUS BLOCKS")
    print("="*80)
    
    metadata = ProcessingMetadata("census_blocks")
    
    # ========================================================================
    # STEP 1: Load raw data
    # ========================================================================
    print("\n📂 Loading raw census data...")
    
    census_path = cfg.paths.data_raw / "census_blocks.geojson"
    gdf = gpd.read_file(census_path)
    
    metadata.statistics['raw_record_count'] = len(gdf)
    print(f"  ✓ Loaded {len(gdf)} census block groups")
    print(f"  ✓ Original CRS: {gdf.crs}")
    print(f"  ✓ Original fields: {len(gdf.columns)} columns")
    
    # ========================================================================
    # STEP 2: Standardize field names
    # ========================================================================
    print("\n🏷️  Standardizing field names...")
    
    # Apply mapping
    rename_map = {}
    for old_name in gdf.columns:
        if old_name in CENSUS_FIELD_MAPPING:
            new_name = CENSUS_FIELD_MAPPING[old_name]
            if new_name != old_name:
                rename_map[old_name] = new_name
    
    if rename_map:
        gdf = gdf.rename(columns=rename_map)
        metadata.statistics['fields_renamed'] = list(rename_map.values())
        metadata.log_transformation(
            'field_renaming',
            f"Renamed {len(rename_map)} fields to standardized names",
            len(rename_map)
        )
        
        # Show sample of renames
        for old, new in list(rename_map.items())[:5]:
            print(f"     {old} → {new}")
        if len(rename_map) > 5:
            print(f"     ... and {len(rename_map)-5} more")
    
    # ========================================================================
    # STEP 3: Fix invalid geometries
    # ========================================================================
    print("\n🔧 Fixing invalid geometries...")
    
    invalid_before = (~gdf.geometry.is_valid).sum()
    if invalid_before > 0:
        print(f"  ⚠️  Found {invalid_before} invalid geometries")
        
        # Apply buffer(0) fix
        gdf['geometry'] = gdf.geometry.buffer(0)
        
        invalid_after = (~gdf.geometry.is_valid).sum()
        fixed = invalid_before - invalid_after
        
        metadata.log_transformation(
            'geometry_fix',
            f"Fixed {fixed} invalid geometries using buffer(0)",
            fixed
        )
        
        if invalid_after > 0:
            print(f"  ⚠️  {invalid_after} geometries still invalid - removing")
            gdf = gdf[gdf.geometry.is_valid]
            metadata.statistics['records_removed'] += invalid_after
    else:
        print(f"  ✓ All geometries valid")
    
    # Remove null geometries
    null_geom = gdf.geometry.isna().sum()
    if null_geom > 0:
        print(f"  ⚠️  Removing {null_geom} records with null geometry")
        gdf = gdf[gdf.geometry.notna()]
        metadata.statistics['records_removed'] += null_geom
    
    # ========================================================================
    # STEP 4: Reproject to UTM
    # ========================================================================
    print("\n🗺️  Reprojecting to UTM Zone 16N...")
    
    original_crs = gdf.crs.to_epsg() if gdf.crs else None
    target_crs = cfg.crs.analysis_utm
    
    if original_crs != target_crs:
        gdf = gdf.to_crs(epsg=target_crs)
        metadata.log_transformation(
            'reprojection',
            f"Reprojected from EPSG:{original_crs} to EPSG:{target_crs}"
        )
        print(f"  ✓ Reprojected: EPSG:{original_crs} → EPSG:{target_crs}")
    else:
        print(f"  ✓ Already in target CRS (EPSG:{target_crs})")
    
    # ========================================================================
    # STEP 5: Calculate derived fields
    # ========================================================================
    print("\n📐 Calculating derived fields...")
    
    # Calculate area in km²
    gdf['area_km2'] = gdf.geometry.area / 1_000_000
    metadata.statistics['fields_created'].append('area_km2')
    print(f"  ✓ Calculated area_km2 from geometry")
    
    # Population density (per km²)
    if 'total_pop' in gdf.columns:
        gdf['pop_density'] = gdf['total_pop'] / gdf['area_km2']
        gdf['pop_density'] = gdf['pop_density'].replace([np.inf, -np.inf], np.nan)
        metadata.statistics['fields_created'].append('pop_density')
        print(f"  ✓ Calculated pop_density (pop/km²)")
    
    # Employment density (per km²)
    if 'employment' in gdf.columns:
        gdf['employment_density'] = gdf['employment'] / gdf['area_km2']
        gdf['employment_density'] = gdf['employment_density'].replace([np.inf, -np.inf], np.nan)
        metadata.statistics['fields_created'].append('employment_density')
        print(f"  ✓ Calculated employment_density (jobs/km²)")
    
    # Add centroid coordinates (for spatial analysis)
    gdf['centroid_x'] = gdf.geometry.centroid.x
    gdf['centroid_y'] = gdf.geometry.centroid.y
    metadata.statistics['fields_created'].extend(['centroid_x', 'centroid_y'])
    print(f"  ✓ Calculated centroid coordinates")
    
    # ========================================================================
    # STEP 6: Handle missing values
    # ========================================================================
    print("\n🔢 Handling missing values...")
    
    # Population - keep as-is (0 is valid for unpopulated areas)
    if 'total_pop' in gdf.columns:
        null_pop = gdf['total_pop'].isna().sum()
        if null_pop > 0:
            # Impute with 0 (assume unpopulated)
            gdf['total_pop'] = gdf['total_pop'].fillna(0)
            metadata.log_imputation('total_pop', 'zero', null_pop)
    
    # Employment - impute missing as 0
    if 'employment' in gdf.columns:
        null_emp = gdf['employment'].isna().sum()
        if null_emp > 0:
            gdf['employment'] = gdf['employment'].fillna(0)
            metadata.log_imputation('employment', 'zero', null_emp)
    
    # Income - leave as NULL (will be excluded from income-based analysis)
    if 'per_capita_income' in gdf.columns:
        null_income = gdf['per_capita_income'].isna().sum()
        if null_income > 0:
            print(f"  ℹ️  Keeping {null_income} NULL income values (will filter in analysis)")
    
    # ========================================================================
    # STEP 7: Add processing metadata to dataframe
    # ========================================================================
    gdf['processing_date'] = datetime.now().isoformat()
    gdf['data_source'] = 'ACS 2019-2023'
    
    # ========================================================================
    # STEP 8: Save processed data
    # ========================================================================
    print("\n💾 Saving processed census data...")
    
    metadata.statistics['final_record_count'] = len(gdf)
    
    # Save as GeoPackage (better than GeoJSON for analysis)
    output_path = cfg.paths.data_processed / "census_blocks_clean.gpkg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    gdf.to_file(output_path, driver='GPKG')
    print(f"  ✓ Saved: {output_path}")
    print(f"  ✓ Format: GeoPackage (GPKG)")
    print(f"  ✓ Records: {len(gdf)}")
    print(f"  ✓ Fields: {len(gdf.columns)}")
    
    # Also save as CSV (for easy inspection)
    csv_path = cfg.paths.data_processed / "census_blocks_attributes.csv"
    gdf_csv = gdf.drop(columns=['geometry'])
    gdf_csv.to_csv(csv_path, index=False)
    print(f"  ✓ Attributes saved: {csv_path}")
    
    # Save metadata
    metadata_path = cfg.paths.data_processed / "census_blocks_metadata.json"
    metadata.save_json(metadata_path)
    
    # Summary statistics
    print("\n📊 Processing Summary:")
    print(f"  • Raw records: {metadata.statistics['raw_record_count']}")
    print(f"  • Final records: {metadata.statistics['final_record_count']}")
    print(f"  • Records removed: {metadata.statistics['records_removed']}")
    print(f"  • Fields renamed: {len(metadata.statistics['fields_renamed'])}")
    print(f"  • Fields created: {len(metadata.statistics['fields_created'])}")
    
    return gdf


# ============================================================================
# PARKING LOT PREPARATION
# ============================================================================

def prepare_parking_lots(cfg: Config) -> gpd.GeoDataFrame:
    """
    Prepare parking lot data for analysis
    
    Steps:
        1. Load raw data
        2. Standardize field names
        3. Convert polygons to centroids
        4. Impute missing capacity
        5. Reproject to UTM
        6. Add unique identifiers
        7. Save processed version
    
    Returns:
        GeoDataFrame with clean, standardized parking data
    """
    print("\n" + "="*80)
    print("PREPARING PARKING LOTS")
    print("="*80)
    
    metadata = ProcessingMetadata("parking_lots")
    
    # ========================================================================
    # STEP 1: Load raw data
    # ========================================================================
    print("\n📂 Loading raw parking data...")
    
    parking_path = cfg.paths.data_raw / "parking_lots.geojson"
    gdf = gpd.read_file(parking_path)
    
    metadata.statistics['raw_record_count'] = len(gdf)
    print(f"  ✓ Loaded {len(gdf)} parking lots")
    print(f"  ✓ Original CRS: {gdf.crs}")
    
    # ========================================================================
    # STEP 2: Standardize field names
    # ========================================================================
    print("\n🏷️  Standardizing field names...")
    
    # Apply mapping
    rename_map = {}
    for old_name in gdf.columns:
        if old_name in PARKING_FIELD_MAPPING:
            new_name = PARKING_FIELD_MAPPING[old_name]
            if new_name != old_name:
                rename_map[old_name] = new_name
    
    if rename_map:
        gdf = gdf.rename(columns=rename_map)
        metadata.statistics['fields_renamed'] = list(rename_map.values())
        metadata.log_transformation(
            'field_renaming',
            f"Renamed {len(rename_map)} fields",
            len(rename_map)
        )
    
    # ========================================================================
    # STEP 3: Convert polygons to centroids
    # ========================================================================
    print("\n📍 Converting geometries to points...")
    
    non_point = gdf[gdf.geometry.type != 'Point']
    if len(non_point) > 0:
        print(f"  ⚠️  Converting {len(non_point)} polygon/multipolygon to centroids")
        
        # Store original area before converting
        gdf['original_area_m2'] = 0.0
        gdf.loc[non_point.index, 'original_area_m2'] = non_point.geometry.area
        
        # Convert to centroids
        gdf.loc[non_point.index, 'geometry'] = non_point.geometry.centroid
        
        metadata.log_transformation(
            'geometry_conversion',
            f"Converted {len(non_point)} polygons to centroids",
            len(non_point)
        )
        metadata.statistics['fields_created'].append('original_area_m2')
    
    # ========================================================================
    # STEP 4: Reproject to UTM
    # ========================================================================
    print("\n🗺️  Reprojecting to UTM Zone 16N...")
    
    original_crs = gdf.crs.to_epsg() if gdf.crs else None
    target_crs = cfg.crs.analysis_utm
    
    if original_crs is None:
        print(f"  ⚠️  No CRS defined - assuming EPSG:4326 (WGS84)")
        gdf = gdf.set_crs(epsg=4326)
        original_crs = 4326
    
    if original_crs != target_crs:
        gdf = gdf.to_crs(epsg=target_crs)
        metadata.log_transformation(
            'reprojection',
            f"Reprojected from EPSG:{original_crs} to EPSG:{target_crs}"
        )
    
    # ========================================================================
    # STEP 5: Add unique parking_id if missing
    # ========================================================================
    print("\n🔢 Adding unique identifiers...")
    
    if 'parking_id' not in gdf.columns:
        gdf['parking_id'] = range(1, len(gdf) + 1)
        metadata.log_transformation(
            'id_creation',
            f"Created parking_id field (1-{len(gdf)})",
            len(gdf)
        )
        metadata.statistics['fields_created'].append('parking_id')
    else:
        # Ensure IDs are unique
        duplicates = gdf['parking_id'].duplicated().sum()
        if duplicates > 0:
            print(f"  ⚠️  Found {duplicates} duplicate IDs - reassigning")
            gdf['parking_id'] = range(1, len(gdf) + 1)
            metadata.log_transformation(
                'id_reassignment',
                f"Reassigned duplicate parking_ids",
                duplicates
            )
    
    # ========================================================================
    # STEP 6: Impute missing capacity
    # ========================================================================
    print("\n🅿️  Handling capacity values...")
    
    if 'capacity' not in gdf.columns:
        print(f"  ⚠️  No capacity field found")
        
        # Check if we have area to estimate
        if 'original_area_m2' in gdf.columns and (gdf['original_area_m2'] > 0).any():
            # Estimate: 1 space per 25 m² (includes aisles, circulation)
            gdf['capacity'] = (gdf['original_area_m2'] / 25).fillna(0).astype(int)
            gdf['capacity_estimated'] = True
            
            estimated = (gdf['original_area_m2'] > 0).sum()
            metadata.log_imputation(
                'capacity',
                'area-based estimation (25 m²/space)',
                estimated
            )
            metadata.statistics['fields_created'].extend(['capacity', 'capacity_estimated'])
        else:
            # Use default median
            default_capacity = 50  # Conservative estimate
            gdf['capacity'] = default_capacity
            gdf['capacity_estimated'] = True
            
            metadata.log_imputation(
                'capacity',
                f'default value ({default_capacity} spaces)',
                len(gdf)
            )
            metadata.statistics['fields_created'].extend(['capacity', 'capacity_estimated'])
    else:
        # Capacity field exists - handle missing values
        null_capacity = gdf['capacity'].isna() | (gdf['capacity'] == 0)
        
        if null_capacity.sum() > 0:
            print(f"  ⚠️  Imputing {null_capacity.sum()} missing/zero capacity values")
            
            # Create imputation flag
            gdf['capacity_estimated'] = False
            gdf.loc[null_capacity, 'capacity_estimated'] = True
            
            # Strategy 1: Use area if available
            has_area = null_capacity & (gdf.get('original_area_m2', 0) > 0)
            if has_area.sum() > 0:
                gdf.loc[has_area, 'capacity'] = (gdf.loc[has_area, 'original_area_m2'] / 25).astype(int)
                print(f"     {has_area.sum()} estimated from area")
            
            # Strategy 2: Use type-based median
            if 'type' in gdf.columns:
                remaining_nulls = gdf['capacity'].isna() | (gdf['capacity'] == 0)
                for parking_type in gdf['type'].unique():
                    if pd.isna(parking_type):
                        continue
                    
                    type_mask = (gdf['type'] == parking_type) & remaining_nulls
                    if type_mask.sum() > 0:
                        type_median = gdf[gdf['type'] == parking_type]['capacity'].median()
                        if pd.notna(type_median) and type_median > 0:
                            gdf.loc[type_mask, 'capacity'] = type_median
                            print(f"     {type_mask.sum()} imputed for type '{parking_type}' (median: {type_median:.0f})")
            
            # Strategy 3: Global median for remaining
            still_null = gdf['capacity'].isna() | (gdf['capacity'] == 0)
            if still_null.sum() > 0:
                global_median = gdf[gdf['capacity'] > 0]['capacity'].median()
                if pd.notna(global_median):
                    gdf.loc[still_null, 'capacity'] = global_median
                    print(f"     {still_null.sum()} imputed with global median ({global_median:.0f})")
                else:
                    # Last resort: default value
                    gdf.loc[still_null, 'capacity'] = 50
                    print(f"     {still_null.sum()} imputed with default (50)")
            
            metadata.log_imputation(
                'capacity',
                'multi-strategy (area/type/median)',
                null_capacity.sum()
            )
            
            if 'capacity_estimated' not in metadata.statistics['fields_created']:
                metadata.statistics['fields_created'].append('capacity_estimated')
        else:
            # No missing values
            gdf['capacity_estimated'] = False
            metadata.statistics['fields_created'].append('capacity_estimated')
            print(f"  ✓ All capacity values present")
    
    # Ensure capacity is positive integer
    gdf['capacity'] = gdf['capacity'].clip(lower=1).astype(int)
    
    # ========================================================================
    # STEP 7: Handle parking type
    # ========================================================================
    print("\n🏷️  Standardizing parking types...")
    
    if 'type' not in gdf.columns:
        gdf['type'] = 'unknown'
        metadata.statistics['fields_created'].append('type')
        print(f"  ✓ Created type field (all 'unknown')")
    else:
        null_type = gdf['type'].isna().sum()
        if null_type > 0:
            gdf['type'] = gdf['type'].fillna('unknown')
            print(f"  ✓ Set {null_type} missing types to 'unknown'")
        
        # Standardize type names (lowercase, consistent)
        gdf['type'] = gdf['type'].str.lower().str.strip()
        
        # Show distribution
        type_counts = gdf['type'].value_counts()
        print(f"  ℹ️  Type distribution:")
        for ptype, count in type_counts.head(5).items():
            pct = (count / len(gdf)) * 100
            print(f"     {ptype}: {count} ({pct:.1f}%)")
    
    # ========================================================================
    # STEP 8: Add location name (lat/lon for reference)
    # ========================================================================
    if 'name' not in gdf.columns:
        # Create descriptive name from coordinates
        # Convert centroid back to WGS84 for readable coords
        gdf_wgs84 = gdf.to_crs(epsg=4326)
        gdf['name'] = gdf_wgs84.apply(
            lambda row: f"Location ({row.geometry.y:.4f}, {row.geometry.x:.4f})",
            axis=1
        )
        metadata.statistics['fields_created'].append('name')
        print(f"  ✓ Generated location names from coordinates")
    
    # ========================================================================
    # STEP 9: Add processing metadata
    # ========================================================================
    gdf['processing_date'] = datetime.now().isoformat()
    gdf['data_source'] = 'OpenStreetMap'
    
    # ========================================================================
    # STEP 10: Save processed data
    # ========================================================================
    print("\n💾 Saving processed parking data...")
    
    metadata.statistics['final_record_count'] = len(gdf)
    
    # Save as GeoPackage
    output_path = cfg.paths.data_processed / "parking_lots_clean.gpkg"
    gdf.to_file(output_path, driver='GPKG')
    print(f"  ✓ Saved: {output_path}")
    
    # Save as CSV (attributes only)
    csv_path = cfg.paths.data_processed / "parking_lots_attributes.csv"
    gdf_csv = gdf.drop(columns=['geometry'])
    gdf_csv.to_csv(csv_path, index=False)
    print(f"  ✓ Attributes saved: {csv_path}")
    
    # Save metadata
    metadata_path = cfg.paths.data_processed / "parking_lots_metadata.json"
    metadata.save_json(metadata_path)
    
    # Summary
    print("\n📊 Processing Summary:")
    print(f"  • Raw records: {metadata.statistics['raw_record_count']}")
    print(f"  • Final records: {metadata.statistics['final_record_count']}")
    print(f"  • Capacity imputed: {null_capacity.sum() if 'capacity' in gdf.columns else 0}")
    print(f"  • Average capacity: {gdf['capacity'].mean():.0f} spaces")
    print(f"  • Total capacity: {gdf['capacity'].sum():,} spaces")
    
    return gdf


# ============================================================================
# GTFS PREPARATION
# ============================================================================

def prepare_gtfs_stops(cfg: Config) -> gpd.GeoDataFrame:
    """
    Prepare GTFS transit stops for analysis
    
    Steps:
        1. Load stops.txt
        2. Create point geometries from lat/lon
        3. Filter invalid coordinates
        4. Reproject to UTM
        5. Calculate stop frequency (from stop_times.txt)
        6. Save processed version
    
    Returns:
        GeoDataFrame with transit stops + frequency data
    """
    print("\n" + "="*80)
    print("PREPARING GTFS TRANSIT STOPS")
    print("="*80)
    
    metadata = ProcessingMetadata("gtfs_stops")
    
    # ========================================================================
    # STEP 1: Load stops
    # ========================================================================
    print("\n📂 Loading GTFS stops...")
    
    stops_path = cfg.paths.data_raw / "gtfs" / "stops.txt"
    df = pd.read_csv(stops_path)
    
    metadata.statistics['raw_record_count'] = len(df)
    print(f"  ✓ Loaded {len(df)} stops")
    
    # ========================================================================
    # STEP 2: Filter invalid coordinates
    # ========================================================================
    print("\n🧹 Filtering invalid coordinates...")
    
    # Remove null coordinates
    valid = df['stop_lat'].notna() & df['stop_lon'].notna()
    removed_null = (~valid).sum()
    if removed_null > 0:
        df = df[valid]
        metadata.statistics['records_removed'] += removed_null
        print(f"  ⚠️  Removed {removed_null} stops with null coordinates")
    
    # Remove zero coordinates
    valid = (df['stop_lat'] != 0.0) | (df['stop_lon'] != 0.0)
    removed_zero = (~valid).sum()
    if removed_zero > 0:
        df = df[valid]
        metadata.statistics['records_removed'] += removed_zero
        print(f"  ⚠️  Removed {removed_zero} stops with (0, 0) coordinates")
    
    # Remove out-of-bounds coordinates (not in Chattanooga region)
    in_bounds = (
        (df['stop_lat'] >= 34.0) & (df['stop_lat'] <= 36.5) &
        (df['stop_lon'] >= -86.5) & (df['stop_lon'] <= -84.0)
    )
    removed_bounds = (~in_bounds).sum()
    if removed_bounds > 0:
        df = df[in_bounds]
        metadata.statistics['records_removed'] += removed_bounds
        print(f"  ⚠️  Removed {removed_bounds} stops outside Chattanooga bounds")
    
    metadata.log_transformation(
        'coordinate_filtering',
        f"Removed {removed_null + removed_zero + removed_bounds} invalid stops"
    )
    
    # ========================================================================
    # STEP 3: Create geometries
    # ========================================================================
    print("\n📍 Creating point geometries...")
    
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['stop_lon'], df['stop_lat']),
        crs='EPSG:4326'
    )
    
    print(f"  ✓ Created {len(gdf)} point geometries")
    
    # ========================================================================
    # STEP 4: Reproject to UTM
    # ========================================================================
    print("\n🗺️  Reprojecting to UTM Zone 16N...")
    
    gdf = gdf.to_crs(epsg=cfg.crs.analysis_utm)
    metadata.log_transformation(
        'reprojection',
        f"Reprojected from EPSG:4326 to EPSG:{cfg.crs.analysis_utm}"
    )
    
    # ========================================================================
    # STEP 5: Calculate stop frequency
    # ========================================================================
    print("\n🚌 Calculating transit service frequency...")
    
    try:
        # Load stop_times to count trips per stop
        stop_times_path = cfg.paths.data_raw / "gtfs" / "stop_times.txt"
        stop_times = pd.read_csv(stop_times_path)
        
        # Count unique trips per stop
        stop_freq = stop_times.groupby('stop_id').agg({
            'trip_id': 'nunique'
        }).reset_index()
        stop_freq.columns = ['stop_id', 'daily_trips']
        
        # Merge with stops
        gdf = gdf.merge(stop_freq, on='stop_id', how='left')
        gdf['daily_trips'] = gdf['daily_trips'].fillna(0).astype(int)
        
        metadata.statistics['fields_created'].append('daily_trips')
        metadata.log_transformation(
            'frequency_calculation',
            f"Calculated daily trips for {len(stop_freq)} stops"
        )
        
        print(f"  ✓ Added daily_trips field")
        print(f"  ℹ️  Average trips/stop: {gdf['daily_trips'].mean():.1f}")
        print(f"  ℹ️  Max trips/stop: {gdf['daily_trips'].max()}")
        
    except Exception as e:
        print(f"  ⚠️  Could not calculate frequency: {e}")
        gdf['daily_trips'] = 0
    
    # ========================================================================
    # STEP 6: Calculate route count per stop
    # ========================================================================
    print("\n🛤️  Calculating route diversity...")
    
    try:
        trips_path = cfg.paths.data_raw / "gtfs" / "trips.txt"
        trips = pd.read_csv(trips_path)
        
        # Join stop_times with trips to get routes
        stop_times_routes = stop_times.merge(trips[['trip_id', 'route_id']], on='trip_id')
        
        # Count unique routes per stop
        route_counts = stop_times_routes.groupby('stop_id')['route_id'].nunique().reset_index()
        route_counts.columns = ['stop_id', 'num_routes']
        
        gdf = gdf.merge(route_counts, on='stop_id', how='left')
        gdf['num_routes'] = gdf['num_routes'].fillna(0).astype(int)
        
        metadata.statistics['fields_created'].append('num_routes')
        print(f"  ✓ Added num_routes field")
        print(f"  ℹ️  Average routes/stop: {gdf['num_routes'].mean():.1f}")
        
    except Exception as e:
        print(f"  ⚠️  Could not calculate routes: {e}")
        gdf['num_routes'] = 0
    
    # ========================================================================
    # STEP 7: Add processing metadata
    # ========================================================================
    gdf['processing_date'] = datetime.now().isoformat()
    gdf['data_source'] = 'GTFS (CARTA)'
    
    # ========================================================================
    # STEP 8: Save processed data
    # ========================================================================
    print("\n💾 Saving processed transit stops...")
    
    metadata.statistics['final_record_count'] = len(gdf)
    
    output_path = cfg.paths.data_processed / "transit_stops_clean.gpkg"
    gdf.to_file(output_path, driver='GPKG')
    print(f"  ✓ Saved: {output_path}")
    
    # Save metadata
    metadata_path = cfg.paths.data_processed / "transit_stops_metadata.json"
    metadata.save_json(metadata_path)
    
    # Summary
    print("\n📊 Processing Summary:")
    print(f"  • Raw stops: {metadata.statistics['raw_record_count']}")
    print(f"  • Valid stops: {metadata.statistics['final_record_count']}")
    print(f"  • Removed: {metadata.statistics['records_removed']}")
    
    return gdf


# ============================================================================
# MAIN PREPARATION ORCHESTRATOR
# ============================================================================

def prepare_all_data(config_path: Optional[str] = None):
    """
    Prepare all datasets for analysis
    
    Runs:
        1. Census blocks preparation
        2. Parking lots preparation
        3. GTFS stops preparation
    
    Creates clean, analysis-ready datasets in data/processed/
    """
    cfg = Config(config_path)
    
    print("\n" + "="*80)
    print("DATA PREPARATION & HARMONIZATION")
    print("="*80)
    print(f"Project: {cfg.project_name}")
    print(f"Target CRS: EPSG:{cfg.crs.analysis_utm} (UTM Zone 16N)")
    print("="*80)
    
    start_time = datetime.now()
    
    # Prepare each dataset
    try:
        census_gdf = prepare_census_blocks(cfg)
        parking_gdf = prepare_parking_lots(cfg)
        stops_gdf = prepare_gtfs_stops(cfg)
        
        # Final summary
        print("\n" + "="*80)
        print("✅ DATA PREPARATION COMPLETE")
        print("="*80)
        
        duration = (datetime.now() - start_time).total_seconds()
        print(f"\nTotal processing time: {duration:.1f} seconds")
        
        print(f"\n📁 Processed datasets saved to: {cfg.paths.data_processed}")
        print(f"\nDatasets ready for analysis:")
        print(f"  ✓ Census blocks: {len(census_gdf)} features")
        print(f"  ✓ Parking lots: {len(parking_gdf)} features")
        print(f"  ✓ Transit stops: {len(stops_gdf)} features")
        
        print(f"\n📊 All datasets in CRS: EPSG:{cfg.crs.analysis_utm}")
        print(f"📝 Processing metadata saved alongside each dataset")
        
        print("\n" + "="*80)
        print("NEXT STEP: Accessibility Analysis")
        print("="*80)
        print("Run: python src/02_accessibility/transit_access.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ PREPARATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    """Run data preparation"""
    
    success = prepare_all_data()
    
    if success:
        print("\n✅ Data preparation successful")
        exit(0)
    else:
        print("\n❌ Data preparation failed")
        exit(1)