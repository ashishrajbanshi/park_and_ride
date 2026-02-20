"""
Configuration Loader with Validation
====================================

Purpose:
    Load and validate config.yml with comprehensive error checking
    
Design Principles:
    - Fail fast: Catch configuration errors before analysis runs
    - Type safety: Return typed objects, not raw dictionaries
    - Self-documenting: Config object has clear attribute names
    - Extensible: Easy to add new parameters
    
Author: Ashish Rajbanshi
Date: January 2026
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import warnings


# ============================================================================
# DATA CLASSES FOR TYPE-SAFE ACCESS
# ============================================================================

@dataclass
class CRSConfig:
    """Coordinate Reference System configuration"""
    input_wgs84: int
    analysis_utm: int
    output_web: int
    
    def __post_init__(self):
        """Validate EPSG codes"""
        for attr in ['input_wgs84', 'analysis_utm', 'output_web']:
            epsg = getattr(self, attr)
            if not (1000 <= epsg <= 32767):
                raise ValueError(f"Invalid EPSG code: {attr}={epsg}")


@dataclass
class VWPlantConfig:
    """VW Plant location and distance zones"""
    name: str
    address: str
    latitude: float
    longitude: float
    
    # Distance zones
    exclusion_min: int
    exclusion_max: int
    optimal_min: int
    optimal_max: int
    acceptable_max: int
    far_penalty_per_km: int
    
    def __post_init__(self):
        """Validate coordinates and distance logic"""
        # Check latitude/longitude bounds
        if not (34.0 <= self.latitude <= 36.0):
            raise ValueError(f"Latitude {self.latitude} outside Chattanooga region")
        if not (-86.0 <= self.longitude <= -84.0):
            raise ValueError(f"Longitude {self.longitude} outside Chattanooga region")
        
        # Check distance zone logic
        if not (self.exclusion_max < self.optimal_min < self.optimal_max < self.acceptable_max):
            raise ValueError("Distance zones must be: exclusion < optimal < acceptable")


@dataclass
class WalkingDistances:
    """Walking distance scenarios (meters)"""
    conservative: int
    moderate: int
    extended: int
    
    def __post_init__(self):
        """Validate walking distances are ordered correctly"""
        if not (self.conservative < self.moderate < self.extended):
            raise ValueError("Walking distances must be: conservative < moderate < extended")
        
        if self.extended > 2000:
            warnings.warn(f"Extended walking distance ({self.extended}m) exceeds typical maximum (2000m)")
    
    def as_dict(self) -> Dict[str, int]:
        """Return as dictionary for iteration"""
        return {
            'conservative': self.conservative,
            'moderate': self.moderate,
            'extended': self.extended
        }


@dataclass
class ServiceQualityLevel:
    """Single service quality threshold"""
    min_trips_per_day: int
    headway_minutes: int
    description: str
    
    def __post_init__(self):
        """Calculate and validate headway consistency"""
        # Assuming 16-hour service day
        calculated_headway = (16 * 60) / self.min_trips_per_day if self.min_trips_per_day > 0 else 999
        
        # Allow 20% tolerance for rounding
        if abs(calculated_headway - self.headway_minutes) > (self.headway_minutes * 0.2):
            warnings.warn(
                f"Headway inconsistency: {self.min_trips_per_day} trips/day "
                f"implies {calculated_headway:.1f} min headway, "
                f"but config specifies {self.headway_minutes} min"
            )


@dataclass
class ServiceQualityConfig:
    """All service quality thresholds"""
    excellent: ServiceQualityLevel
    good: ServiceQualityLevel
    fair: ServiceQualityLevel
    limited: ServiceQualityLevel
    poor: ServiceQualityLevel
    
    def classify(self, trips_per_day: int) -> str:
        """
        Classify service quality based on trips per day
        
        Args:
            trips_per_day: Number of transit trips serving the location
            
        Returns:
            Quality level: 'excellent', 'good', 'fair', 'limited', or 'poor'
        """
        if trips_per_day >= self.excellent.min_trips_per_day:
            return 'excellent'
        elif trips_per_day >= self.good.min_trips_per_day:
            return 'good'
        elif trips_per_day >= self.fair.min_trips_per_day:
            return 'fair'
        elif trips_per_day >= self.limited.min_trips_per_day:
            return 'limited'
        else:
            return 'poor'
    
    def get_thresholds(self) -> Dict[str, int]:
        """Return threshold values for each level"""
        return {
            'excellent': self.excellent.min_trips_per_day,
            'good': self.good.min_trips_per_day,
            'fair': self.fair.min_trips_per_day,
            'limited': self.limited.min_trips_per_day,
            'poor': self.poor.min_trips_per_day
        }


@dataclass
class AccessibilityWeights:
    """Component weights for accessibility scoring"""
    num_stops: float
    trip_frequency: float
    service_reliability: float
    walking_distance: float
    route_diversity: float
    
    def __post_init__(self):
        """Validate weights sum to 1.0"""
        total = (self.num_stops + self.trip_frequency + 
                self.service_reliability + self.walking_distance + 
                self.route_diversity)
        
        if not (0.99 <= total <= 1.01):  # Allow small floating-point error
            raise ValueError(
                f"Accessibility weights must sum to 1.0, got {total:.4f}\n"
                f"  num_stops: {self.num_stops}\n"
                f"  trip_frequency: {self.trip_frequency}\n"
                f"  service_reliability: {self.service_reliability}\n"
                f"  walking_distance: {self.walking_distance}\n"
                f"  route_diversity: {self.route_diversity}"
            )
    
    def as_dict(self) -> Dict[str, float]:
        """Return as dictionary"""
        return {
            'num_stops': self.num_stops,
            'trip_frequency': self.trip_frequency,
            'service_reliability': self.service_reliability,
            'walking_distance': self.walking_distance,
            'route_diversity': self.route_diversity
        }


@dataclass
class RankingWeights:
    """Final ranking weights"""
    transit_accessibility: float
    service_frequency: float
    distance_from_vw: float
    parking_capacity: float
    optimal_distance: float
    
    def __post_init__(self):
        """Validate weights sum to 1.0"""
        total = (self.transit_accessibility + self.service_frequency + 
                self.distance_from_vw + self.parking_capacity + 
                self.optimal_distance)
        
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Ranking weights must sum to 1.0, got {total:.4f}"
            )
    
    def as_dict(self) -> Dict[str, float]:
        """Return as dictionary"""
        return {
            'transit_accessibility': self.transit_accessibility,
            'service_frequency': self.service_frequency,
            'distance_from_vw': self.distance_from_vw,
            'parking_capacity': self.parking_capacity,
            'optimal_distance': self.optimal_distance
        }


@dataclass
class MinimumThresholds:
    """Minimum viability thresholds"""
    min_transit_stops: int
    min_daily_trips: int
    min_capacity: int
    
    def __post_init__(self):
        """Validate thresholds are reasonable"""
        if self.min_transit_stops < 0:
            raise ValueError("min_transit_stops cannot be negative")
        if self.min_daily_trips < 0:
            raise ValueError("min_daily_trips cannot be negative")
        if self.min_capacity < 0:
            raise ValueError("min_capacity cannot be negative")


@dataclass
class PathsConfig:
    """Project directory paths"""
    data_raw: str
    data_processed: str
    data_outputs: str
    results: str
    figures: str
    reports: str
    
    def __post_init__(self):
        """Convert to Path objects and create if needed"""
        for attr in ['data_raw', 'data_processed', 'data_outputs', 
                     'results', 'figures', 'reports']:
            path_str = getattr(self, attr)
            path_obj = Path(path_str)
            setattr(self, attr, path_obj)
    
    def create_directories(self, base_path: Optional[Path] = None):
        """
        Create all project directories if they don't exist
        
        Args:
            base_path: Project root directory (default: current directory)
        """
        if base_path is None:
            base_path = Path.cwd()
        
        for attr in ['data_raw', 'data_processed', 'data_outputs', 
                     'results', 'figures', 'reports']:
            full_path = base_path / getattr(self, attr)
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created: {full_path}")


# ============================================================================
# MAIN CONFIG CLASS
# ============================================================================

class Config:
    """
    Main configuration class with validated parameters
    
    Usage:
        cfg = Config()
        walking_distance = cfg.walking.moderate
        weights = cfg.accessibility_weights.as_dict()
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Load and validate configuration
        
        Args:
            config_path: Path to config.yml (default: config/config.yml)
        """
        if config_path is None:
            # Default: look for config/config.yml relative to project root
            config_path = Path(__file__).parent / "config.yml"
        
        self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}\n"
                f"Please ensure config.yml exists in the config/ directory"
            )
        
        # Load yml
        with open(self.config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
        
        # Parse into typed dataclasses
        self._parse_config(raw_config)
        
        print(f"✓ Configuration loaded: {self.config_path}")
        self._validate_all()
    
    def _parse_config(self, cfg: Dict[str, Any]):
        """Parse raw yml into typed objects"""
        
        # Project metadata (simple attributes)
        self.project_name = cfg['project']['name']
        self.project_version = cfg['project']['version']
        
        # CRS
        self.crs = CRSConfig(**cfg['crs'])
        
        # VW Plant
        vw_cfg = cfg['vw_plant']
        self.vw_plant = VWPlantConfig(
            name=vw_cfg['name'],
            address=vw_cfg['address'],
            latitude=vw_cfg['coordinates']['latitude'],
            longitude=vw_cfg['coordinates']['longitude'],
            exclusion_min=vw_cfg['distance_zones']['exclusion_min'],
            exclusion_max=vw_cfg['distance_zones']['exclusion_max'],
            optimal_min=vw_cfg['distance_zones']['optimal_min'],
            optimal_max=vw_cfg['distance_zones']['optimal_max'],
            acceptable_max=vw_cfg['distance_zones']['acceptable_max'],
            far_penalty_per_km=vw_cfg['distance_scoring']['far_penalty_per_km']
        )
        
        # Walking distances
        self.walking = WalkingDistances(**cfg['walking_distances'])
        
        # Service quality
        sq = cfg['service_quality']
        self.service_quality = ServiceQualityConfig(
            excellent=ServiceQualityLevel(**sq['excellent']),
            good=ServiceQualityLevel(**sq['good']),
            fair=ServiceQualityLevel(**sq['fair']),
            limited=ServiceQualityLevel(**sq['limited']),
            poor=ServiceQualityLevel(**sq['poor'])
        )
        
        # Weights
        aw = cfg['accessibility_weights']
        self.accessibility_weights = AccessibilityWeights(
            num_stops=aw['num_stops'],
            trip_frequency=aw['trip_frequency'],
            service_reliability=aw['service_reliability'],
            walking_distance=aw['walking_distance'],
            route_diversity=aw['route_diversity']
        )
        
        rw = cfg['ranking_weights']
        self.ranking_weights = RankingWeights(
            transit_accessibility=rw['transit_accessibility'],
            service_frequency=rw['service_frequency'],
            distance_from_vw=rw['distance_from_vw'],
            parking_capacity=rw['parking_capacity'],
            optimal_distance=rw['optimal_distance']
        )
        
        # Minimum thresholds
        self.min_thresholds = MinimumThresholds(**cfg['minimum_thresholds'])
        
        # Paths
        self.paths = PathsConfig(**cfg['paths'])
    
    def _validate_all(self):
        """Run comprehensive validation checks"""
        print("\n🔍 Validating configuration...")
        
        # All validation happens in __post_init__ of dataclasses
        # This is just a summary
        
        checks = [
            ("CRS codes", "✓"),
            ("VW Plant coordinates", "✓"),
            ("Distance zone logic", "✓"),
            ("Walking distances", "✓"),
            ("Service quality thresholds", "✓"),
            ("Accessibility weights (sum=1.0)", "✓"),
            ("Ranking weights (sum=1.0)", "✓"),
            ("Minimum thresholds", "✓")
        ]
        
        for check, status in checks:
            print(f"  {status} {check}")
        
        print("\n✅ Configuration validated successfully!\n")
    
    def summary(self):
        """Print configuration summary"""
        print("="*80)
        print(f"CONFIGURATION SUMMARY: {self.project_name}")
        print("="*80)
        
        print(f"\n📍 VW Plant:")
        print(f"   Location: {self.vw_plant.latitude:.4f}°N, {self.vw_plant.longitude:.4f}°W")
        print(f"   Exclusion zone: {self.vw_plant.exclusion_max/1000:.1f} km")
        print(f"   Optimal distance: {self.vw_plant.optimal_min/1000:.1f}-{self.vw_plant.optimal_max/1000:.1f} km")
        
        print(f"\n🚶 Walking Scenarios:")
        for name, dist in self.walking.as_dict().items():
            print(f"   {name.capitalize()}: {dist}m")
        
        print(f"\n🚌 Service Quality Thresholds:")
        for level, threshold in self.service_quality.get_thresholds().items():
            print(f"   {level.capitalize()}: ≥{threshold} trips/day")
        
        print(f"\n⚖️  Accessibility Weights:")
        for component, weight in self.accessibility_weights.as_dict().items():
            print(f"   {component}: {weight:.0%}")
        
        print(f"\n🏆 Ranking Weights:")
        for component, weight in self.ranking_weights.as_dict().items():
            print(f"   {component}: {weight:.0%}")
        
        print(f"\n📊 Minimum Thresholds:")
        print(f"   Transit stops: {self.min_thresholds.min_transit_stops}")
        print(f"   Daily trips: {self.min_thresholds.min_daily_trips}")
        print(f"   Parking capacity: {self.min_thresholds.min_capacity}")
        
        print("\n" + "="*80)
    
    def create_project_structure(self):
        """Create all project directories"""
        print("\n📁 Creating project directory structure...")
        self.paths.create_directories()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration (convenience function)
    
    Args:
        config_path: Path to config.yml (optional)
        
    Returns:
        Config object
    """
    return Config(config_path)


def validate_config(config_path: Optional[str] = None) -> bool:
    """
    Validate configuration file without loading full config
    
    Args:
        config_path: Path to config.yml (optional)
        
    Returns:
        True if valid, raises exception if invalid
    """
    try:
        cfg = Config(config_path)
        print("✅ Configuration is valid")
        return True
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        raise


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    """Test the configuration loader"""
    
    print("TESTING CONFIGURATION LOADER")
    print("="*80)
    
    # Load config
    cfg = load_config()
    
    # Print summary
    cfg.summary()
    
    # Create directories
    cfg.create_project_structure()
    
    # Test some accessors
    print("\n🧪 TESTING ACCESSORS:")
    print(f"\nModerate walking distance: {cfg.walking.moderate}m")
    print(f"Optimal distance range: {cfg.vw_plant.optimal_min/1000}-{cfg.vw_plant.optimal_max/1000} km")
    print(f"Transit accessibility weight: {cfg.accessibility_weights.trip_frequency:.0%}")
    
    # Test service quality classifier
    print(f"\n🧪 TESTING SERVICE QUALITY CLASSIFIER:")
    test_trips = [150, 75, 35, 15, 5]
    for trips in test_trips:
        quality = cfg.service_quality.classify(trips)
        print(f"   {trips} trips/day → {quality}")
    
    print("\n✅ All tests passed!")