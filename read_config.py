import random
import sys
import time
import re
import os
import pickle
import hashlib
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union, Tuple, List, Set
import numpy as np
import ast


@dataclass
class Distribution:
    """Base class for different probability distributions"""
    dist_type: str
    params: Union[Tuple[float, float], float, Tuple[int, float]]

    def sample(self, local_random: random.Random) -> float:
        raise NotImplementedError

class UniformDist(Distribution):
    """Uniform distribution with min and max parameters"""
    def sample(self, local_random: random.Random) -> float:
        min_val, max_val = self.params
        return local_random.uniform(min_val, max_val)

class NormalDist(Distribution):
    """Normal distribution with mean and std parameters"""
    def sample(self, local_random: random.Random) -> float:
        mean, std = self.params
        return local_random.gauss(mean, std)

class PoissonDist(Distribution):
    """Poisson distribution with lambda parameter"""
    def sample(self, local_random: random.Random) -> float:
        np.random.seed(int(local_random.random() * 2**32))
        return float(np.random.poisson(self.params))

class BinomialDist(Distribution):
    """Binomial distribution with n trials and p probability parameters"""
    def sample(self, local_random: random.Random) -> int:
        n, p = self.params
        np.random.seed(int(local_random.random() * 2**32))
        return int(np.random.binomial(n=n, p=p))


class TruncatedNormalDist(Distribution):
    """Normal distribution with bounds - prevents negative/excessive values"""
    def __init__(self, dist_type: str, params: Tuple[float, float], 
                 lower: float, upper: float):
        super().__init__(dist_type, params)
        self.lower = lower
        self.upper = upper
    
    def sample(self, local_random: random.Random) -> float:
        mean, std = self.params
        
        # Rejection sampling with fallback
        for _ in range(100):
            value = local_random.gauss(mean, std)
            if self.lower <= value <= self.upper:
                return value
        
        # Fallback: clip
        return max(self.lower, min(self.upper, local_random.gauss(mean, std)))


class BoundedPoissonDist(Distribution):
    """Poisson distribution with bounds - prevents infinite values"""
    def __init__(self, dist_type: str, params: float, 
                 lower: float, upper: float):
        super().__init__(dist_type, params)
        self.lower = lower
        self.upper = upper
    
    def sample(self, local_random: random.Random) -> float:
        np.random.seed(int(local_random.random() * 2**32))
        value = float(np.random.poisson(self.params))
        return max(self.lower, min(self.upper, value))




@dataclass
class ShovelConfiguration:
    """Container for shovel-specific configuration"""
    shovel_id: int
    performance_class: int
    location_cluster: int
    
    def get_loading_time_key(self) -> str:
        return f"TRL_C{self.performance_class}"
    
    def get_travel_time_keys(self) -> Dict[str, str]:
        cluster = self.location_cluster
        return {
            'to_crusher': f"SZ{cluster}C",
            'to_dump': f"SZ{cluster}D"
        }

@dataclass
class CachedConfigData:
    """Container for cached configuration data with fixed/episodic separation"""
    # Configuration definitions (distributions, fixed values, lists)
    fixed_configurations: Dict[str, Union[Distribution, float, List[int]]]
    episodic_configurations: Dict[str, Distribution]
    
    # Pre-sampled fixed values (sampled once at initialization)
    fixed_sampled_data: Dict[str, Union[float, int, List[int]]]
    
    # Shovel configurations derived from fixed parameters
    shovel_configs: Dict[int, ShovelConfiguration]
    
    # Cache metadata
    config_hash: str
    cache_timestamp: float
    seed_used: int

class ConfigSampler:
    """
    Enhanced configuration sampler with separation of fixed and episodic parameters.
    
    Features:
    - Fixed parameters: Sampled once at initialization, cached permanently
    - Episodic parameters: Sampled fresh every episode
    - Automatic parameter classification based on config file structure
    - Caching for fast repeated loads of fixed parameters
    - Backward compatible interface
    """
    
    def __init__(self, filename: str, cnfg_seed: Optional[int] = None, enable_cache: bool = True, scenario_overrides: Optional[Dict[str, float]] = None):
        self.filename = filename
        self.enable_cache = enable_cache
        self.cache_dir = ".config_cache"

        # Store scenario overrides
        self.scenario_overrides = scenario_overrides or {}
        
        # Create cache directory if it doesn't exist
        if self.enable_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Seed selection logic (preserved from original)
        use_fixed_seed = True
        if cnfg_seed is not None:
            seed_value = cnfg_seed
        elif use_fixed_seed:
            seed_value = 42
        else:
            seed_value = int(time.time())
        
        self.seed_value = seed_value
        self.local_random = random.Random(self.seed_value)
        
        # Current episode data (refreshed each episode)
        self.current_episodic_data: Dict[str, Union[float, int]] = {}
        self.episode_count = 0
        
        # Try to load from cache first, otherwise parse
        if self.enable_cache and self._load_from_cache():
            pass  # Successfully loaded from cache
        else:
            self._parse_and_initialize_config()
            if self.enable_cache:
                self._save_to_cache()

    def _get_cache_filename(self) -> str:
        """Generate cache filename based on config file and seed"""
        base_name = os.path.splitext(os.path.basename(self.filename))[0]
        return os.path.join(self.cache_dir, f"{base_name}_seed{self.seed_value}.pkl")

    def _get_config_hash(self) -> str:
        """Generate hash of config file content for change detection"""
        try:
            with open(self.filename, 'r') as f:
                content = f.read()
            return hashlib.md5(content.encode()).hexdigest()
        except:
            return ""

    def _get_fixed_parameter_keys(self) -> Set[str]:
        """
        Define which parameters are fixed (sampled once) vs episodic (sampled per episode).
        Based on the config file structure and mining operational reality.
        """
        return {
            # Equipment counts - define environment structure
            'TR', 'Num_trucks',
            'SH', 'Num_shovels', 
            'CR', 'Num_crushers',
            'DS', 'Num_dumps',
            'scheduler_choice',
            'Sdur', 'Shift_duration',
            
            # Equipment characteristics - inherent mine infrastructure
            'shovel_performance_class',
            'shovel_location_cluster',
            
            # Infrastructure-based parameters - fixed mine layout
            'SZ1C', 'SZ1D', 'SZ2C', 'SZ2D', 'SZ3C', 'SZ3D',
            
            # Equipment-specific operational parameters - inherent to equipment
            'TRDM', 'truck_dumping_dmp',
            'TRCR', 'truck_dumping_crush',
            
            # Base reliability parameters - standardized repair procedures (MTTR)
            'RSH', 'RTR', 'RCR', 'RDS',
            
            # Target production - set for longer periods
            'PVol_targ', 'target_pvol'
        }

    def _get_episodic_parameter_keys(self) -> Set[str]:
        """
        Define parameters that change per episode/shift.
        These represent daily operational variability.
        """
        return {
            # Daily operational variability
            'Eps', 'epsilon',
            
            # Cost parameters - fluctuate with market conditions
            'known_cost', 'estimated_cost',
            
            # Equipment performance variations - daily conditions/operator skill
            'TRL_C1', 'TRL_C2', 'TRL_C3',
            
            # Daily operational conditions
            'FW', 'truck_waste',
            'FO', 'truck_ore', 
            'FE', 'truck_empty',
            'TE', 'TL',
            
            # Equipment health status - varies with usage/maintenance (MTBF)
            'FSH', 'FTR', 'FCR', 'FDS',
            
            # Experimental/scenario parameters
            'STF', 'shovels_to_fail',
            'TTF', 'trucks_to_fail', 
            'SIB', 'shovel_initial_breakdown',
            'TIB', 'truck_initial_breakdown'
        }

    def _classify_parameter(self, key: str) -> str:
        """Classify parameter as 'fixed' or 'episodic'"""
        fixed_keys = self._get_fixed_parameter_keys()
        episodic_keys = self._get_episodic_parameter_keys()
        
        if key in fixed_keys:
            return 'fixed'
        elif key in episodic_keys:
            return 'episodic'
        
        # Check aliases
        aliases = self._get_key_aliases(key)
        for alias in aliases:
            if alias in fixed_keys:
                return 'fixed'
            elif alias in episodic_keys:
                return 'episodic'
        
        # Default to episodic for unknown parameters (safer for variability)
        return 'episodic'

    def _load_from_cache(self) -> bool:
        """Load fixed configuration from cache if valid"""
        cache_file = self._get_cache_filename()
        
        if not os.path.exists(cache_file):
            return False
        
        try:
            with open(cache_file, 'rb') as f:
                cached_data: CachedConfigData = pickle.load(f)
            
            # Verify cache is still valid
            current_hash = self._get_config_hash()
            config_mod_time = os.path.getmtime(self.filename) if os.path.exists(self.filename) else 0
            
            if (cached_data.config_hash != current_hash or 
                cached_data.seed_used != self.seed_value or
                cached_data.cache_timestamp < config_mod_time):
                return False
            
            # Load cached data
            self.fixed_configurations = cached_data.fixed_configurations
            self.episodic_configurations = cached_data.episodic_configurations
            self.fixed_sampled_data = cached_data.fixed_sampled_data
            self.shovel_configs = cached_data.shovel_configs
            
            return True
            
        except Exception:
            return False

    def _save_to_cache(self):
        """Save fixed configuration to cache"""
        cache_file = self._get_cache_filename()
        
        try:
            cached_data = CachedConfigData(
                fixed_configurations=self.fixed_configurations,
                episodic_configurations=self.episodic_configurations,
                fixed_sampled_data=self.fixed_sampled_data,
                shovel_configs=self.shovel_configs,
                config_hash=self._get_config_hash(),
                cache_timestamp=time.time(),
                seed_used=self.seed_value
            )
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
        except Exception:
            pass  # Silently fail cache save

    def _parse_and_initialize_config(self):
        """Parse config file and separate fixed vs episodic parameters"""
        all_configurations = self._parse_config_file(self.filename)
        
        # Separate configurations by type
        self.fixed_configurations: Dict[str, Union[Distribution, float, List[int]]] = {}
        self.episodic_configurations: Dict[str, Distribution] = {}
        
        for key, value in all_configurations.items():
            param_type = self._classify_parameter(key)
            
            if param_type == 'fixed':
                self.fixed_configurations[key] = value
            else:  # episodic
                if isinstance(value, Distribution):
                    self.episodic_configurations[key] = value
                else:
                    # Non-distribution episodic parameters are treated as fixed
                    self.fixed_configurations[key] = value
        
        # Sample fixed parameters once
        self.fixed_sampled_data: Dict[str, Union[float, int, List[int]]] = {}
        self._sample_fixed_parameters()
        
        # Initialize shovel configurations from fixed parameters
        self._initialize_shovel_configs()
        
        # Initialize episode data (will be refreshed on each new_episode call)
        self.current_episodic_data = {}

    def _sample_fixed_parameters(self):
        """Sample all fixed parameters once using the main seed"""
        for key, value in self.fixed_configurations.items():
            if isinstance(value, Distribution):
                self.fixed_sampled_data[key] = value.sample(self.local_random)
            elif isinstance(value, list):
                self.fixed_sampled_data[key] = value
            else:
                self.fixed_sampled_data[key] = value

    def new_episode(self, episode_seed: Optional[int] = None):
        """
        Start a new episode by sampling all episodic parameters.
        
        Args:
            episode_seed: Optional seed for this episode. If None, uses episode counter.
        """
        self.episode_count += 1
        
        # Create episode-specific random generator
        if episode_seed is not None:
            episode_random = random.Random(episode_seed)
        else:
            # Use main seed + episode count for reproducible but varied episodes
            episode_random = random.Random(self.seed_value + self.episode_count)
        
        # Sample all episodic parameters
        self.current_episodic_data.clear()
        for key, distribution in self.episodic_configurations.items():
            self.current_episodic_data[key] = distribution.sample(episode_random)

    def get_sampled_value(self, key: str) -> Union[float, int, List[int]]:
        """
        Get sampled value - checks scenario overrides first, then fixed values, 
        then episodic values from current episode.
        Maintains backward compatibility with original interface.
        """
        # Check scenario overrides FIRST (highest priority)
        if key in self.scenario_overrides:
            return self.scenario_overrides[key]
    
        # Check aliases for scenario overrides
        aliases = self._get_key_aliases(key)
        for alias in aliases:
            if alias in self.scenario_overrides:
                return self.scenario_overrides[alias]
    
        # Check fixed parameters
        if key in self.fixed_sampled_data:
            return self.fixed_sampled_data[key]
    
        # Check current episodic data
        if key in self.current_episodic_data:
            return self.current_episodic_data[key]
    
        # Check aliases for fixed parameters
        for alias in aliases:
            if alias in self.fixed_sampled_data:
                return self.fixed_sampled_data[alias]
            if alias in self.current_episodic_data:
                return self.current_episodic_data[alias]
    
        # If not found and episodic data is empty, auto-initialize episode
        if not self.current_episodic_data:
            self.new_episode()
        
            # Try again after episode initialization
            if key in self.current_episodic_data:
                return self.current_episodic_data[key]
            for alias in aliases:
                if alias in self.current_episodic_data:
                    return self.current_episodic_data[alias]
    
        raise KeyError(f"Key '{key}' and its aliases {aliases} not found in configuration.")

    def _get_key_aliases(self, key: str) -> List[str]:
        """Enhanced alias map that includes ALL parameters from the config file"""
        alias_map = {
            # Equipment counts
            'TR': ['Num_trucks'],
            'SH': ['Num_shovels'], 
            'CR': ['Num_crushers'],
            'DS': ['Num_dumps'],
            'Sdur': ['Shift_duration'],
            
            # Production and costs
            'PVol_targ': ['target_pvol'],
            'known_cost': ['operational_cost', 'known_operational_cost'],
            'estimated_cost': ['estimated_operational_cost', 'cost_estimate'],
            
            # Fuel consumption parameters
            'FW': ['truck_waste', 'fuel_waste', 'fuel_with_waste'],
            'FO': ['truck_ore', 'fuel_ore', 'fuel_with_ore'], 
            'FE': ['truck_empty', 'fuel_empty', 'fuel_when_empty'],
            
            # Truck speed parameters
            'TE': ['truck_empty_speed', 'empty_speed', 'speed_empty'],
            'TL': ['truck_loaded_speed', 'loaded_speed', 'speed_loaded'],
            
            # MTBF parameters
            'FSH': ['shovel_mtbf', 'Shovel', 'mtbf_shovel'],
            'FTR': ['truck_mtbf', 'Truck', 'mtbf_truck'],
            'FCR': ['crusher_mtbf', 'Crusher', 'mtbf_crusher'],
            'FDS': ['dump_mtbf', 'Dumping_site', 'mtbf_dump'],
            
            # MTTR parameters
            'RSH': ['shovel_mttr', 'repair_shovel', 'mttr_shovel'],
            'RTR': ['truck_mttr', 'repair_truck', 'mttr_truck'],
            'RCR': ['crusher_mttr', 'repair_crusher', 'mttr_crusher'],
            'RDS': ['dump_mttr', 'repair_dump', 'mttr_dump'],
            
            # Stochastic parameters
            'Eps': ['epsilon', 'stripping_ratio', 'crusher_probability'],
            
            # Configuration parameters
            'scheduler_choice': ['scheduler', 'scheduler_type', 'scheduling_algorithm'],
            
            # Dumping times
            'TRDM': ['truck_dumping_dmp'],
            'TRCR': ['truck_dumping_crush'],
            
            # Experimentation parameters
            'STF': ['shovels_to_fail'],
            'TTF': ['trucks_to_fail'],
            'SIB': ['shovel_initial_breakdown'],
            'TIB': ['truck_initial_breakdown'],
            
            # Performance class loading times
            'TRL_C1': ['loading_time_class1', 'high_performance_loading'],
            'TRL_C2': ['loading_time_class2', 'standard_loading'],
            'TRL_C3': ['loading_time_class3', 'low_performance_loading'],
            
            # Zone-to-destination travel times
            'SZ1C': ['zone1_to_crusher', 'sz1_crusher'],
            'SZ1D': ['zone1_to_dump', 'sz1_dump'],
            'SZ2C': ['zone2_to_crusher', 'sz2_crusher'],
            'SZ2D': ['zone2_to_dump', 'sz2_dump'],
            'SZ3C': ['zone3_to_crusher', 'sz3_crusher'],
            'SZ3D': ['zone3_to_dump', 'sz3_dump'],
        }
        
        return alias_map.get(key, [])

    def _parse_list_parameter(self, value: str) -> List[int]:
        """Parse list parameters like [1, 2, 1, 3, 2, 3, 1, 2]"""
        try:
            clean_value = value.strip()
            return ast.literal_eval(clean_value)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Invalid list format: {value}. Error: {e}")

    def _extract_key_from_line(self, line: str) -> Tuple[str, str]:
        """Extract key and value from various line formats"""
        if ':' not in line:
            raise ValueError(f"Invalid line format (no colon): {line}")
        
        key_part, value_part = [part.strip() for part in line.split(':', 1)]
        
        # Remove inline comments
        if '%' in value_part:
            value_part = value_part.split('%')[0].strip()
        if '//' in value_part:
            value_part = value_part.split('//')[0].strip()
        
        # Handle parentheses in key part
        if '(' in key_part and ')' in key_part:
            match = re.search(r'\(([^)]+)\)', key_part)
            if match:
                key = match.group(1).strip()
            else:
                key = key_part.split('(')[0].strip()
        else:
            key = key_part
            
        return key, value_part

    def _normalize_distribution_format(self, value: str) -> str:
        """Normalize distribution formats by removing extra spaces"""
        value = value.strip()
        patterns = [
            (r'Normal\s+\(', 'Normal('),
            (r'Poisson\s+\(', 'Poisson('),
            (r'Binomial\s+\(', 'Binomial('),
            (r'Uniform\s+\(', 'Uniform(')
        ]
        
        for pattern, replacement in patterns:
            value = re.sub(pattern, replacement, value)
            
        return value

    def _parse_distribution(self, value: str) -> Union[Distribution, float, List[int]]:
        """Enhanced distribution parser"""
        value = self._normalize_distribution_format(value)
        
        if value.startswith('[') and value.endswith(']'):
            return self._parse_list_parameter(value)
        
        try:
            return int(value)
        except ValueError:
            pass

        try:
            return float(value)
        except ValueError:
            pass

        if 'Uniform' in value and 'Min:' in value and 'Max:' in value:
            parts = value.split()
            min_idx = parts.index("Min:") + 1
            max_idx = parts.index("Max:") + 1
            min_val = float(parts[min_idx].strip(','))
            max_val = float(parts[max_idx])
            return UniformDist("uniform", (min_val, max_val))
            
        elif 'Normal(' in value:
            params_str = value.split('Normal(')[1].split(')')[0]
            params = [float(p.strip()) for p in params_str.split(',')]
            if len(params) != 2:
                raise ValueError(f"Normal distribution requires exactly 2 parameters: {value}")
            return NormalDist("normal", (params[0], params[1]))
            
        elif 'Poisson(' in value:
            lambda_str = value.split('Poisson(')[1].split(')')[0]
            lambda_val = float(lambda_str.strip())
            return PoissonDist("poisson", lambda_val)
            
        elif 'Binomial(' in value:
            params_str = value.split('Binomial(')[1].split(')')[0]
            params = params_str.split(',')
            if len(params) != 2:
                raise ValueError(f"Binomial distribution requires exactly 2 parameters: {value}")
            n = int(params[0].strip())
            p = float(params[1].strip())
            if not (0 <= p <= 1):
                raise ValueError(f"Binomial probability must be between 0 and 1, got {p}")
            return BinomialDist("binomial", (n, p))
            
        raise ValueError(f"Unknown distribution format: {value}")

    def _parse_config_file(self, filename: str) -> Dict[str, Union[Distribution, float, List[int]]]:
        """Parse configuration file with robust format handling"""
        distributions = {}
        
        with open(filename, 'r') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line or line.startswith('%') or line.startswith('#'):
                    continue
                    
                try:
                    key, value = self._extract_key_from_line(line)
                    distributions[key] = self._parse_distribution(value)
                except Exception:
                    continue
                    
        return distributions

    def _initialize_shovel_configs(self):
        """Initialize shovel configurations from fixed parameters"""
        self.shovel_configs: Dict[int, ShovelConfiguration] = {}
        
        performance_classes = self.fixed_sampled_data.get('shovel_performance_class')
        location_clusters = self.fixed_sampled_data.get('shovel_location_cluster')
        
        if performance_classes is not None and location_clusters is not None:
            if len(performance_classes) != len(location_clusters):
                raise ValueError("Performance classes and location clusters must have same length")
            
            for i, (perf_class, loc_cluster) in enumerate(zip(performance_classes, location_clusters)):
                self.shovel_configs[i] = ShovelConfiguration(
                    shovel_id=i,
                    performance_class=perf_class,
                    location_cluster=loc_cluster
                )

    # Shovel-specific methods (preserved from original)
    def get_shovel_loading_time(self, shovel_id: int) -> float:
        """Get loading time for specific shovel based on its performance class"""
        if shovel_id in self.shovel_configs:
            config = self.shovel_configs[shovel_id]
            loading_key = config.get_loading_time_key()
            return self.get_sampled_value(loading_key)
        else:
            try:
                return self.get_sampled_value('TRL')
            except KeyError:
                available_classes = []
                for i in range(1, 4):
                    try:
                        available_classes.append(self.get_sampled_value(f'TRL_C{i}'))
                    except KeyError:
                        continue
                if available_classes:
                    return sum(available_classes) / len(available_classes)
                else:
                    raise KeyError("No loading time parameters found")

    def get_shovel_travel_time(self, shovel_id: int, destination: str) -> float:
        """Get travel time for specific shovel to destination"""
        if shovel_id in self.shovel_configs:
            config = self.shovel_configs[shovel_id]
            travel_keys = config.get_travel_time_keys()
            
            if destination.lower() == 'crusher':
                return self.get_sampled_value(travel_keys['to_crusher'])
            elif destination.lower() == 'dump':
                return self.get_sampled_value(travel_keys['to_dump'])
            else:
                raise ValueError(f"Unknown destination: {destination}")
        else:
            if destination.lower() == 'crusher':
                return self.get_sampled_value('STC')
            elif destination.lower() == 'dump':
                return self.get_sampled_value('STD')
            else:
                raise ValueError(f"Unknown destination: {destination}")

    def get_return_travel_time(self, shovel_id: int, origin: str) -> float:
        """Get return travel time from destination back to shovel"""
        if shovel_id in self.shovel_configs:
            config = self.shovel_configs[shovel_id]
            travel_keys = config.get_travel_time_keys()
            
            if origin.lower() == 'crusher':
                return self.get_sampled_value(travel_keys['to_crusher'])
            elif origin.lower() == 'dump':
                return self.get_sampled_value(travel_keys['to_dump'])
            else:
                raise ValueError(f"Unknown origin: {origin}")
        else:
            if origin.lower() == 'crusher':
                return self.get_sampled_value('CTS')
            elif origin.lower() == 'dump':
                return self.get_sampled_value('DTS')
            else:
                raise ValueError(f"Unknown origin: {origin}")

    # Utility methods
    def is_heterogeneous_config(self) -> bool:
        """Check if configuration includes heterogeneous shovel parameters"""
        return bool(self.shovel_configs)

    def get_parameter_info(self) -> Dict[str, Any]:
        """Get detailed information about parameter classification"""
        return {
            'fixed_parameters': list(self.fixed_configurations.keys()),
            'episodic_parameters': list(self.episodic_configurations.keys()),
            'shovel_configs': len(self.shovel_configs),
            'episode_count': self.episode_count,
            'cache_enabled': self.enable_cache
        }

    def print_parameter_classification(self):
        """Print classification of all parameters"""
        print("Parameter Classification:")
        print("=" * 50)
        print("FIXED PARAMETERS (sampled once):")
        for key in sorted(self.fixed_configurations.keys()):
            value = self.fixed_sampled_data.get(key, "Not sampled")
            print(f"  {key}: {value}")
        
        print("\nEPISODIC PARAMETERS (sampled per episode):")
        for key in sorted(self.episodic_configurations.keys()):
            dist_info = self.episodic_configurations[key]
            current_value = self.current_episodic_data.get(key, "Not sampled yet")
            print(f"  {key}: {dist_info.dist_type}{dist_info.params} -> {current_value}")
        print("=" * 50)

    def clear_cache(self, specific_seed: Optional[int] = None):
        """Clear cache files"""
        if specific_seed is not None:
            temp_seed = self.seed_value
            self.seed_value = specific_seed
            cache_file = self._get_cache_filename()
            self.seed_value = temp_seed
            
            if os.path.exists(cache_file):
                os.remove(cache_file)
        else:
            if os.path.exists(self.cache_dir):
                base_name = os.path.splitext(os.path.basename(self.filename))[0]
                pattern = f"{base_name}_seed"
                
                for file in os.listdir(self.cache_dir):
                    if file.startswith(pattern) and file.endswith('.pkl'):
                        os.remove(os.path.join(self.cache_dir, file))

    def __repr__(self) -> str:
        config_type = "heterogeneous" if self.shovel_configs else "homogeneous"
        return f"ConfigSampler({config_type}, {len(self.fixed_configurations)} fixed, {len(self.episodic_configurations)} episodic, episode {self.episode_count})"


    def enable_validation(self):
        """
        Enable validation for all distributions
        Call this ONCE after __init__
        """
        VALIDATION_BOUNDS = {
            # Time parameters (min=1, max=60 minutes)
            'SZ1C': (1.0, 60.0), 'SZ1D': (1.0, 60.0), 
            'SZ2C': (1.0, 60.0), 'SZ2D': (1.0, 60.0),
            'SZ3C': (1.0, 60.0), 'SZ3D': (1.0, 60.0),
        
            # Dumping times (min=1, max=30 minutes)
            'TRDM': (1.0, 30.0), 'truck_dumping_dmp': (1.0, 30.0),
            'TRCR': (1.0, 30.0), 'truck_dumping_crush': (1.0, 30.0),
        
            # Loading times (min=2, max=30 minutes)
            'TRL_C1': (2.0, 30.0), 'TRL_C2': (2.0, 30.0), 'TRL_C3': (2.0, 30.0),
        
            # Epsilon (ore grade) CRITICAL - must be [0, 1]
            'Eps': (0.0, 1.0), 'epsilon': (0.0, 1.0),
        
            # Speed (min=10, max=67.1 km/hr)
            'TE': (10.0, 67.1), 'truck_empty_speed': (10.0, 67.1),
            'TL': (10.0, 67.1), 'truck_loaded_speed': (10.0, 67.1),
        
            # Fuel consumption (L/ton-km)
            'FW': (0.20, 0.50), 'truck_waste': (0.20, 0.50),
            'FO': (0.20, 0.50), 'truck_ore': (0.20, 0.50),
            'FE': (0.20, 0.50), 'truck_empty': (0.20, 0.50),
        
            # MTTR - should not exceed shift duration
            'RSH': (1.0, 360.0), 'shovel_mttr': (1.0, 360.0),
            'RTR': (1.0, 360.0), 'truck_mttr': (1.0, 360.0),
            'RCR': (1.0, 360.0), 'crusher_mttr': (1.0, 360.0),
            'RDS': (1.0, 360.0), 'dump_mttr': (1.0, 360.0),
        
            # MTBF - should not exceed shift duration
            'FSH': (1.0, 360.0), 'shovel_mtbf': (1.0, 360.0),
            'FTR': (1.0, 360.0), 'truck_mtbf': (1.0, 360.0),
            'FCR': (1.0, 360.0), 'crusher_mtbf': (1.0, 360.0),
            'FDS': (1.0, 360.0), 'dump_mtbf': (1.0, 360.0),
        }
    
        # Wrap fixed distributions
        for key, dist in self.fixed_configurations.items():
            if key in VALIDATION_BOUNDS and isinstance(dist, NormalDist):
                lower, upper = VALIDATION_BOUNDS[key]
                self.fixed_configurations[key] = TruncatedNormalDist(
                    'truncated_normal', dist.params, lower, upper
                )
            elif key in VALIDATION_BOUNDS and isinstance(dist, PoissonDist):
                lower, upper = VALIDATION_BOUNDS[key]
                self.fixed_configurations[key] = BoundedPoissonDist(
                    'bounded_poisson', dist.params, lower, upper
                )
    
        # Wrap episodic distributions
        for key, dist in self.episodic_configurations.items():
            if key in VALIDATION_BOUNDS and isinstance(dist, NormalDist):
                lower, upper = VALIDATION_BOUNDS[key]
                self.episodic_configurations[key] = TruncatedNormalDist(
                    'truncated_normal', dist.params, lower, upper
                )
            elif key in VALIDATION_BOUNDS and isinstance(dist, PoissonDist):
                lower, upper = VALIDATION_BOUNDS[key]
                self.episodic_configurations[key] = BoundedPoissonDist(
                    'bounded_poisson', dist.params, lower, upper
                )
    
        # Re-sample fixed parameters with validated distributions
        self.fixed_sampled_data.clear()
        self._sample_fixed_parameters()
    
        # Fix load_per_trip if it exceeds truck capacity
        for load_key in ['load_per_trip', 'LO']:  # Check both possible keys
            if load_key in self.fixed_sampled_data:
                load = self.fixed_sampled_data[load_key]
                if load > 98.2:
                    self.fixed_sampled_data[load_key] = 93.3
                    print(f"⚠️  Reduced {load_key} from {load}t to 93.3t (95% of CAT 777 capacity)")
                break
    
        print(f"✅ Validation enabled: {len(VALIDATION_BOUNDS)} parameters protected")

# Usage example and testing
def test_enhanced_config_sampler():
    """Test the enhanced ConfigSampler with fixed/episodic separation"""
    
    # Initialize the sampler
    config = ConfigSampler('config_extend_review.txt', cnfg_seed=42)
    
    # Print parameter classification
    config.print_parameter_classification()
    
    print("\n" + "="*60)
    print("TESTING FIXED VS EPISODIC BEHAVIOR")
    print("="*60)
    
    # Test fixed parameters (should be same across episodes)
    print("Fixed parameter (TR) across multiple episodes:")
    tr_value = config.get_sampled_value('TR')
    print(f"Initial: {tr_value}")
    
    for episode in range(3):
        config.new_episode()
        tr_new = config.get_sampled_value('TR')
        print(f"Episode {episode + 1}: {tr_new} (same: {tr_value == tr_new})")
    
    # Test episodic parameters (should vary across episodes)
    print("\nEpisodic parameter (Eps) across multiple episodes:")
    config.new_episode()
    eps_values = []
    for episode in range(5):
        config.new_episode()
        eps_value = config.get_sampled_value('Eps')
        eps_values.append(eps_value)
        print(f"Episode {episode + 1}: {eps_value:.4f}")
    
    print(f"All values different: {len(set(eps_values)) == len(eps_values)}")
    
    # Test shovel-specific methods
    print("\nShovel-specific parameter access:")
    if config.is_heterogeneous_config():
        for shovel_id in range(min(3, config.get_sampled_value('SH'))):
            loading_time = config.get_shovel_loading_time(shovel_id)
            travel_time_crusher = config.get_shovel_travel_time(shovel_id, 'crusher')
            travel_time_dump = config.get_shovel_travel_time(shovel_id, 'dump')
            print(f"Shovel {shovel_id}: Loading={loading_time:.2f}, "
                  f"Travel to Crusher={travel_time_crusher:.2f}, "
                  f"Travel to Dump={travel_time_dump:.2f}")
    
    # Test parameter info
    print("\nParameter Info:")
    info = config.get_parameter_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    
    return config

if __name__ == "__main__":
    # Run the test
    test_config = test_enhanced_config_sampler()