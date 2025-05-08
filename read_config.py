import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np

@dataclass
class Distribution:
    """Base class for different probability distributions"""
    dist_type: str
    params: Union[Tuple[float, float], float, Tuple[int, float]]  # Support for all distribution parameters

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
        # Using numpy's poisson since Python's random doesn't have built-in Poisson
        np.random.seed(int(local_random.random() * 2**32))
        return float(np.random.poisson(self.params))

class BinomialDist(Distribution):
    """Binomial distribution with n trials and p probability parameters"""
    def sample(self, local_random: random.Random) -> int:
        n, p = self.params
        # Using numpy's binomial since Python's random doesn't have built-in Binomial
        np.random.seed(int(local_random.random() * 2**32))
        return int(np.random.binomial(n=n, p=p))

class ConfigSampler:
    """
    Reads configuration and parameter information from a txt file.
    Handles multiple distribution types (Uniform, Normal, Poisson, Binomial) and returns sampled values.
    """
    
    def __init__(self, filename: str, cnfg_seed: Optional[int] = None):
        """
        Initialize the ConfigSampler with the path to the configuration file.
        
        Args:
            filename: Path to the configuration file
            cnfg_seed: Random seed for sampling (default: current timestamp)
        """

        self.configurations = self._parse_config_file(filename)
        self.sampled_data: Dict[str, float] = {}
        # Seed selection logic
        # If cnfg_seed is None, we'll decide here whether to use a fixed or random seed
        use_fixed_seed = True  # Set this to True for fixed seed, False for random seed

        if cnfg_seed is not None:
            # Use the provided seed
            seed_value = cnfg_seed
        elif use_fixed_seed:
            # Use a fixed predefined seed
            seed_value = 42  # You can change this to any fixed value you prefer
        else:
            # Use current timestamp for random seed
            seed_value = int(time.time())

        self.local_random = random.Random(seed_value)

        '''
        # Define time-related parameters
        self.time_parameters = {
            'Sdur',  # Shift duration
            'TIB',   # Time to Initial Breakdown
            'RSH',   # Repair time for Shovel/Truck
            'SIB',   # Shovel Initial Breakdown time
            'FSH',   # Time between Shovel failures
            'FTR',   # Time between Truck failures
            'STC',   # Shovel to Crusher travel time
            'STD',   # Shovel to Dump travel time
            'CTS',   # Crusher to Shovel travel time
            'DTS',   # Dump to Shovel travel time
            'TRCR',  # Time for truck at crusher
            'TRDM',  # Time for truck at dump
            'RTR',   # Truck Repair Time
            'FCR',   # Crusher Failure Time
            'FDS',   # Dumping Site Failure Time
            'RCR',   # Crusher Repair Time
            'RDS',   # Dumping Site Repair Time
            'TRL',   # Truck Loading Time
        }
        '''
    def __repr__(self) -> str:
        """
        String representation of the ConfigSampler object.
        Returns a summary of the configuration and sampled data.
        """
        return f"ConfigSampler(configurations={self.configurations}, sampled_data={self.sampled_data})"

    def _parse_distribution(self, value: str) -> Union[Distribution, float]:
        """
        Parse distribution string and return appropriate Distribution object.
        
        Args:
            value: String containing distribution information
            
        Returns:
            Distribution object or float for fixed values
        """
        value = value.strip()
        
        # Try parsing as fixed number first (integer or float)
        try:
            return int(value)
        except ValueError:
            pass

        try:
            return float(value)
        except ValueError:
            pass

        # Parse distributions
        if 'Uniform' in value:
            # Parse "Uniform Min: X, Max: Y"
            parts = value.split()
            min_val = float(parts[parts.index("Min:") + 1].strip(','))
            max_val = float(parts[parts.index("Max:") + 1])
            return UniformDist("uniform", (min_val, max_val))
            
        elif 'Normal' in value:
            # Parse "Normal (mean,std)"
            params = value.split('(')[1].split(')')[0].split(',')
            mean = float(params[0])
            std = float(params[1])
            return NormalDist("normal", (mean, std))
            
        elif 'Poisson' in value:
            # Parse "Poisson (lambda)"
            lambda_val = float(value.split('(')[1].split(')')[0])
            return PoissonDist("poisson", lambda_val)
            
        elif 'Binomial' in value:
            # Parse "Binomial (n,p)"
            params = value.split('(')[1].split(')')[0].split(',')
            n = int(params[0])  # number of trials must be integer
            p = float(params[1])  # probability must be float
            if not (0 <= p <= 1):
                raise ValueError(f"Binomial probability must be between 0 and 1, got {p}")
            return BinomialDist("binomial", (n, p))
            
        raise ValueError(f"Unknown distribution format: {value}")

    def _parse_config_file(self, filename: str) -> Dict[str, Union[Distribution, float]]:
        """
        Parse the configuration file and extract distribution parameters and fixed values.
        
        Args:
            filename: Path to the configuration file
            
        Returns:
            Dictionary mapping parameter names to their distributions or fixed values
        """
        distributions: Dict[str, Union[Distribution, float]] = {}
        
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith('%'):
                    continue
                    
                if ':' in line:
                    key, value = [part.strip() for part in line.split(':', 1)]
                    # Extract actual parameter name if provided in parentheses
                    if '(' in key and ')' in key:
                        key = key.split('(')[1].split(')')[0]
                    distributions[key] = self._parse_distribution(value)
                    
        return distributions

    def get_sampled_value(self, key: str) -> Union[float, int]:
        """
        Get a sampled value for the given parameter key.
        
        Args:
            key: Parameter name to sample
            
        Returns:
            Sampled value for the parameter (float for most distributions, int for Binomial)
            
        Raises:
            KeyError: If key not found in configuration
        """
        if key in self.sampled_data:
            return self.sampled_data[key]
            
        value = self.configurations.get(key)
        if value is None:
            raise KeyError(f"Key '{key}' not found in the configuration.")
            
        if isinstance(value, Distribution):
            sampled_value = value.sample(self.local_random)
        else:  # Fixed value
            sampled_value = value

        # Scale the value if it's a time parameter
        #if key in self.time_parameters:
        #    sampled_value = sampled_value / self.time_scale
            
        self.sampled_data[key] = sampled_value
        return sampled_value

    def reset_samples(self):
        """Clear all cached sampled values"""
        self.sampled_data.clear()