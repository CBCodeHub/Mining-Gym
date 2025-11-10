"""
Minimal scenario loader for test configurations
"""

def load_scenario(scenario_name):
    """
    Load test scenario parameters from T_scene_config.txt
    
    Args:
        scenario_name: Name like 'A', 'B', 'C', etc. or None for normal mode
        
    Returns:
        dict: Parameter overrides or empty dict if no scenario
    """
    if scenario_name is None:
        return {}
    
    scenario_name = scenario_name.upper()
    section_name = f"[SCENARIO_{scenario_name}]"
    
    try:
        with open('T_scene_config.txt', 'r') as f:
            lines = f.readlines()
        
        # Find the scenario section
        in_section = False
        overrides = {}
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Check if entering the target section
            if line == section_name:
                in_section = True
                continue
            
            # Check if entering a different section
            if line.startswith('[') and in_section:
                break
            
            # Parse parameters in the target section
            if in_section and ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # Map to internal parameter names
                param_map = {
                    'epsilon': 'Eps',
                    'shovels_to_fail': 'STF',
                    'trucks_to_fail': 'TTF',
                    'shovel_initial_breakdown': 'SIB',
                    'truck_initial_breakdown': 'TIB'
                }
                
                if key in param_map:
                    try:
                        overrides[param_map[key]] = float(value)
                    except ValueError:
                        print(f"Warning: Invalid value for {key}: {value}")
        
        if overrides:
            print(f"\nLoaded SCENARIO_{scenario_name}:")
            for k, v in overrides.items():
                print(f"  {k} = {v}")
        else:
            print(f"\nWarning: SCENARIO_{scenario_name} not found in T_scene_config.txt")
        
        return overrides
        
    except FileNotFoundError:
        print("Warning: T_scene_config.txt not found. Running without scenario overrides.")
        return {}