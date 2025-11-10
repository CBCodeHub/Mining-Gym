import argparse
import mGym_DesEnv as denv
import numpy as np
import csv
import json
import os
import random
import sys
from scenario_loader import load_scenario


def save_temp_data(cfg_seed_info, data):
    with open(cfg_seed_info, 'w') as file:
        json.dump(data, file)

def gen_seed(iteration, initial_seed=None, ax=1664525, cx=1013904223, mx=2**32):
    """
    Generate a seed based on the iteration using a Linear Congruential Generator (LCG).
    
    - iteration: The current iteration (episode).
    - initial_seed: The starting seed. If None, use a truly random seed.
    - ax, cx, mx: Constants for the Linear Congruential Generator.
    
    Returns the seed for the given iteration.
    """
    # Use a truly random initial seed if one is not provided
    if initial_seed is None:
        initial_seed = random.randint(0, mx - 1)
    
    epi_seed = initial_seed
    for tx in range(iteration):
        epi_seed = (ax * epi_seed + cx) % mx
    return epi_seed



# Set up argparse to parse command-line arguments
parser = argparse.ArgumentParser(description="Run the simulation with specified number of iterations and algorithm choice.")
parser.add_argument('--num_episodes', type=int, default=10, help="Number of episodes to run the simulation")
parser.add_argument('--algo_choice', type=int, required=True, help="Choice of scheduling algorithm (integer only)")
parser.add_argument('--scenario', type=str, default=None, help="Test scenario (A, B, C, D, E, F)")  
parser.add_argument('--seed', type=int, default=None, help="Initial integer seed for the simulation episodes (optional)")


args = parser.parse_args()

# Get the number of iterations and algorithm choice from command line arguments
iter = args.num_episodes
algo_choice = args.algo_choice
scenario_overrides = load_scenario(args.scenario) 
initial_seed = args.seed # NEW: Get the initial seed
DUMMY_CSV_PATH = 'default_run_dummy.csv'

# Array to store KPI values for each iteration
arr = []

# Run the simulation for the specified number of iterations
for epsd in range(iter):
    if os.path.exists('alloc.json'):
        os.remove('alloc.json')
    print("Cleaned up stale alloc.json from previous run")
    
    # --- NEW: Generate a unique seed for the current episode ---
    episode_seed = gen_seed(epsd, initial_seed=initial_seed)
    random.seed(episode_seed)      
    np.random.seed(episode_seed)
    
    # Run the simulation with the specified parameters
    kpi_01 = denv.runDes(fsim=False, 
                         flag_RL_sched=False, 
                         fdef_schdlr_choice=algo_choice,
                         episode_seed=episode_seed, 
                         scenario_overrides=scenario_overrides,
                         csv_path=DUMMY_CSV_PATH ) 
    print(f"Episode {epsd+1} Seed: {episode_seed}, Value of KPI01-PVol: {kpi_01}")
    # Append the result to the list
    arr.append(kpi_01)

# Calculate the average KPI value
mean_kpi01 = np.mean(arr)
print(f"Average KPI01-PVol: {mean_kpi01}, over {iter} repeats")

# Save the array as a CSV file
scenario_suffix = f"_scenario_{args.scenario}" if args.scenario else ""
seed_suffix = f"_seed_{initial_seed}" if initial_seed is not None else ""
fil_name = f"SchdSchm{algo_choice}{scenario_suffix}{seed_suffix}_Pvol.csv"

# Write the data to CSV
with open(fil_name, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Episodes", "KPI0_PVol"])  # Writing the header
    for idx, value in enumerate(arr, 1):
        writer.writerow([idx, value])