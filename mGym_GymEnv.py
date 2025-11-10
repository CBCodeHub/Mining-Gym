'''
OpenAI gym compatible environment setting thats wraps the DES Mining site simulator (MIDDLE of 3)
'''

import numpy as np
import gymnasium as gym
from gymnasium import Env
from gymnasium import spaces
from gymnasium.spaces import Discrete, Dict, MultiBinary, Box
import time
import os
import json
import csv
import random
import multiprocessing
import traceback
import sys
from read_config import ConfigSampler

# Registration with Gymnasium
from gymnasium.envs.registration import register



class Minegym(Env):
    metadata = {"render_modes": ["console"]}

    def __init__(self, render_mode="console", scenario_overrides=None, csv_path=None, scenario_name=None, play_seed=None):
        super(Minegym, self).__init__()

        self.render_mode = render_mode
        self.scenario_overrides = scenario_overrides
        self.scenario_name = scenario_name
        self.play_seed = play_seed

        if csv_path is None:
            raise ValueError("csv_path must be explicitly provided when creating Minegym environment")

        self.file_path = csv_path # <-- Use the dynamic path

        # Load configuration values
        cfg_samplr = ConfigSampler('config_extend_review.txt')  # Load from configuration file.** NO seed needed since no distribution sample
        #cfg_samplr = ConfigSampler('config_extend.txt', time_scale=5.0)
        self.NumTrucks = int(cfg_samplr.get_sampled_value('TR'))
        self.NumShovels = int(cfg_samplr.get_sampled_value('SH'))
        self.id_counter = 0  # Initialize ID counter
        self.tender_mode = render_mode

        # Define the file path
        #self.file_path = 'envDes_shrd.csv'

        
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
            print(f"{self.file_path} has been deleted.")

        # Define action and observation spaces
        self.action_space = Discrete(self.NumShovels, start=0)
        self.observation_space = Dict({
            "ShovelID": MultiBinary(self.NumShovels * 4),
            "Queue_length": Box(low=0, high=float('inf'), shape=(self.NumShovels,), dtype=np.float32),
            "SH_Status": MultiBinary(self.NumShovels),

            # --- Single Truck Change ---
            "TruckID_Active": MultiBinary(1 * 6),  # 1 truck * 6 bits
            "Trips_complete_Active": Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32), # 1 truck
            "TR_Status_Active": MultiBinary(1 * 3), # 1 truck * 3 bits
            # ------

            "Fleet_Avg_Trips": Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
            "Recent_Shovel_Usage": Box(low=0, high=1.0, shape=(self.NumShovels,), dtype=np.float32),
            "Fleet_Diversity": Box(low=0, high=1.0, shape=(1,), dtype=np.float32),

        })
        self.des_process = None  # Store the DES process
        self.done = False  # Initialize done flag

    #--------------------------------csv Init and write----------------------------------------------------------------#
    def is_des_alive(self):
        """Check if DES process is alive and responsive."""
        return self.des_process is not None and self.des_process.is_alive()

    def ensure_des_running(self):
        """Ensure DES process is running, restart if needed."""
        if not self.is_des_alive():
            print("WARNING: DES process is dead. Attempting restart...")
            self.cleanup_resources()
            if self.render_mode == "human":
                self.start_DES(fsim=True)
            else:
                self.start_DES(fsim=False)
            time.sleep(2)  # Give DES time to start
            return True
        return False


    def initialize_csv(self):
        """Initialize the CSV file with headers if it doesn't already exist."""
        if not os.path.exists(self.file_path):
            with open(self.file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                # Write the headers with the required column names and order
                writer.writerow(['Seq. no', 'Action', 'Read', 'Observation', 'Reward', 'Terminated', 'Info'])
                file.flush()

    def generate_seq_id(self):
        """Generate a seq ID."""
        self.id_counter += 1
        return f"ID_{self.id_counter}"


    def write_action(self, seq_id, action):
        # Write action to CSV
        with open(self.file_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['Seq. no', 'Action', 'Read', 'Observation', 'Reward', 'Terminated', 'Info'])
            # Add the new action as a new row
            writer.writerow({
                'Seq. no': seq_id,
                'Action': action,
                'Read': 'False',
                'Observation': '',
                'Reward': '',
                'Terminated': 'FALSE',
                'Info': ''
            })
            file.flush()
        print(f"Code mGym: Action {action} written.")
        time.sleep(2)

    def cleanup_resources(self):
        """Cleanup resources if the DES process is running or after it finishes."""
        if self.des_process is not None and self.des_process.is_alive():
            print("Cleaning up previous DES process...")
            self.des_process.terminate()  # Terminate the stuck or long-running process
            self.des_process.join()  # Ensure termination is complete
        self.des_process = None  # Reset the process reference

    def start_DES(self, fsim, scenario_name=None, play_seed=None):
        """Start DES as a parallel process and wait for it to be ready."""
        try:
            if self.des_process is None or not self.des_process.is_alive():
                from mGym_DesEnv import runDes as des_main
            
                if fsim is None:
                    raise ValueError("fsim is not initialized.")
            
                if self.des_process is not None:
                    print("Waiting for previous DES process to finish...")
                    self.des_process.join(timeout=60)
                    if self.des_process.is_alive():
                        print("Previous DES process timed out. Terminating it.")
                        self.des_process.terminate()
                        self.des_process.join(timeout=60)

                print(f"Starting a new DES process with fsim: {fsim}")
                self.des_process = multiprocessing.Process(
                    target=des_main, 
                    args=(fsim,),
                    kwargs={
                        'scenario_overrides': self.scenario_overrides, 
                        'csv_path': self.file_path,
                        'scenario_name': scenario_name,
                        'play_seed': play_seed
                    }
                )
                self.des_process.start()
                print("DES process started.")
            
                self._wait_for_des_ready()
            
            else:
                print("DES process is already running, waiting for it to finish.")
    
        except Exception as e:
            print(f"Error starting DES process: {e}")
            traceback.print_exc()
            self.cleanup_resources()

    def _wait_for_des_ready(self, timeout=10):
        """Wait for DES to be ready to receive actions."""
        print("Waiting for DES to initialize...")
        start_time = time.time()
    
        while time.time() - start_time < timeout:
            # Check if CSV exists and has headers (DES has initialized)
            if os.path.exists(self.file_path):
                try:
                    with open(self.file_path, 'r') as f:
                        first_line = f.readline()
                        if 'Seq. no' in first_line:  # Headers written
                            time.sleep(2) 
                            print("✓ DES ready")
                            return True
                except:
                    pass
            time.sleep(0.5)
    
        print("WARNING: DES initialization timeout")
        return False


    def terminate_DES(self):
        """Terminate the DES process if it's running."""
        if self.des_process is not None and self.des_process.is_alive():
            self.des_process.terminate()
            self.des_process.join()  # Ensure the process is fully terminated
            print("DES process terminated.")

    def check_flag(self, flag_type, expected_value, seq_id):
        """
        Check if a specific flag (Read or Terminated) in the CSV has the expected value.
        """
        flag_column_index = {
            'Read': 2,         # Column index for 'Read' flag
            'Terminated': 5    # Column index for 'Terminated' flag
        }.get(flag_type)

        if flag_column_index is None:
            raise ValueError(f"Invalid flag type: {flag_type}. Choose 'Read' or 'Terminated'.")
        time.sleep(0.05)

        with open(self.file_path, mode='r') as file:
            reader = csv.DictReader(file)
            #headers = next(reader)  # Skip headers
            for row in reader:
                if row['Seq. no'] == seq_id:
                    print("\n ===================================")
                    print(f"Checking row with seq_id: {seq_id}")  # Debugging line
                    terminated_value = row['Terminated'].strip().upper()  # Get the 'Terminated' value
                    if terminated_value == expected_value.upper():
                        return True
                    break  # Exit after checking the current row
        return False
 

    def convert_obs_to_numpy(self, observation):
        """Convert observation from JSON to numpy arrays with correct dtypes"""
        observation["ShovelID"] = np.array(observation["ShovelID"], dtype=np.int8)
        observation["Queue_length"] = np.array(observation["Queue_length"], dtype=np.float32)
        observation["SH_Status"] = np.array(observation["SH_Status"], dtype=np.int8)

        # Active truck
        observation["TruckID_Active"] = np.array(observation["TruckID_Active"], dtype=np.int8)
        observation["Trips_complete_Active"] = np.array(observation["Trips_complete_Active"], dtype=np.float32)
        observation["TR_Status_Active"] = np.array(observation["TR_Status_Active"], dtype=np.int8)
    
        # Fleet context features
        observation["Fleet_Avg_Trips"] = np.array(observation["Fleet_Avg_Trips"], dtype=np.float32)
        observation["Recent_Shovel_Usage"] = np.array(observation["Recent_Shovel_Usage"], dtype=np.float32)
        observation["Fleet_Diversity"] = np.array(observation["Fleet_Diversity"], dtype=np.float32)

        return observation



    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.cleanup_resources() 

        if os.path.exists(self.file_path):
            os.remove(self.file_path)
            print(f"{self.file_path} has been deleted for new episode.")

        self.initialize_csv() 

        # Initialize the mine state (initial observation)
        self.mine_state = {
            "ShovelID": np.zeros(self.NumShovels * 4, dtype=np.int8),
            "Queue_length": np.zeros(self.NumShovels, dtype=np.float32),
            "SH_Status": np.ones(self.NumShovels, dtype=np.int8),
        
            # Single active truck (initialized to zeros)
            "TruckID_Active": np.zeros(1 * 6, dtype=np.int8),
            "Trips_complete_Active": np.zeros(1, dtype=np.float32),
            "TR_Status_Active": np.ones(1 * 3, dtype=np.int8),

            # NEW: Fleet context initialized to defaults
            "Fleet_Avg_Trips": np.zeros(1, dtype=np.float32),
            "Recent_Shovel_Usage": np.zeros(self.NumShovels, dtype=np.float32),
            "Fleet_Diversity": np.ones(1, dtype=np.float32),  # Start with perfect diversity
        }
        self.info = None
        self.terminated = False
        self.done = False  # Reset done flag

        self.steps_this_episode = 0

        self.initialize_csv()  # Ensure CSV is initialized
        if self.render_mode == "human":
            self.start_DES(fsim=True, scenario_name=self.scenario_name, play_seed=self.play_seed)
        elif self.render_mode == "console":
            self.start_DES(fsim=False, scenario_name=self.scenario_name, play_seed=self.play_seed)
        else:
            raise ValueError(f"Invalid render_mode: {self.render_mode}. Please choose either 'human' or 'console'.")

        return self.mine_state, {}


    def step(self, action):
        # Check 1: Episode already terminated
        if self.done:
            print("WARNING: step() called on terminated environment. Call reset() first.")
            return self.mine_state, 0.0, True, True, {"error": "step_after_done"}

        self.steps_this_episode += 1 
    
        # Check 2: DES process liveness - FIXED VERSION
        if self.des_process is None or not self.des_process.is_alive():
            # **FIXED: Only treat as error if we're mid-episode AND early**
            # If this is step 1-2 after reset, DES might still be starting
            if self.steps_this_episode <= 2:
                print(f"WARNING: DES not alive at step {self.steps_this_episode}, attempting recovery...")
                # Try to restart
                if self.render_mode == "human":
                    self.start_DES(fsim=True)
                else:
                    self.start_DES(fsim=False)
                # Give it one more chance to process this step
            elif self.steps_this_episode < 50:  # Mid-episode failure
                print(f"ERROR: DES died mid-episode at step {self.steps_this_episode}")
                self.done = True
                return self.mine_state, -100.0, True, False, {"error": "DES_premature_death"}
            else:
                # Near end - natural termination
                print(f"DES completed at step {self.steps_this_episode} - episode ending")
                self.done = True
                return self.mine_state, 0.0, True, False, {"info": "shift_complete"}
    
    # Rest of existing step() code...

        terminate = False  # Flag to signal when to terminate
        self.done = False
        truncated =False

        # Generate and write the action with a unique ID
        seq_id = self.generate_seq_id()
        self.write_action(seq_id, action)
        print(f"mGym: Action {action} with seq_id {seq_id} written to CSV.")

        #try:
        observation_filled = False
        attempt_count = 0

        time.sleep(1)  # <<<<

        while not observation_filled:
            with open(self.file_path, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    # Check if this is the correct row by matching 'Seq. no' (seq_id) and ensure 'Observation' is filled
                    if row['Seq. no'] == seq_id and row['Observation'].strip():
                        observation_filled = True
                        #observation = row['Observation']
                        observation = json.loads(row['Observation'])
                        observation = self.convert_obs_to_numpy(observation)

                        # Safely get the reward, setting a default if missing
                        reward = row.get('Reward')
                        if reward is not None:
                            reward = float(reward)  # Assuming reward is filled correctly
                        else:
                            reward = 0.0

                        info_str = row.get('Info', {})
                        try:
                            info = json.loads(info_str)  # Convert the string into a dictionary
                        except json.JSONDecodeError:
                            info = {}  # In case the Info column is not a valid JSON string, default to an empty dict
                        print(f"mGym: Observation and reward for seq_id {seq_id} retrieved.")
                        break  # Exit the for-loop when found

            if not observation_filled:
                print(f"mGym: Observation for seq_id {seq_id} is not filled. Waiting...")
                attempt_count += 1
                if attempt_count >= 10:
                    print(f"mGym: Max attempts reached ({attempt_count}). Exiting step.")
                    self.done = True
                    self.cleanup_resources()
                    return None, 0.0, True, True, {} #Return and stop episode
                time.sleep(3)  # Delay to avoid tight looping

        # Check if Terminate Flag is set to True and stop execution
        terminate = self.check_flag('Terminated', 'TRUE', seq_id)
        truncated = False

        if terminate:
            self.cleanup_resources()  
            print(f"mGym: Terminate flag detected. Stopping execution.")
            self.done= True
            #return observation, reward, True, truncated, {}
            return observation, reward, self.done, truncated, info

        # Continue the episode if termination not detected
        #return observation, reward, False, False, {}
        return observation, reward, self.done, truncated, info


    def render(self):
        # Implementation for rendering (done in DES)
        pass

    def close(self):
        self.terminate_DES()
        print("Environment Closed")

    

