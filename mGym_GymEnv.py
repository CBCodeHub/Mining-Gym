'''
OpenAI gym compatible environment setting thats wraps the DES Mining site simulator
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

    def __init__(self, render_mode="console"):
        super(Minegym, self).__init__()

        self.render_mode = render_mode

        # Load configuration values
        cfg_samplr = ConfigSampler('config_extend.txt')  # Load from configuration file.** NO seed needed since no distribution sample
        #cfg_samplr = ConfigSampler('config_extend.txt', time_scale=5.0)
        self.NumTrucks = cfg_samplr.get_sampled_value('TR')
        self.NumShovels = cfg_samplr.get_sampled_value('SH')
        self.id_counter = 0  # Initialize ID counter
        self.tender_mode = render_mode

        # Define the file path
        self.file_path = 'envDes_shrd.csv'

        
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
            print(f"{self.file_path} has been deleted.")

        # Define action and observation spaces
        self.action_space = Discrete(self.NumShovels, start=0)
        self.observation_space = Dict({
            "ShovelID": MultiBinary(self.NumShovels * 3),  # Shovel IDs, each represented by 3 bits
            "Queue_length": Box(low=0, high=float('inf'), shape=(self.NumShovels,), dtype=np.float32),  # Queue length for each shovel, dtype = float32
            "SH_Status": MultiBinary(self.NumShovels),  # Binary status for each shovel (0 or 1), dtype will default to np.uint8
            "TruckID": MultiBinary(self.NumTrucks * 5),  # Truck IDs, each represented by 5 bits, dtype will default to np.uint8
            "Trips_complete": Box(low=0, high=float('inf'), shape=(self.NumTrucks,), dtype=np.float32),  # Number of trips completed by each truck, dtype = float32
            "TR_Status": MultiBinary(self.NumTrucks * 3),  # Status for each truck, dtype will default to np.uint8
        })
        self.des_process = None  # Store the DES process
        self.done = False  # Initialize done flag

    #--------------------------------csv Init and write----------------------------------------------------------------#
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

    def start_DES(self, fsim):
        """Start DES as a parallel process."""
        try:
            # If no process is running or it's finished, start a new one
            if self.des_process is None or not self.des_process.is_alive():
                #update json with current cnf_seed
                time.sleep(0.1)
                from mGym_DesEnv import runDes as des_main  # Import the main function from Code BB
            
                # Ensure fsim is not None or improperly initialized
                if fsim is None:
                    raise ValueError("fsim is not initialized.")
            
                # If a previous process is running, wait for it to finish with a timeout
                if self.des_process is not None:
                    print("Waiting for previous DES process to finish...")
                    self.des_process.join(timeout=60)  # Wait up to 60 seconds

                # If the process is still alive after the timeout, force termination
                if self.des_process is not None and self.des_process.is_alive():
                    print("Previous DES process timed out. Terminating it.")
                    self.des_process.terminate()  # Forcefully terminate the process if stuck
                    self.des_process.join(timeout=60)  # Wait for the process to terminate gracefully

                # Start a new DES process
                print(f"Starting a new DES process with fsim: {fsim}")
                self.des_process = multiprocessing.Process(target=des_main, args=(fsim,))
                self.des_process.start()
                print("DES process started.")

            else:
                print("DES process is already running, waiting for it to finish.")
    
        except Exception as e:
            print(f"Error starting DES process: {e}")
            traceback.print_exc()  # Print stack trace for debugging
            self.cleanup_resources()  # Ensure cleanup if an error occurs


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
        # Convert dictionary values to correct numpy types based on observation space definitions
        observation["ShovelID"] = np.array(observation["ShovelID"], dtype=np.int8)  # << Cast to uint8
        observation["Queue_length"] = np.array(observation["Queue_length"], dtype=np.float32)  # << Cast to float32
        observation["SH_Status"] = np.array(observation["SH_Status"], dtype=np.int8)  # << Cast to uint8
        observation["TruckID"] = np.array(observation["TruckID"], dtype=np.int8)  # << Cast to uint8
        observation["Trips_complete"] = np.array(observation["Trips_complete"], dtype=np.float32)  # << Cast to float32
        observation["TR_Status"] = np.array(observation["TR_Status"], dtype=np.int8)  # << Cast to uint8
        return observation



    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)


        # Initialize the mine state (initial observation)
        self.mine_state = {
            "ShovelID": np.zeros(self.NumShovels * 3, dtype=np.int8),  # Shovel IDs as 3-bit binary values
            "Queue_length": np.zeros(self.NumShovels, dtype=np.float32),  # Queue length as float for each shovel
            "SH_Status": np.ones(self.NumShovels, dtype=np.int8),  # Binary status for each shovel (initialized to 1)
            "TruckID": np.zeros(self.NumTrucks * 5, dtype=np.int8),  # Truck IDs as 5-bit binary values
            "Trips_complete": np.zeros(self.NumTrucks, dtype=np.float32),  # Number of trips completed by each truck as float
            "TR_Status": np.ones(self.NumTrucks * 3, dtype=np.int8),  # Truck status represented by 3 bits (initialized to 1)
        }
        self.info = None
        self.terminated = False
        self.done = False  # Reset done flag

        self.initialize_csv()  # Ensure CSV is initialized
        if self.render_mode == "human":
            self.start_DES(fsim=True) #, cnfg_rand_seed = config_seed)  # Start DES process
        elif self.render_mode == "console":
            self.start_DES(fsim=False)#, cnfg_rand_seed = config_seed)  
        else:
            raise ValueError(f"Invalid render_mode: {self.render_mode}. Please choose either 'human' or 'console'.")

        return self.mine_state, {}



    def step(self, action):
        if self.done:
            print("Environment is done. No further steps will be taken.")
            #return None, None, self.done, None, {}
            return None, 0.0, True, True, {}

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

    

