'''
Representing the Mining Load and Haul Cycle using Discrete Event Simulation in  Salabim (BOTTOM of 3)
'''

from pickle import FALSE
import salabim as sim
sim.yieldless(False)
import gymnasium as gym
from gymnasium.spaces import Dict, MultiBinary, Box
from multiprocessing import Process
import random
import csv
import time
import re
import json
import scheduler as sch 
import kpi_calc as kcal
from read_config import ConfigSampler
import threading
import tempfile
import shutil
import queue
from collections import deque, Counter
import numpy as np
import time
import sys
import os
import random
from datetime import datetime

# Global lock for CSV operations
csv_lock = threading.Lock()

'''
Previous version DES_comm_03
'''

update_freq = 20 #10 units of time
shift_start_time = 0
all_trk_shv_dec = None 

cfg_samp = ConfigSampler('config_extend_review.txt') #, cnfg_seed = cnfg_seed) #load from configuration file
cfg_samp.episode_count = 0
#cfg_samp = ConfigSampler('config_extend.txt', time_scale=5.0)
#time_scale = 5

# Fixed parameters loaded from config (cached, same across episodes)
Num_trucks = None 
Num_shovels = None
Num_crushers = None 
Num_dumps = None 
shift_dura = None 
targ_pvol = None 
load_per_trip = None
choice = None 

# Episodic parameters (will be updated in runDes for each episode)
epsilon = 0.35  # Default value, updated in runDes()


#file_path = 'envDes_shrd.csv'
RL_sched = True # initializing the flag
def_schdlr_choice = None


r_optimal = (1-epsilon)/epsilon #used for calculating waste trip balance

xs_init = 450  # x init value of shovel queue
xd_init = 100  # x init value of dump queue

# Define Truck state code using bytes
phase_shovel = '000'
phase_crusher = '001'
phase_dump = '010'
phase_travel_shovel_crusher = '011'
phase_travel_shovel_dump = '100'
phase_travel_crusher_shovel = '101'
phase_travel_dump_shovel = '110'
phase_broken_down = '111'

k  = 5 # Sliding window length for reward calculation
alpha = 0.5 # Decay rate for exponential weight used in sliding window
trip_times = deque(maxlen=k)
shovel_queues = deque(maxlen=k)
r_imm_d_pt = None
terminated = False
pvol = 0


# Global variables
total_trips = 0  # Initialize total_trips to 0
avg_idle_orig_time = 0
sim_exit = False
total_crush_trips = 0  # Initialize total dump trips counter



# Add this global variable to keep track of trips
truck_trip_counts = {}
truck_phases = {}  # Dictionary to track truck phases



# Deque manipulation code
def add_item(item):
    # Make the deque global to allow modifications
    global all_trk_shv_dec
    all_trk_shv_dec.append(item)
    #print(f"Updated deque: {all_trk_shv_dec}")

def diversity_score() -> float:
    """
    Calculate diversity using Shannon entropy (ORIGINAL version).
    NO recency weighting - that made it worse!
    """
    global all_trk_shv_dec, Num_shovels, Num_trucks
    
    #window_size = min(Num_trucks * 2, len(all_trk_shv_dec))
    window_size = min(30, len(all_trk_shv_dec)) 
    trk_shv_dec = deque(list(all_trk_shv_dec)[-window_size:])
    
    if len(trk_shv_dec) < Num_shovels:
        return 1.0
    
    # Simple Shannon entropy over full window
    counts = Counter(trk_shv_dec)
    total = len(trk_shv_dec)
    entropy = 0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p + 1e-10)
    
    max_entropy = np.log2(Num_shovels)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    return normalized_entropy


def calculate_shovel_imbalance(shovels):
    global all_trk_shv_dec
    # Count the frequency of each shovel choice
    shovel_counts = Counter(all_trk_shv_dec)
    
    # Ensure all shovels are counted (even those not in deque)
    for shovel in shovels:
        if shovel not in shovel_counts:
            shovel_counts[shovel] = 0
    
    # Calculate the total number of shovels chosen
    total_shovels = len(all_trk_shv_dec)
    
    # Calculate the imbalance (difference between max and min shovel counts)
    if total_shovels == 0:
        return 0.0  # No actions in the deque
    
    # Get the max and min shovel counts
    max_count = max(shovel_counts.values())
    min_count = min(shovel_counts.values())
    
    # The imbalance score is the difference between the max and min counts
    imbalance = max_count - min_count
    
    # ADD THESE 2 LINES:
    zero_usage_count = sum(1 for count in shovel_counts.values() if count == 0)
    starvation_penalty = zero_usage_count * (total_shovels / Num_shovels)
    
    return imbalance + starvation_penalty


def calculate_streak_penalty() -> float:
    """
    Direct penalty for consecutive same-shovel decisions.
    Returns value in [0, 1] where higher = worse streaking.
    """
    global all_trk_shv_dec
    
    if len(all_trk_shv_dec) < 15:
        return 0.0
    
    # Look at last 15 decisions
    recent = list(all_trk_shv_dec)[-15:]
    
    # Count how many consecutive same choices
    same_count = sum(1 for i in range(1, len(recent)) if recent[i] == recent[i-1])
    streak_ratio = same_count / (len(recent) - 1)
    
    return streak_ratio

def scheduler_assign(choice, truck_id=None):
    '''
    Reads user choice of scheduler and queries it
    '''
    def_scheduler = sch.DefaultScheduler() #instance of default scheduler class
   
    if choice == 1:
        # Random Scheduler
        scheduled_equip = def_scheduler.random_sel(shovels)
    elif choice == 2:
        # Fixed Scheduler
        scheduled_equip = def_scheduler.fixed(truck_id,Num_trucks, shovels)
    elif choice == 3:
        # Shortest Queue
        scheduled_equip = def_scheduler.shortest_queue(shovels)
    else:
        raise ValueError("Choice must be 1, 2 or 3")
     
    # Return the updated values of all variables
    return scheduled_equip

def print_episode_summary():
    """Print brief episode summary for debugging"""
    try:
        print(f"Episode {cfg_samp.episode_count}: epsilon={cfg_samp.get_sampled_value('Eps'):.3f}")
        
        # Show if we have shovel classes (indicates heterogeneous setup)
        try:
            classes = cfg_samp.get_sampled_value('shovel_performance_class')
            clusters = cfg_samp.get_sampled_value('shovel_location_cluster')
            print(f"  Shovel classes: {classes}")
            print(f"  Location clusters: {clusters}")
        except KeyError:
            print("  Using homogeneous shovel configuration")
        
        # Infrastructure-based parameters
        try:
            sz1c = cfg_samp.get_sampled_value('SZ1C')
            sz1d = cfg_samp.get_sampled_value('SZ1D')
            sz2c = cfg_samp.get_sampled_value('SZ2C')
            sz2d = cfg_samp.get_sampled_value('SZ2D')
            sz3c = cfg_samp.get_sampled_value('SZ3C')
            sz3d = cfg_samp.get_sampled_value('SZ3D')
            print(f"  Infrastructure (SZ): Zone1-Crush: {sz1c:.2f}, Zone1-Dump: {sz1d:.2f}")
            print(f"    Zone2-Crush: {sz2c:.2f}, Zone2-Dump: {sz2d:.2f}")
            print(f"    Zone3-Crush: {sz3c:.2f}, Zone3-Dump: {sz3d:.2f}")
        except KeyError:
            pass # Silent fail if these aren't in the config

        # Equipment-specific operational parameters
        try:
            trdm = cfg_samp.get_sampled_value('TRDM')
            trcr = cfg_samp.get_sampled_value('TRCR')
            print(f"  Equipment Dump Times: Truck-Dump: {trdm:.2f}, Truck-Crush: {trcr:.2f}")
        except KeyError:
            pass

        # Base reliability parameters
        try:
            rsh = cfg_samp.get_sampled_value('RSH')
            rtr = cfg_samp.get_sampled_value('RTR')
            rcr = cfg_samp.get_sampled_value('RCR')
            rds = cfg_samp.get_sampled_value('RDS')
            print(f"  Base MTTR: Shovel: {rsh:.2f}, Truck: {rtr:.2f}, Crusher: {rcr:.2f}, Dump: {rds:.2f}")
        except KeyError:
            pass
            
        # Cost parameters
        try:
            known_cost = cfg_samp.get_sampled_value('known_cost')
            estimated_cost = cfg_samp.get_sampled_value('estimated_cost')
            print(f"  Cost Parameters: Known: {known_cost:.2f}, Estimated: {estimated_cost:.2f}")
        except KeyError:
            pass

        # Equipment performance variations
        try:
            trl_c1 = cfg_samp.get_sampled_value('TRL_C1')
            trl_c2 = cfg_samp.get_sampled_value('TRL_C2')
            trl_c3 = cfg_samp.get_sampled_value('TRL_C3')
            print(f"  Loading Times: Class1: {trl_c1:.2f}, Class2: {trl_c2:.2f}, Class3: {trl_c3:.2f}")
        except KeyError:
            pass

        # Daily operational conditions
        try:
            fw = cfg_samp.get_sampled_value('FW')
            fo = cfg_samp.get_sampled_value('FO')
            fe = cfg_samp.get_sampled_value('FE')
            te = cfg_samp.get_sampled_value('TE')
            tl = cfg_samp.get_sampled_value('TL')
            print(f"  Fuel Consumptions: Waste: {fw:.2f}, Ore: {fo:.2f}, Empty: {fe:.2f}")
            print(f"  Truck Speeds: Empty: {te:.2f}, Loaded: {tl:.2f}")
        except KeyError:
            pass
            
        # Equipment health status
        try:
            fsh = cfg_samp.get_sampled_value('FSH')
            ftr = cfg_samp.get_sampled_value('FTR')
            fcr = cfg_samp.get_sampled_value('FCR')
            fds = cfg_samp.get_sampled_value('FDS')
            print(f"  Equipment MTBF: Shovel: {fsh:.2f}, Truck: {ftr:.2f}, Crusher: {fcr:.2f}, Dump: {fds:.2f}")
        except KeyError:
            pass
            
    except KeyError as e:
        print(f"Episode summary error: {e}")

#------------CSV Updating -----------------------------------------------------------------------#
#------------Write back Immediate and Final reward and corresponding Observed state vector-------#

def update_csv_action(seq_id, action):
    """ Read action from csv row and set READ flag """
    global RL_sched, file_path

    if RL_sched == False: #return if not doing RL scheduling
        return 
  
    # First read the entire file
    with open(file_path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = reader.fieldnames

    # Find and update the row with matching seq_id and action
    for row in rows:
        if row['Seq. no'] == seq_id and row['Action'] == action:
            row['Read'] = 'True'

    rows = [{ki: vi for ki, vi in row.items() if ki in fieldnames} for row in rows]

    # Write all rows back with a single write operation - which avoids truncation
    with open(file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()  # Write the header
        writer.writerows(rows)  # Write all rows
        file.flush()  # Ensure data is written to disk

    print(f"Code DES: Updated 'Read' flag in CSV for seq_id {seq_id} with action {action}.")
    time.sleep(0.1)  # Small delay to let other processes access the file



def update_empty_obs_rew_row(observ, r_imm_d_pt, info, max_retries=10, retry_delay=1):
    """Update row with Observation and IMMediate Reward """ 
    global RL_sched, sim_exit, file_path

    if RL_sched == False or sim_exit: #return if not doing RL scheduling or simulation is exiting
        return 

    retry_count = 0
    while retry_count < max_retries:
        try:
            # Read all rows at once
            with open(file_path, mode='r', newline='') as file:
                reader = csv.DictReader(file)
                rows = list(reader)
                fieldnames = reader.fieldnames

            # Find the first row where Observation and Reward are empty
            updated = False
            for row in rows:
                if row['Observation'] == '' and row['Reward'] == '':
                    row['Observation'] = json.dumps(observ)
                    row['Reward'] = str(r_imm_d_pt)
                    row['Info'] = json.dumps(info)
                    row['Terminated'] = 'FALSE'
                    updated = True
                    break
            
            if updated:
                # Write all data back in a single operation - prevents truncation
                with open(file_path, mode='w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                    file.flush()

                print("\n *Code DES: Updated row with immediate reward")
                return
            else:
                print("\n Code DES: No row found with empty Observation and Reward columns. Retrying...")
                retry_count += 1
                time.sleep(retry_delay)
                
        except Exception as e:
            print(f"Error updating CSV: {e}")
            retry_count += 1
            time.sleep(retry_delay)

    print("Maximum retries reached. No update made.")
    sys.exit()
    


def final_step_update(observ, r_epi, info, max_retries=10, retry_delay=1):
    """Update with Observation and TERminal Reward"""
    global RL_sched, file_path

    if RL_sched == False: #return if not doing RL scheduling
        return 

    retry_count = 0
    while retry_count < max_retries:
        try:
            # Read all rows at once
            with open(file_path, mode='r', newline='') as file:
                reader = csv.DictReader(file)
                rows = list(reader)
                fieldnames = reader.fieldnames

            # Find the first row where Observation and Reward are empty
            updated = False
            for row in rows:
                if row['Observation'] == '' and row['Reward'] == '':
                    row['Observation'] = json.dumps(observ)
                    row['Reward'] = str(r_epi)
                    row['Info'] = json.dumps(info)
                    row['Terminated'] = 'TRUE'
                    updated = True
                    break
            
            if updated:
                # Write all data back in a single operation - prevents truncation
                with open(file_path, mode='w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                    file.flush()

                print("\n *Code DES: Updated row with end of episode reward")
                return
            else:
                print("\n Code DES: No row found with empty Observation and Reward columns. Retrying...")
                retry_count += 1
                time.sleep(retry_delay)
                
        except Exception as e:
            print(f"Error updating CSV: {e}")
            retry_count += 1
            time.sleep(retry_delay)

    print("Maximum retries reached. No update made.")

#-----------end_of_csv_writing---------------------------------------------------#


def get_shovel_from_integer(shovel_list, index):
    """
    Given a list of shovel objects and an integer, return the corresponding shovel object.
    :param shovel_list: List of shovel objects
    :param index: Integer index corresponding to a shovel in the list
    :return: Corresponding shovel object or None if index is out of bounds
    """
    if 0 <= index < len(shovel_list):
        return shovel_list[index]
    else:
        print("\n Shovel number out of range ")
        return None


def create_observation(num_shovels, num_trucks, shovel_data, active_truck_data, fleet_summary):
    '''
    Takes dictionaries of shovel_data, active_truck_data, and fleet_summary
    Returns formatted observation vector for Gym with fleet context
    
    Args:
        num_shovels: Number of shovels in the mine
        num_trucks: Total number of trucks in the fleet
        shovel_data: List of tuples [(queue_len, status), ...] for each shovel
        active_truck_data: Dict with single truck's data {truck_name: {'trip_count': X, 'phase': 'XXX'}}
        fleet_summary: Dict with {'avg_trips': float, 'recent_decisions': deque, 'diversity_score': float}
    
    Returns:
        observation: Dict with all observation components as lists
    '''
    
    # --- SHOVEL STATE ---
    shovel_id = np.zeros(num_shovels * 4, dtype=int)
    queue_length = np.zeros(num_shovels, dtype=float)
    sh_status = np.zeros(num_shovels, dtype=int)
    
    # Populate shovel data with bounded normalization
    for i, (queue_len, status) in enumerate(shovel_data):
        # FIX: Bounded queue normalization (handles extreme congestion)
        max_queue = num_trucks * 2
        queue_length[i] = round(min(queue_len / max_queue, 1.0), 4)
        sh_status[i] = status
        
        # Binary encoding of shovel ID (4 bits supports up to 16 shovels)
        bin_str = format(i + 1, '04b')
        shovel_id[i * 4:i * 4 + 4] = list(map(int, bin_str))

    # --- ACTIVE TRUCK STATE ---
    truck_id_active = np.zeros(1 * 6, dtype=int) 
    trips_complete_active = np.zeros(1, dtype=float)
    tr_status_active = np.zeros(1 * 3, dtype=int)

    # Extract the single active truck's data
    truck_key = list(active_truck_data.keys())[0]
    truck_info = active_truck_data[truck_key]
    
    # Parse truck information
    trip_count = truck_info['trip_count']
    phase = truck_info['phase']
    truck_index = int(truck_key.split('.')[1]) - 1
    
    # Binary encoding of truck ID (6 bits supports up to 64 trucks)
    binary_index = f'{truck_index + 1:06b}'
    truck_id_active[0:6] = list(map(int, binary_index))
    
    # Normalize trip count
    trips_complete_active[0] = round(trip_count / 500, 4)
    
    # Convert phase string to binary list
    phase_list = [int(char) for char in phase]
    tr_status_active[0:3] = phase_list

    # --- FLEET CONTEXT (NEW: Essential for coordination) ---
    
    # 1. Fleet average trip progress (scalar)
    fleet_avg_trips = np.zeros(1, dtype=float)
    fleet_avg_trips[0] = round(fleet_summary['avg_trips'] / 500, 4)
    
    # 2. Recent shovel usage distribution (vector showing fleet behavior)
    recent_shovel_usage = np.zeros(num_shovels, dtype=float)
    if len(fleet_summary['recent_decisions']) >= 5:
        # Look at last 10 decisions to see distribution
        recent_window = list(fleet_summary['recent_decisions'])[-10:]
        recent_counts = Counter(recent_window)
        
        for shovel_name, count in recent_counts.items():
            # Extract shovel index from name like "Shovel_3"
            idx = int(shovel_name.split('_')[1])
            # Normalize by window size to get proportion
            recent_shovel_usage[idx] = count / len(recent_window)
    
    # 3. Fleet diversity score (scalar indicating balance)
    fleet_diversity = np.zeros(1, dtype=float)
    fleet_diversity[0] = round(fleet_summary['diversity_score'], 4)
    
    # --- CREATE OBSERVATION DICTIONARY ---
    observation = {
        # Shovel information
        "ShovelID": shovel_id.tolist(),                      # num_shovels * 4 binary
        "Queue_length": queue_length.tolist(),                # num_shovels floats
        "SH_Status": sh_status.tolist(),                      # num_shovels binary
        
        # Active truck information
        "TruckID_Active": truck_id_active.tolist(),           # 6 binary
        "Trips_complete_Active": trips_complete_active.tolist(), # 1 float
        "TR_Status_Active": tr_status_active.tolist(),        # 3 binary
        
        # Fleet context (NEW)
        "Fleet_Avg_Trips": fleet_avg_trips.tolist(),          # 1 float
        "Recent_Shovel_Usage": recent_shovel_usage.tolist(),  # num_shovels floats
        "Fleet_Diversity": fleet_diversity.tolist()           # 1 float
    }
    
    return observation


class RewardCalculator:
    def __init__(self, k, alpha=0.5):
        """
        Initialize the RewardCalculator with a sliding window of length k.

        Parameters:
        k (int): Length of the sliding window to consider for averaging.
        alpha (float): Exponential weighting factor. Controls the rate of increase of the weights.
        """
        self.k = k
        self.alpha = alpha
        self.lamda = 0.1
        self.trip_times = trip_times #deque(maxlen=k)
        self.shovel_queues = shovel_queues #deque(maxlen=k)
        self.TT_Avg_min = 10
        self.TT_Avg_max = shift_dura
        self.Q_Avg_d_min = 0.00001 #very small positive value
        self.Q_Avg_d_max = 2 * shift_dura

    def min_max_normalize(self, current_value, min_value, max_value):
        """
        Normalize a value using Min-Max normalization.
        """
        if max_value == min_value:
            raise ValueError("max_value and min_value cannot be the same")
    
        normalized_value = (current_value - min_value) / (max_value - min_value)
        return normalized_value

    def update(self, tau_d, Q_SH_d):
        """ Update the sliding window with the current decision point data """

        self.trip_times.append(tau_d)
        self.shovel_queues.append(Q_SH_d)
    
    def compute_weighted_average(self, data):
        """ Compute the weighted (exponential) average for the sliding window data """
        n = len(data)
        if n == 0:
            return 0
        
        # Calculate exponential weights
        weights = np.exp(self.alpha * np.arange(1, n + 1))
        
        # Normalize the weights so they sum to 1
        normalized_weights = weights / np.sum(weights)
        
        # Compute the weighted average
        weighted_avg = np.dot(normalized_weights, data)
        
        return weighted_avg

    
    def compute_r_imm_d(self):
        """ Compute the immediate reward at the current decision point """
        if len(self.trip_times) == 0 or len(self.shovel_queues) == 0:
            return 0

        TT_Avg = self.compute_weighted_average(self.trip_times)
        Q_Avg_d = self.compute_weighted_average(self.shovel_queues)
        TT_Avg_norm =  self.min_max_normalize(TT_Avg, self.TT_Avg_min, self.TT_Avg_max)
        Q_Avg_d_norm =  self.min_max_normalize(Q_Avg_d, self.Q_Avg_d_min, self.Q_Avg_d_max)

        # Four-component reward with explicit anti-streak mechanism
        efficiency_penalty = -0.25 * TT_Avg_norm   # Increased weight (Credit Assignment) 0.15
        queue_penalty = -0.15 * Q_Avg_d_norm       # Reduced weight (Covered by efficiency)
        diversity_penalty = -0.30 * (1 - diversity_score())  #0.40
        streak_penalty = -0.30 * calculate_streak_penalty()  # Reduced weight
        
        r_imm_d = efficiency_penalty + queue_penalty + diversity_penalty + streak_penalty

        if len(all_trk_shv_dec) % 20 == 0:
            episode_num = cfg_samp.episode_count if hasattr(cfg_samp, 'episode_count') else '?'
            print(f"[Episode {episode_num}] Step {len(all_trk_shv_dec)}: "
                  f"Eff: {-efficiency_penalty:.3f}, Queue: {-queue_penalty:.3f}, "
                  f"Div: {diversity_score():.3f}, Reward: {r_imm_d:.4f}")


        return {
            'r_imm_d': r_imm_d,
            'trip_times': list(self.trip_times),
            'shovel_queues': list(self.shovel_queues)
        }


class ShovelWaitTimeTracker:
    def __init__(self, num_shovels):
        self.num_shovels = num_shovels
        self.shovel_waiting_times = {f"Shovel_{i}": 0 for i in range(num_shovels)}
        self.shovel_request_counts = {f"Shovel_{i}": 0 for i in range(num_shovels)}
        self.shovel_truck_waiting_times = {f"Shovel_{i}": {} for i in range(num_shovels)}  # Track individual truck waiting times

    def add_truck_to_queue(self, shovel_name, arrival_time, truck_id):
        """
        Called when a truck joins the shovel queue.
        Increments the request count for the shovel.
        """
        self.shovel_request_counts[shovel_name] += 1
        self.shovel_truck_waiting_times[shovel_name][truck_id] = arrival_time

    def remove_truck_from_queue(self, shovel_name, departure_time, truck_id):
        """
        Called when a truck leaves the shovel queue.
        Decrements the request count for the shovel and removes the last max waiting time of the truck.
        """
        if truck_id in self.shovel_truck_waiting_times[shovel_name]:
            arrival_time = self.shovel_truck_waiting_times[shovel_name].pop(truck_id)
            waiting_time = departure_time - arrival_time
            self.shovel_waiting_times[shovel_name] += waiting_time
            if self.shovel_request_counts[shovel_name] > 0:
                self.shovel_request_counts[shovel_name] -= 1

    def get_average_waiting_times(self):
        """
        Returns the average waiting time for each shovel.
        """
        average_times = {}
        for shovel, total_waiting_time in self.shovel_waiting_times.items():
            request_count = self.shovel_request_counts[shovel]
            average_times[shovel] = total_waiting_time / request_count if request_count > 0 else 0
        return average_times

# Initialize the tracker
#shovel_wait_time_tracker = ShovelWaitTimeTracker(Num_shovels)




class Truck(sim.Component):
    scheduler_assigned = False
    trucks_failed = 0  # Class variable to track number of trucks that have experienced breakdown
    trucks_to_fail = None  # Class variable to store the IDs of trucks that will fail
   
    def setup(self):
        global RL_sched, def_schdlr_choice, truck_breakdown_manager, Num_trucks
        global truck_last_trip_times
        global truck_trip_counts  
        global truck_phases       

        self.truck_id = ''.join(filter(str.isdigit, self.name()))  # Extract truck ID

        self.trip_count = 0
        self.truck_id = ''.join(filter(str.isdigit, self.name()))  # Extract truck ID
        #truck_trip_counts[self.name()] = self.trip_count
        #truck_last_trip_times[self.truck_id] = 0  # Initialize the last trip time for the truck

        self.breakdown_display = None  # Initialize display reference as None

        # Initialize trucks_to_fail if not already done
        if Truck.trucks_to_fail is None:
            num_trucks_to_fail = min(int(cfg_samp.get_sampled_value('TTF')), Num_trucks)
            # Randomly select which trucks will experience breakdowns
            Truck.trucks_to_fail = random.sample(range(1, Num_trucks + 1), num_trucks_to_fail)
        
        # Set initial breakdown time based on whether this truck is scheduled to fail
        if int(self.truck_id) in Truck.trucks_to_fail:
            self.next_breakdown_time = cfg_samp.get_sampled_value('TIB')  # Each truck gets its own sampled TIB
        else:
            self.next_breakdown_time = float('inf')  # Trucks not scheduled to fail get infinity


        # Initial breakdown time based on TIB (Time to Initial Breakdown)
        #self.next_breakdown_time = cfg_samp.get_sampled_value('TIB')
        self.has_failed = False  # Track if this truck has had its first breakdown
        #self.time_to_repair = cfg_samp.get_sampled_value('RSH')
        self.time_to_repair = cfg_samp.get_sampled_value('RTR')
        self.phase = phase_shovel
        self.last_phase = None
        self.shovel_name = None  # Track the shovel this truck is assigned to
        self.truck_id = ''.join(filter(str.isdigit, self.name()))  # Extract truck ID
        truck_trip_counts[self.name()] = self.trip_count # Add the truck to the global tracking dictionary
        truck_last_trip_times[self.truck_id] = 0  # Initialize the last trip time for the truck
        
        # Instantiate RewardCalculator with a window size of 5 and alpha of 0.5
        self.reward_calculator = RewardCalculator(k, alpha)

    def handle_breakdown(self):

        if int(self.truck_id) not in Truck.trucks_to_fail:
            return False

        # Check if we haven't reached the total failure target
        max_failures = len(Truck.trucks_to_fail)
        if Truck.trucks_failed >= max_failures:
            return False

        if not self.has_failed:
            if env.now() >= self.next_breakdown_time:
                self.has_failed = True
                Truck.trucks_failed += 1
                return True
            return False
            


    def update_phase(self, new_phase): 
        global truck_phases
        self.phase = new_phase
        truck_phases[self.name()] = self.phase  # Update truck phase dictionary
        #print(f"Truck {self.name()} phase updated to {self.phase}")  # Debug print

        # Add breakdown display when entering breakdown phase
        if new_phase == phase_broken_down:
            # Calculate even spacing using actual screen dimensions
            min_x = 400
            max_x = 600
            #spacing = (max_x - min_x) / (Num_trucks - 1) if Num_trucks > 1 else 0
            spacing=10
            x_pos = min_x + (int(self.truck_id) - 1) * spacing
        
            # Show broken truck image with ID
            self.breakdown_display = sim.AnimateImage(
                "dump_truck_broken.png", 
                width=70,
                x=x_pos,
                y=50,
                text=str(self.truck_id),
                text_anchor="c",
                fontsize=20,
                textcolor="yellow"
            )
        # Remove breakdown display when leaving breakdown phase
        elif self.breakdown_display is not None:
            self.breakdown_display.remove()
            self.breakdown_display = None


    def process(self):
        global total_trips  # Declare as global to modify the global variable
        global total_crush_trips
        global r_imm_d_pt
        first_trip = True  # Flag to indicate the first round
        processed_items = set()  # Track processed truck IDs
        seq_id=None
        action = None
                        
        while True:
            
            #seed = int(time.time() * 1000) % 4294967296
            #self.env.random_seed(seed)
            #trk_load = cfg_samp.get_sampled_value('TRL') 
            
            while True:
                #selected_shovel = random.choice(shovels)
                curr_truck_id = ''.join(filter(str.isdigit, self.name()))
                elapsed_time = env.now() - shift_start_time
                time_left = shift_dura - elapsed_time

                # Decide on the Shovel selection
                # Truck uses default method for first scheduling round (Shortest Queue length)
                # First trip - Classical mode
                if (first_trip == True and RL_sched == False):
                    #print(f"\n Non-RL scheduling used: algo_choice={def_schdlr_choice} \n")
                    selected_shovel = scheduler_assign(def_schdlr_choice, truck_id=int(curr_truck_id))
                    first_trip = False
                    add_item(selected_shovel.name())
    
                # First trip - RL mode    
                elif (first_trip == True and RL_sched == True):
                    #print("\n Default scheduling used for RL first trip \n ")
                    selected_shovel = scheduler_assign(choice, truck_id=int(curr_truck_id))
                    first_trip = False
                    add_item(selected_shovel.name())
                    
                # RL Scheduling chosen
                elif (first_trip == False and RL_sched == True):
                    # Here the observation and reward is to be updated to the csv
                    #---immediate reward calculation
                    # Last trip times of each truck in the fleet
                    #tau_d = sum(truck_last_trip_times.values()) / len(truck_last_trip_times) #tau_d, averaged over all trucks
                    tau_d = truck_last_trip_times[curr_truck_id]

                    average_waiting_times = shovel_wait_time_tracker.get_average_waiting_times() # Average waiting times at individual shovels at 'current event'
                    Q_SH_d  = sum(average_waiting_times.values()) / len(average_waiting_times) #Q_SH_d, averaged over all the shovels
                    self.reward_calculator.update(tau_d, Q_SH_d) # Update the 'K event sliding window' with the current event data
                    r_dict = self.reward_calculator.compute_r_imm_d()  # Compute the immediate reward at the current decision point
                    r_imm_d_pt= r_dict['r_imm_d']  #Create immediate reward

                    # *** OBSERVATION MISMATCH FIX ***
                    # Only pass the data for the current truck (self) to create_observation
                    # print_trip_counts(ppflag=0) returns ALL truck data. We need to filter it.
                    all_truck_data = print_trip_counts(ppflag=0)
                    active_truck_data = {self.name(): all_truck_data[self.name()]} # Filter to only the active truck

                    all_trips = list(truck_trip_counts.values())
                    fleet_summary = {
                        'avg_trips': sum(all_trips) / len(all_trips) if all_trips else 0,
                        'recent_decisions': all_trk_shv_dec,  # The global deque
                        'diversity_score': diversity_score()
                    }

                    #observ = create_observation(Num_shovels, Num_trucks, print_resource_status(pflag=0)['Shovels'], print_trip_counts(ppflag=0))
                    observ = create_observation(Num_shovels, Num_trucks, print_resource_status(pflag=0)['Shovels'], active_truck_data,fleet_summary)


                    info = {'truck_id': curr_truck_id}

                    global sim_exit
                    if sim_exit:
                        break
                    else:
                        # Update CSV with truck ID
                        update_empty_obs_rew_row(observ, r_imm_d_pt, info)


                    print("\n Querying RL Policy \n ")
                    with open(file_path, mode='r') as file:
                        reader = csv.reader(file)
                        header = next(reader) #skip header row
                        rows = list(reader)
                        # Find the first row with a matching action but no truck ID
                        for row in rows:
                            seq_id = row[0] #Get sequence ID
                            action = row[1] #Get action
                            read_flag = row[2]  # Read column

                            if action and (not read_flag or read_flag.lower() == 'false'):  # Check if Action exists and Read flag is empty or False
                                selected_shovel = get_shovel_from_integer(shovels, int(action))
                                add_item(selected_shovel.name())
                                print(f"\n\nCode DES: Received Action {action}.\n\n")
                                row[2] = 'True'
        
                                # Assuming update_csv_action() function handles CSV update
                                update_csv_action(seq_id,action)
                                processed_items.add((seq_id, action))
                                time.sleep(0.5)  # Sleep for 1 second to allow code AA to check the file
                                #break  # Process only the first matching row

                # Non-RL Scheduling chosen
                #elif (first_trip is False and RL_sched is False):
                elif (RL_sched == False):
                    #print(f'Scheduler algo Choice" {def_schdlr_choice}')
                    selected_shovel = scheduler_assign(def_schdlr_choice, truck_id=int(curr_truck_id))  
                    #print(f"[VERIFY] Truck {curr_truck_id} → {selected_shovel.name()} at time {env.now():.2f}")
                    add_item(selected_shovel.name())

                               
                #print('Selected SHOVEL:'+str(selected_shovel))
                self.shovel_name = selected_shovel.name()  # Set the shovel name
                shovel_wait_time_tracker.add_truck_to_queue(self.shovel_name, env.now(), self.truck_id)

                yield self.request((selected_shovel,1,4))
                start_time = env.now()

                # Get loading time based on selected shovel's performance class
                try:
                    shovel_id = int(selected_shovel.name().split('_')[1])
                    shovel_classes = cfg_samp.get_sampled_value('shovel_performance_class')
                    performance_class = shovel_classes[shovel_id]
                    trk_load = cfg_samp.get_sampled_value(f'TRL_C{performance_class}')
                except (KeyError, IndexError) as e:
                    print(f"ERROR: Could not get loading time for shovel {selected_shovel.name()}: {e}")


                yield self.hold(trk_load)

                if self.isbumped():
                    trk_load -= env.now() - start_time
                    continue
                break

            shovel_wait_time_tracker.remove_truck_from_queue(self.shovel_name, env.now(), self.truck_id)
            self.release()
     
            # Determine whether to go to a crusher or dump based on epsilon-greedy strategy
            if random.random() < epsilon:  # Epsilon = 1 for Crusher only
                #[Go To CRUSHER]
                self.update_phase(phase_travel_shovel_crusher) 
                selected_crusher = random.choice(crushers)  
                #norm_time_cr = cfg_samp.get_sampled_value('STC')  # Travel time to crusher
                try:
                    shovel_id = int(selected_shovel.name().split('_')[1])
                    location_clusters = cfg_samp.get_sampled_value('shovel_location_cluster')
                    location_cluster = location_clusters[shovel_id]
                    norm_time_cr = cfg_samp.get_sampled_value(f'SZ{location_cluster}C')  # Zone to Crusher
                except (KeyError, IndexError) as e:
                    print(f"ERROR: Could not get crusher travel time for shovel {selected_shovel.name()}: {e}")

                vmax_rnd_cr = 300 / (norm_time_cr)
                traj_c01 = sim.TrajectoryCircle(radius=200, x_center=750, y_center=600, angle0=230, angle1=360, v0=0, vmax=vmax_rnd_cr)

                # Check for breakdown (on way to Crusher)
                if self.handle_breakdown():
                    #print(f"{self.name()} breaks down at {env.now():.2f}") 
                    self.passivate() 
                    curr_phase = self.phase
                    self.update_phase(phase_broken_down) #Update truck state
                    breakdown_start_time = env.now() 
                    yield self.hold(self.time_to_repair)  # Repair time based on MTTR 
                    #print(f"{self.name()} repaired at {env.now():.2f}") 

                    # Keep track of repair time and update trajetory information
                    # for animation purpose
                    repair_end_time = env.now()
                    repair_duration = repair_end_time - breakdown_start_time  # Calculate repair duration
                    adjusted_time_cr = norm_time_cr + repair_duration  # Adjust travel time to include repair time
                    vmax_rnd_cr = 300 / adjusted_time_cr  # Recalculate vmax_rnd_cr with adjusted travel time
                    traj_c01 = sim.TrajectoryCircle(radius=200, x_center=750, y_center=600, angle0=230, angle1=360, v0=0, vmax=vmax_rnd_cr)  # Update trajectory with new vmax_rnd_cr

                    self.activate()  
                    self.update_phase(curr_phase) 
                    # Schedule next breakdown using FTR
                    self.next_breakdown_time = env.now() + cfg_samp.get_sampled_value('FTR')

                # Animation and travel for crusher
                txt = str(self.name())
                num = txt.split('.')[-1]
                self.dump_truck_cr = sim.AnimateImage("dump_truck_01.png", width=70, x=traj_c01.x, y=traj_c01.y, angle=traj_c01.angle, text=str(num), text_anchor="c", fontsize=20, textcolor="yellow")
                yield self.hold(norm_time_cr)
                self.dump_truck_cr.remove()

                # Unloading in Crusher phase
                yield self.request(selected_crusher)
                crush_dump = cfg_samp.get_sampled_value('TRCR')
                yield self.hold(crush_dump)
                self.release()

                # Increment counter right after crusher operation completes
                global total_crush_trips
                total_crush_trips += 1

                self.update_phase(phase_travel_crusher_shovel) 
                
                #Return to Shovel animate
                #norm_time_rev_cr = cfg_samp.get_sampled_value('CTS') #Crusher to shovel travel time
                try:
                    shovel_id = int(selected_shovel.name().split('_')[1])
                    location_clusters = cfg_samp.get_sampled_value('shovel_location_cluster')
                    location_cluster = location_clusters[shovel_id]
                    norm_time_rev_cr = cfg_samp.get_sampled_value(f'SZ{location_cluster}C')  # Zone to Crusher (return)
                except (KeyError, IndexError) as e:
                    print(f"ERROR: Could not get return travel time from crusher: {e}")
                vmax_rnd_rev_cr = 330/ (norm_time_rev_cr)
                traj_c02 = sim.TrajectoryCircle(radius=230, x_center=750, y_center=600, angle0=330, angle1=230, v0 = 0, vmax  = vmax_rnd_rev_cr)
                self.dump_truck_cr_2 = sim.AnimateImage("dump_truck_02.png", width=70, x=traj_c02.x, y=traj_c02.y,angle=traj_c02.angle,text=str(num), text_anchor = "c", fontsize= 20, textcolor = "red")
                yield self.hold(norm_time_rev_cr)
                self.dump_truck_cr_2.remove()

                # Update last trip time for truck
                truck_last_trip_times[self.truck_id] = norm_time_cr + crush_dump + norm_time_rev_cr


            else:
                #[Go To DUMP SITE]

                self.update_phase(phase_travel_shovel_dump)  

                selected_dump = random.choice(dumps)
                # Define trajectory for traveling to a dump
                #norm_time = cfg_samp.get_sampled_value('STD')  # Travel to dump time
                try:
                    shovel_id = int(selected_shovel.name().split('_')[1])
                    location_clusters = cfg_samp.get_sampled_value('shovel_location_cluster')
                    location_cluster = location_clusters[shovel_id]
                    norm_time = cfg_samp.get_sampled_value(f'SZ{location_cluster}D')  # Zone to Dump
                except (KeyError, IndexError) as e:
                    print(f"ERROR: Could not get dump travel time for shovel {selected_shovel.name()}: {e}")
                vmax_rnd = 830 / norm_time
                traj_03 = sim.TrajectoryCircle(radius=230, x_center=300, y_center=470, angle0=360, angle1=150, vmax=vmax_rnd)

                #Check for breakdown ( on way to Dumping Site)
                if self.handle_breakdown():
                    #print(f"{self.name()} breaks down at {env.now():.2f}") 
                    self.passivate() 
                    curr_phase = self.phase
                    self.update_phase(phase_broken_down)  #Update truck state
                    breakdown_start_time = env.now() 
                    yield self.hold(self.time_to_repair)  # Repair time based on MTTR 
                    #print(f"{self.name()} repaired at {env.now():.2f}") 

                    # Keep track of repair time and update trajetory information
                    # for animation purpose
                    repair_end_time = env.now()
                    repair_duration = repair_end_time - breakdown_start_time  # Calculate repair duration
                    adjusted_time = norm_time + repair_duration  # Adjust travel time to include repair time
                    vmax_rnd = 830 / adjusted_time  # Recalculate vmax_rnd with adjusted travel time
                    traj_03 = sim.TrajectoryCircle(radius=230, x_center=300, y_center=470, angle0=360, angle1=150, vmax=vmax_rnd)  # Update trajectory with new vmax_rnd

                    self.activate() 
                    self.update_phase(curr_phase) 
                    # Schedule next breakdown using FTR
                    self.next_breakdown_time = env.now() + cfg_samp.get_sampled_value('FTR')

                # Animation and travel to dump
                txt = str(self.name())
                num = txt.split('.')[-1]
                self.dump_truck_ds = sim.AnimateImage("dump_truck_01.png", width=70, x=traj_03.x, y=traj_03.y, angle=traj_03.angle, text=str(num), text_anchor="c", fontsize=20, textcolor="yellow")   
                yield self.hold(norm_time)
                self.dump_truck_ds.remove()

                # Dumping phase
                yield self.request(selected_dump)
                dump_time = cfg_samp.get_sampled_value('TRDM')
                yield self.hold(dump_time)
                self.release()
                self.update_phase(phase_travel_dump_shovel)  # << Added
                
                #Return to Shovel animate
                #norm_time_rev = max(1,sim.Normal(40,10).sample())
                #norm_time_rev = cfg_samp.get_sampled_value('DTS')
                try:
                    shovel_id = int(selected_shovel.name().split('_')[1])
                    location_clusters = cfg_samp.get_sampled_value('shovel_location_cluster')
                    location_cluster = location_clusters[shovel_id]
                    norm_time_rev = cfg_samp.get_sampled_value(f'SZ{location_cluster}D')  # Zone to Dump (return)
                except (KeyError, IndexError) as e:
                    print(f"ERROR: Could not get return travel time from dump: {e}")
                vmax_rnd_rev = 330/ (norm_time_rev)
                traj_04 = sim.TrajectoryCircle(radius=180, x_center=300, y_center=470, angle0=150, angle1=360, v0 = 0, vmax  = vmax_rnd_rev)
                self.dump_truck_2 = sim.AnimateImage("dump_truck_02.png", width=70, x=traj_04.x, y=traj_04.y,angle=traj_04.angle,text=str(num), text_anchor = "c", fontsize= 20, textcolor = "red")
                yield self.hold(norm_time_rev)
                self.dump_truck_2.remove()
               
                # Update last trip time for truck
                truck_last_trip_times[self.truck_id] = norm_time + dump_time + norm_time_rev


            # Increment trip count
            self.trip_count += 1
            truck_trip_counts[self.name()] = self.trip_count

            # Update total trips
            global total_trips
            total_trips = sum(truck_trip_counts.values())



#---Shovel Breakdown event handling -----

class BreakdownManager(sim.Component):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self):
        self.shovel_breakdowns = []
    
    def _select_shovels_to_fail(self):
        """Centralized selection of which shovels should fail with their unique initial breakdown times"""
        num_shovels = cfg_samp.get_sampled_value('SH')
        num_shovels_to_fail = min(int(cfg_samp.get_sampled_value('STF')), num_shovels)
        
        # Get all shovel IDs
        shovel_ids = [int(s.name().split('_')[1]) for s in shovels]
        
        # Randomly select shovels to fail
        shovels_to_fail = random.sample(shovel_ids, num_shovels_to_fail)
        
        # Sample initial breakdown times for each failing shovel with timing separation
        breakdown_schedule = {}
        for s in shovels:
            shovel_id = int(s.name().split('_')[1])
            if shovel_id in shovels_to_fail:
                # Get unique initial breakdown time for this shovel with jitter
                base_time = cfg_samp.get_sampled_value('SIB')
                jitter = random.uniform(0, 50)  # Add random jitter between 0-10 time units
                initial_breakdown = base_time + jitter
                
                breakdown_schedule[s.name()] = {
                    'should_fail': True,
                    'initial_breakdown': initial_breakdown
                }
            else:
                breakdown_schedule[s.name()] = {
                    'should_fail': False,
                    'initial_breakdown': None
                }
        
        print(f"\nSelected {num_shovels_to_fail} shovels to fail with schedule:")
        for name, schedule in breakdown_schedule.items():
            if schedule['should_fail']:
                print(f"{name}: Initial breakdown at {schedule['initial_breakdown']}")
        
        return breakdown_schedule

    def process(self):
        try:
            # Centrally determine which shovels should fail and their schedules
            breakdown_schedule = self._select_shovels_to_fail()
            
            # Create individual breakdown components for each shovel with timing separation
            for idx, shovel in enumerate(shovels):
                schedule = breakdown_schedule[shovel.name()]
                
                # Add small delay between component creation to ensure temporal separation
                if idx > 0:
                    yield self.hold(random.uniform(0.1, 0.5))
                
                breakdown = IndividualShovelBreakdown(
                    shovel=shovel,
                    shovel_animations=shovel_animations,
                    should_fail=schedule['should_fail'],
                    initial_breakdown_time=schedule['initial_breakdown']
                )
                self.shovel_breakdowns.append(breakdown)
            
            yield self.passivate()  # Manager becomes passive after creating breakdown components
            
        except Exception as e:
            print(f"Error in BreakdownManager process: {str(e)}")
            raise


class IndividualShovelBreakdown(sim.Component):
    """Individual component to handle each shovel's breakdowns"""
    def setup(self, shovel, shovel_animations, should_fail, initial_breakdown_time=None):
        self.shovel = shovel
        self.shovel_animations = shovel_animations
        self.has_failed = False
        self.is_broken = False
        self.should_fail = should_fail
        
        # Use provided initial breakdown time if shovel should fail
        if self.should_fail:
            self.next_breakdown_time = initial_breakdown_time
            print(f"{self.shovel.name()}: Initial breakdown scheduled at {self.next_breakdown_time}")
        else:
            self.next_breakdown_time = float('inf')

    def process(self):
        try:
            while True:
                if not self.has_failed and self.should_fail:
                    # Wait until breakdown time
                    time_until_breakdown = max(0, self.next_breakdown_time - self.env.now())
                    if time_until_breakdown > 0:
                        yield self.hold(time_until_breakdown)

                    # Handle breakdown
                    yield self.request((self.shovel, 1, 1))
                    print(f"\nShovel {self.shovel.name()} breaking down at time {self.env.now()}")
                    
                    # Calculate repair time with small random variation
                    base_repair_time = cfg_samp.get_sampled_value('RSH')
                    repair_time = base_repair_time + random.uniform(0, 5)  # Add up to 5 time units of variation
                    
                    self.has_failed = True
                    self.is_broken = True
                    
                    # Update animation
                    self.shovel_animations[self.shovel].image = "shovel_broken.png"
                    print(f"Repair time for {self.shovel.name()}: {repair_time:.2f}")
                    
                    # Wait for repair
                    yield self.hold(repair_time)
                    
                    # Repair complete
                    self.shovel_animations[self.shovel].image = "shovel_active.png"
                    self.release()
                    self.is_broken = False
                    
                    # Schedule next breakdown using FSH with jitter
                    base_next_time = cfg_samp.get_sampled_value('FSH')
                    jitter = random.uniform(0, 10)
                    self.next_breakdown_time = self.env.now() + base_next_time + jitter
                    print(f"Next breakdown for {self.shovel.name()} scheduled at {self.next_breakdown_time}")
                
                else:
                    # Wait until next shift
                    time_until_next_shift = shift_dura - (self.env.now() % shift_dura)
                    yield self.hold(time_until_next_shift)
                    
                    # Reset for next shift
                    self.has_failed = False
                    if self.should_fail:
                        base_time = cfg_samp.get_sampled_value('SIB')
                        jitter = random.uniform(0, 10)
                        self.next_breakdown_time = self.env.now() + base_time + jitter
                        print(f"New shift: {self.shovel.name()} scheduled for breakdown at {self.next_breakdown_time}")
                        
        except Exception as e:
            print(f"Error in IndividualShovelBreakdown process for {self.shovel.name()}: {str(e)}")
            raise


#--Trace printing methods...
def print_trip_counts(ppflag=1):
    truck_info = {} #Dictionary to store truck information
    #print("\nTruck ID | Total Trips | Current Phase")
    #print("--------------------------------------")
    for truck_id, trip_count in truck_trip_counts.items():
        phase = truck_phases.get(truck_id, '000')  # Get the phase for the truck, default to "Unknown" if not found
        truck_info[truck_id] = {'trip_count': trip_count, 'phase': phase}

        if ppflag == 1:
            pass
            #print(f"{truck_id:8} | {trip_count:11} | {phase}")
        else:
            pass
    total_trips = sum(truck_trip_counts.values())
    #print(f"\n **Total number of trips made by all trucks: {total_trips}")
    return truck_info



# Event to print truck trip counts every 100 units of time
class PrintTripCountsEvent(sim.Component):
    def process(self):
        while True:
            yield self.hold(5)  # Wait for 100 units of time
            print_trip_counts()

class TrackIdleTime(sim.Component):
    def process(self):
        while True:
            yield self.hold(5)  # Check every time unit
            current_time = env.now()
            for shovel in shovels:
                if not shovel.claimers():
                    shovel_idle_times[shovel.name()] += current_time - shovel_last_check[shovel.name()]
                shovel_last_check[shovel.name()] = current_time

class PrintShovelIdleTimes(sim.Component):
    def process(self):
        while True:
            yield self.hold(1)  # Wait for 100 units of time
            #print("\nShovel ID | Idle Time")
            #print("----------------------")
            for shovel in shovels:
                idle_time = shovel_idle_times[shovel.name()]
                #print(f"{shovel.name():8} | {idle_time:10.2f}")

            global avg_idle_orig_time
            avg_idle_orig_time = sum(shovel_idle_times.values())/len(shovels)
            #print(f"Total shovel idle time: {avg_idle_orig_time}")

class PrintClaimersStatusEvent(sim.Component):
    def process(self):
        while True:
            yield self.hold(10)  # Wait for 100 units of time
            print_resource_status()

def print_resource_status(pflag=1):
    if pflag==1:
        pass
        #print("\nResource Status:")
        #print("-----------------")
    else:
        pass
    
    resource_status = {}

    def is_resource_broken_down(resource):
        # Check if the claimer has a 'name' attribute
        srn = str(resource.claimers().head())
        return 'breakdownevent' in srn

    # Print Shovel claimers and status
    #print("\nShovels:")
    resource_status['Shovels'] = []

    for shovel in shovels:
        claimers = list(shovel.claimers())
        num_claimers = len(claimers)
        num_requesters = len(list(shovel.requesters()))
        total = num_claimers + num_requesters
        status = 1 if not is_resource_broken_down(shovel) else 0
        resource_status['Shovels'].append((total, status))
        if pflag==1:
            pass
            #print(f"{shovel.name()}: ({total}, {status})")
        else:
            pass

    
    # Print Crusher claimers and status
    #print("\nCrushers:")
    resource_status['Crushers'] = []

    for crusher in crushers:
        claimers = list(crusher.claimers())
        num_claimers = len(claimers)
        num_requesters = len(list(crusher.requesters()))
        total = num_claimers + num_requesters
        status = 1 if not is_resource_broken_down(crusher) else 0
        resource_status['Crushers'].append((total, status))
        if pflag==1:
            pass
            #print(f"{crusher.name()}: ({total}, {status})")
        else:
            pass
    
    # Print Dump claimers and status
    #print("\nDumps:")
    resource_status['Dumps'] = []

    for dump in dumps:
        claimers = list(dump.claimers())
        num_claimers = len(claimers)
        num_requesters = len(list(dump.requesters()))
        total = num_claimers + num_requesters
        status = 1 if not is_resource_broken_down(dump) else 0
        resource_status['Dumps'].append((total, status))
        if pflag==1:
            pass
            #print(f"{dump.name()}: ({total}, {status})")
        else:
            pass

    return resource_status



class KPICalculator(sim.Component):
    #def setup(self):
    def setup(self, scenario_name=None, seed=None):
        self.HIGH_FREQ = 2
        self.SHIFT_DURATION = shift_dura
        self.HOUR = 60  # Assuming time units are in minutes
        self.last_high_freq_update = 0
        self.last_shift_update = 0
        self.last_hour_update = 0
        
        # Build filename with scenario and seed
        scenario_part = f"scenario_{scenario_name}_" if scenario_name else ""
        seed_part = f"seed_{seed}_" if seed is not None else ""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = f'kpi_log_{scenario_part}{seed_part}{timestamp}.csv'
        self.file_exists = os.path.exists(self.csv_filename)
        self.shift_counter = 0
        
        # Initialize hourly tracking
        self.last_hour_trips = 0
        self.last_hour_volume = 0
        self.trips_per_hour = 0
        self.volume_per_hour = 0
        
        # Initialize equipment state tracking with more detailed states
        self.shovel_states = {shovel.name(): {'state': 'operational', 'last_state_change': 0} 
                            for shovel in shovels}
        self.truck_states = {truck.name(): {'state': 'operational', 'last_state_change': 0} 
                           for truck in truck}
        
        # Initialize queue monitors
        self.shovel_monitors = {
            shovel.name(): {'queue': shovel.requesters()} for shovel in shovels
        }
        self.crusher_monitors = {
            crusher.name(): {'queue': crusher.requesters()} for crusher in crushers
        }
        self.dump_monitors = {
            dump.name(): {'queue': dump.requesters()} for dump in dumps
        }

        self.headers = [
            'Timestamp',
            'Shift_Number',
            'Shovel_Queue_Lengths',
            'Crusher_Queue_Lengths', 
            'Dump_Queue_Lengths',
            'Changed_Shovel',
            'New_State',
            'Changed_Truck',
            'New_State_Truck',
            'Trips_Per_Hour',
            'Production_Volume_Per_Hour',
            'Total_Production_Volume',
            'Cost_Per_Ton',
            'Total_Fuel_Consumption'
        ]
        
        if RL_sched:
            self.headers.insert(5, 'Immediate_Reward')
            self.headers.append('Shift_Reward')
        
        if not self.file_exists:
            self.initialize_csv()

    def calculate_hourly_metrics(self):
        """Calculate metrics that are tracked on an hourly basis"""
        global total_trips, load_per_trip, total_crush_trips
        
        current_hour = self.env.now() // self.HOUR
        if current_hour > self.last_hour_update // self.HOUR:
            # Calculate trips in the last hour
            current_trips = total_trips
            self.trips_per_hour = current_trips - self.last_hour_trips
            self.last_hour_trips = current_trips
            
            # Calculate production volume in the last hour
            current_volume = total_crush_trips * load_per_trip
            self.volume_per_hour = current_volume - self.last_hour_volume
            self.last_hour_volume = current_volume
            
            self.last_hour_update = self.env.now()
        
        return {
            'Trips_Per_Hour': self.trips_per_hour,
            'Production_Volume_Per_Hour': self.volume_per_hour
        }

    def is_shovel_broken(self, shovel):
        """
        Enhanced check for shovel breakdown state
        """
        for claimer in shovel.claimers():
            claimer_str = str(claimer).lower()
            if 'breakdownevent' in claimer_str or 'individualshovelbreakdown' in claimer_str:
                return True
        return False

    def check_equipment_states(self):
        """
        Enhanced equipment state monitoring with immediate CSV updates for state changes
        """
        current_time = self.env.now()
        changed_equipment = []
        
        # Check shovels
        for shovel in shovels:
            current_state = 'breakdown' if self.is_shovel_broken(shovel) else 'operational'
            prev_state = self.shovel_states[shovel.name()]['state']
            
            if current_state != prev_state:
                self.shovel_states[shovel.name()]['state'] = current_state
                self.shovel_states[shovel.name()]['last_state_change'] = current_time
                
                metrics = {
                    'Timestamp': current_time,
                    'Shift_Number': self.shift_counter,
                    'Shovel_Queue_Lengths': None,
                    'Crusher_Queue_Lengths': None,
                    'Dump_Queue_Lengths': None,
                    'Changed_Shovel': shovel.name(),
                    'New_State': current_state,
                    'Changed_Truck': None,
                    'New_State_Truck': None,
                    'Trips_Per_Hour': None,
                    'Production_Volume_Per_Hour': None,
                    'Total_Production_Volume': None,
                    'Cost_Per_Ton': None,
                    'Total_Fuel_Consumption': None
                }
                
                if RL_sched:
                    metrics['Immediate_Reward'] = None
                    metrics['Shift_Reward'] = None
                
                self.update_csv(metrics)
                changed_equipment.append(('shovel', shovel.name(), current_state))
        
        # Check trucks
        for t in truck:
            current_state = 'breakdown' if t.phase == phase_broken_down else 'operational'
            prev_state = self.truck_states[t.name()]['state']
            
            if current_state != prev_state:
                self.truck_states[t.name()]['state'] = current_state
                self.truck_states[t.name()]['last_state_change'] = current_time
                
                metrics = {
                    'Timestamp': current_time,
                    'Shift_Number': self.shift_counter,
                    'Shovel_Queue_Lengths': None,
                    'Crusher_Queue_Lengths': None,
                    'Dump_Queue_Lengths': None,
                    'Changed_Shovel': None,
                    'New_State': None,
                    'Changed_Truck': t.name(),
                    'New_State_Truck': current_state,
                    'Trips_Per_Hour': None,
                    'Production_Volume_Per_Hour': None,
                    'Total_Production_Volume': None,
                    'Cost_Per_Ton': None,
                    'Total_Fuel_Consumption': None
                }
                
                if RL_sched:
                    metrics['Immediate_Reward'] = None
                    metrics['Shift_Reward'] = None
                
                self.update_csv(metrics)
                changed_equipment.append(('truck', t.name(), current_state))
        
        return changed_equipment

    def calculate_high_freq_metrics(self):
        """Calculate metrics that are updated at high frequency"""
        try:
            metrics = {
                'Shovel_Queue_Lengths': [m['queue'].length() for m in self.shovel_monitors.values()],
                'Crusher_Queue_Lengths': [m['queue'].length() for m in self.crusher_monitors.values()],
                'Dump_Queue_Lengths': [m['queue'].length() for m in self.dump_monitors.values()],
                'Changed_Shovel': None,
                'New_State': None,
                'Changed_Truck': None,
                'New_State_Truck': None
            }
            
            hourly_metrics = self.calculate_hourly_metrics()
            metrics.update(hourly_metrics)
            
            if RL_sched:
                metrics['Immediate_Reward'] = r_imm_d_pt if 'r_imm_d_pt' in globals() else None
                
            return metrics
        except Exception as e:
            print(f"Error calculating high frequency metrics: {e}")
            return {}

    def calculate_shift_metrics(self):
        """Calculate metrics that are tracked on a per-shift basis"""
        try:
            material_load_per_trip = cfg_samp.get_sampled_value('LO')
            
            total_production = total_crush_trips * material_load_per_trip
            total_cost = (
                cfg_samp.get_sampled_value('known_cost') + 
                cfg_samp.get_sampled_value('estimated_cost')
            )
            
            metrics = {
                'Total_Production_Volume': total_production,
                'Cost_Per_Ton': total_cost / total_production if total_production > 0 else 0,
                'Total_Fuel_Consumption': total_trips * cfg_samp.get_sampled_value('FO'),
                'Changed_Shovel': None,
                'New_State': None,
                'Changed_Truck': None,
                'New_State_Truck': None
            }
            
            hourly_metrics = self.calculate_hourly_metrics()
            metrics.update(hourly_metrics)
            
            if RL_sched:
                metrics['Shift_Reward'] = r_epi if 'r_epi' in globals() else None
                
            return metrics
        except Exception as e:
            print(f"Error calculating shift metrics: {e}")
            return {}

    def initialize_csv(self):
        """Initialize the CSV file with headers"""
        try:
            with open(self.csv_filename, 'w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=self.headers)
                writer.writeheader()
                file.flush()
            self.file_exists = True
        except IOError as e:
            print(f"Error initializing CSV: {e}")

    def update_csv(self, metrics):
        """Update the CSV file with new metrics"""
        try:
            with open(self.csv_filename, 'a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=self.headers)
                if os.path.getsize(self.csv_filename) == 0:
                    writer.writeheader()
                writer.writerow(metrics)
                file.flush()
        except IOError as e:
            print(f"Error writing to CSV: {e}")
            if not os.path.exists(self.csv_filename):
                self.initialize_csv()
                self.update_csv(metrics)

    def process(self):
        """Main process loop for the KPI Calculator"""
        while True:
            try:
                yield self.hold(1)
                current_time = env.now()
                
                # Check equipment states more frequently
                self.check_equipment_states()
                
                if current_time - self.last_high_freq_update >= self.HIGH_FREQ:
                    metrics = {
                        'Timestamp': current_time,
                        'Shift_Number': self.shift_counter,
                    }
                    
                    high_freq_metrics = self.calculate_high_freq_metrics()
                    if high_freq_metrics:
                        metrics.update(high_freq_metrics)
                        self.last_high_freq_update = current_time
                
                    if current_time - self.last_shift_update >= self.SHIFT_DURATION:
                        shift_metrics = self.calculate_shift_metrics()
                        if shift_metrics:
                            metrics.update(shift_metrics)
                            self.last_shift_update = current_time
                            self.shift_counter += 1
                    
                    self.update_csv(metrics)
                    
            except Exception as e:
                print(f"Error in KPI Calculator process: {e}")
                raise  # Re-raise the exception for debugging





def runDes(fsim=True, flag_RL_sched=True, fdef_schdlr_choice = None, episode_seed=None, scenario_overrides=None, csv_path=None, scenario_name=None, play_seed=None):

    global file_path
    global env, shovels, truck, dumps, crushers
    global shovel_idle_times, shovel_last_check, shovel_animations
    global RL_sched, def_schdlr_choice, all_trk_shv_dec, epsilon
    global cfg_samp
    global Num_trucks, Num_shovels, Num_crushers, Num_dumps
    global shift_dura, targ_pvol, load_per_trip, choice
    global total_trips, total_crush_trips
    global truck_trip_counts, truck_phases, truck_last_trip_times
    global shovel_wait_time_tracker
    global trip_times, shovel_queues
    global r_imm_d_pt, terminated, pvol, sim_exit

    if csv_path is None:
        raise ValueError("csv_path must be explicitly provided to runDes")

    file_path = csv_path

    # ========================================================================
    # STEP 1: Reset Truck class variables FIRST
    # ========================================================================
    Truck.trucks_to_fail = None
    Truck.trucks_failed = 0

    previous_episode_count = getattr(cfg_samp, 'episode_count', 0)
    # ========================================================================
    # STEP 2: ALWAYS reinitialize ConfigSampler (not just for overrides)
    # ========================================================================
    cfg_samp = ConfigSampler('config_extend_review.txt', 
                             scenario_overrides=scenario_overrides)

     # =====  DEBUG CODE =====
    if scenario_overrides:
        print(f"\n=== Scenario Overrides Active ===")
        print(f"Overrides: {scenario_overrides}")
    # ===== END DEBUG CODE =====

    # ========================================================================
    # STEP 3: Initialize fresh episodic parameters for this episode
    # ========================================================================
    cfg_samp.new_episode(episode_seed)
    # Store scenario and seed for KPI filename
    global current_scenario, current_seed
    current_scenario = scenario_name if scenario_name else "default"
    current_seed = play_seed if play_seed is not None else (episode_seed if episode_seed is not None else 0)

    # ========================================================================
    # STEP 4: Sample ALL parameters BEFORE using them
    # ========================================================================
    Num_trucks = int(cfg_samp.get_sampled_value('TR'))
    Num_shovels = int(cfg_samp.get_sampled_value('SH'))
    Num_crushers = int(cfg_samp.get_sampled_value('CR')) 
    Num_dumps = int(cfg_samp.get_sampled_value('DS'))
    shift_dura = cfg_samp.get_sampled_value('Sdur')
    targ_pvol = cfg_samp.get_sampled_value('PVol_targ')
    load_per_trip = cfg_samp.get_sampled_value('LO')
    choice = int(cfg_samp.get_sampled_value('scheduler_choice'))
    epsilon = cfg_samp.get_sampled_value('Eps')

    # ========================================================================
    # STEP 5: Reset all counters and state dictionaries
    # ========================================================================
    total_trips = 0
    total_crush_trips = 0
    truck_trip_counts = {}
    truck_phases = {}
    truck_last_trip_times = {}
    sim_exit = False           
    r_imm_d_pt = None          
    terminated = False         
    pvol = 0 

    # ========================================================================
    # STEP 6: NOW create deques and trackers with correct sizes
    # ========================================================================
    all_trk_shv_dec = deque(maxlen=Num_trucks * 2)
    shovel_wait_time_tracker = ShovelWaitTimeTracker(Num_shovels)
    
    # Reset rolling reward windows
    trip_times.clear()
    shovel_queues.clear()

    # ========================================================================
    # STEP 7: Print episode summary and continue with rest of function
    # ========================================================================
    print_episode_summary()

    if flag_RL_sched:
        print("\n" + "="*70)
        print(f"    STARTING EPISODE {cfg_samp.episode_count}")
        print("="*70 + "\n")

    shovel_animations = {}
    env = sim.Environment(trace=False, time_unit='minutes') #set simulation to work in minutes
    env.animate(False)
    env.width(1280)
    env.height(1024)
    RL_sched = flag_RL_sched  # Update the global flag
    def_schdlr_choice = fdef_schdlr_choice

    # Initialize print event
    print_event = PrintTripCountsEvent()
    print_event.activate()

    # Generate trucks as Component
    truck = [Truck() for _ in range(Num_trucks)]
    shovels = [sim.Resource(f"Shovel_{i}", capacity=1, preemptive=True) for i in range(Num_shovels)] # Create multiple shovels

    #Initialize dump site and crushers
    dumps = [sim.Resource(f'Dump{j}') for j in range(Num_dumps)]
    crushers = [sim.Resource(f'Crushers{j}') for j in range(Num_crushers)] 

    breakdown_manager = BreakdownManager()
    breakdown_manager.activate()

    # Create the KPI calculator component
    #kpi_calculator = KPICalculator(scenario_name=current_scenario, seed=current_seed)


    # Animation display setup------------------------------------------------------------------
    time_display = lambda: f"Time: {env.t():.2f}" 
    sim.AnimateImage("mine_site_1280_1024.png",x=5,y=5,width= 1020) #Wallpaper
    env.AnimateText(text=time_display, x=800, y=50, fontsize=20, textcolor = "white") #Display time
    env.background_color(("#eeffcc"))

    #Display config parameters 
    #sim.AnimateText(text=config_display(config), x=10, y=80, text_anchor='w', fontsize=12, textcolor='white' )

    # Dictionary to track idle times for each shovel
    shovel_idle_times = {shovel.name(): 0 for shovel in shovels}
    shovel_last_check = {shovel.name(): 0 for shovel in shovels}

    # Initialize idle time tracking and print events
    track_idle_time_event = TrackIdleTime()
    track_idle_time_event.activate()

    print_idle_times_event = PrintShovelIdleTimes()
    print_idle_times_event.activate()


    # Initialize and activate the new event
    print_resource_status_event = PrintClaimersStatusEvent()
    print_resource_status_event.activate()

    # Shovel Section
    sim.AnimateText(text="< SHOVEL >", x=320, y=620, fontsize=20, textcolor="yellow")
    for shovel in shovels:
        shv_txt = str(shovel.base_name())
        sv_id = int(shv_txt.split('_')[-1])
        xs_val = xs_init + sv_id * 50
        shovel_animations[shovel] = sim.AnimateImage("shovel_active.png", x=(xs_init + sv_id * 40) - 120, y=570, width=40, env=env)
        shovel.claimers().animate(x=xs_val, y=660, title=".", direction="e")
        shovel.requesters().animate(x=xs_val, y=580, title=".", direction="s")

    sim.AnimateText(text="Loading", x=340, y=550, fontsize=15, textcolor="white")
    sim.AnimateText(text="Waiting", x=340, y=490, fontsize=15, textcolor="white")

    # Add a header for breakdown status display
    sim.AnimateText(text="< BREAKDOWN STATUS >", x=640, y=900, fontsize=20, textcolor="yellow")

    # Dump Section
    sim.AnimateText(text="< DUMPS>", x=50, y=780, fontsize=20, textcolor="yellow")
    for dump in dumps:
        dp_txt = dump.base_name()  # Assuming .name() method returns the name of the dump
        match = re.search(r'\d+$', dp_txt)  # This regex finds one or more digits at the end of the string
        if match:
            dmp_id = int(match.group())  # Convert the found digits to an integer
            xd_val = xd_init + dmp_id * 50
            dump.claimers().animate(x=xd_val, y=930, title=".", direction="e")
            dump.requesters().animate(x=xd_val, y=870, title=".", direction="s")
        else:
            print("Invalid dump name format:", dp_txt)

    sim.AnimateText(text="Dumping", x=50, y=765, fontsize=15, textcolor="white")
    sim.AnimateText(text="Waiting", x=50, y=713, fontsize=15, textcolor="white")


    # Crusher Section
    xc_init = 1100  # Initial x-coordinate for crushers
    yc_init = 700  # Initial y-coordinate for crushers
    # Add a label for Crushers
    sim.AnimateText(text="< CRUSHERS >", x=800, y=660, fontsize=20, textcolor="yellow")

    for crusher in crushers:
        cp_txt = crusher.base_name()  # Assuming .name() method returns the name of the crusher
        match = re.search(r'\d+$', cp_txt)  # This regex finds one or more digits at the end of the string
        if match:
            crusher_id = int(match.group())  # Convert the found digits to an integer
            xc_val = (xc_init-90) + crusher_id * 50
            yc_val = yc_init  # y-coordinate remains the same for all crushers
            crusher.claimers().animate(x=xc_val, y=yc_val+70, title=".", direction="e")
            crusher.requesters().animate(x=xc_val, y=yc_val, title=".", direction="s")
        else:
            print("Invalid crusher name format:", cp_txt)

    # Add labels for Crusher actions
    sim.AnimateText(text="Crushing", x=xc_init-300, y=yc_init - 60, fontsize=15, textcolor="white")
    sim.AnimateText(text="Waiting", x=xc_init-300, y=yc_init - 120, fontsize=15, textcolor="white")


    print('-------**----')
    print(f"\nStarting simulation. Duration set to: {shift_dura} time units")
    env.run(till=shift_dura)
    print(f"\nSimulation ended at: {env.now()} time units")
    sim_exit = True
    
  
    #trips_cal = total_trips - total_dump_trips
    #trips_cal = np.clip (trips_cal, 0, 1000, out=None)
    pvol = total_crush_trips * load_per_trip #Calculate total production volume for the shift 
    wvol = max(total_trips - total_crush_trips, 0) * load_per_trip
  
    # Calculate final diversity score (e.g., as the diversity of the entire deque history)
    final_diversity_score = diversity_score()

    if RL_sched  == True:
        prod_ratio = min(1.0, pvol/targ_pvol)  # Calculate production ratio (scaled between 0-1)
        bonus = 0.0

        final_diversity = diversity_score()


        if prod_ratio >= 0.95 and final_diversity > 0.65:
            bonus = 0.5
        elif prod_ratio >= 0.90 and final_diversity > 0.50:
            bonus = 0.2
        else:
            bonus = 0.0

        # Balance the components with appropriate weights
        r_epi = 0.4 * prod_ratio - 0.6*( 1- final_diversity) + bonus 

        print('\n **** ')
        print("pvol: {}, r_epi: {}, total_trips: {}, load_per_trip: {}, r_imm_d_pt: {}, targ_pvol: {}".format(pvol, r_epi, total_trips, load_per_trip, r_imm_d_pt, targ_pvol))

        # Get all truck data
        all_truck_data = print_trip_counts(ppflag=0)
    
        # Filter to just one truck (pick any - doesn't matter which)
        if all_truck_data:
            # Use the first available truck
            first_truck_key = list(all_truck_data.keys())[0]
            terminal_truck_data = {first_truck_key: all_truck_data[first_truck_key]}
        else:
            # Fallback (shouldn't happen - there should always be trucks)
            terminal_truck_data = {'Truck.1': {'trip_count': 0, 'phase': '000'}}

        all_trips = list(truck_trip_counts.values())
        fleet_summary = {
            'avg_trips': sum(all_trips) / len(all_trips) if all_trips else 0,
            'recent_decisions': all_trk_shv_dec,
            'diversity_score': final_diversity_score
        }
    
        # Now pass only 1 truck (same structure as during episode)
        observ = create_observation(Num_shovels, Num_trucks,
                                   print_resource_status(pflag=0)['Shovels'],
                                   terminal_truck_data, fleet_summary)  # ← Now passes 1 truck
    
        info = {'PVOL': pvol, 'DivScore': final_diversity_score}
        final_step_update(observ, r_epi, info)
        print("Current observ: "+str(observ))
        #return pvol
        # Increment episode count for tracking
        cfg_samp.episode_count += 1
        sys.exit()
    elif RL_sched  == False:
        return pvol


   