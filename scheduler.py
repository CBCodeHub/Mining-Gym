'''
Contains definitions for Rule-based or classical scheduling algorithms.
'''
import random
import time
from collections import deque
import gymnasium as gym
from gymnasium.spaces import Discrete
import json

class DefaultScheduler:

    def random_sel(self, shvl):
        """
        Choose randomly
        """
        print('Seeking random')
        #random.seed((time.time()))
        svl = random.choice(shvl)
        print(svl)
        return svl
    
    
    def fixed(self, num_trucks, num_shovels, rsd, fxctr, truck_id = 0):
        """
        Schedule tasks in a fixed order. Tasks are scheduled in the order they were added.
        """
        #random.seed(rsd)
        allocation_file = 'alloc.json'

        if fxctr == 0:
            """ Create allocation schedule oncein the start"""
            # Create list of truck IDs as integers
            trucks = [i + 1 for i in range(num_trucks)]
            shovels = [i + 1 for i in range(num_shovels)]  # assuming shovels are numbered 1 to num_shovels

            # Allocate shovels to trucks using round-robin allocation
            allocation = {}
            for i, truck in enumerate(trucks):
                shovel = shovels[i % num_shovels]
                allocation[truck] = shovel

            # Save allocation to a JSON file, replacing old one if it exists
            with open(allocation_file, 'w') as f:
                json.dump(allocation, f)
        
        else:
            # Load existing allocation from JSON file
            if os.path.exists(allocation_file):
                with open(allocation_file, 'r') as f:
                    allocation = json.load(f)
            else:
                raise FileNotFoundError("Allocation file not found. Set fxctr to 0 to generate a new allocation.")

        # Return the allocated shovel for the given truck_id
        allocated_shovel = allocation.get(str(truck_id), None)  # truck_id is a string in JSON
        if allocated_shovel is not None:
            return allocated_shovel
        else:
            return None

    def shortest_queue(self, shovels):
        """ Schedule the task from the queue with the shortest waiting time."""
        best_shovel = min(shovels, key=lambda shovel: len(shovel.requesters()))
        print(best_shovel)
        #time.sleep(10)
        return best_shovel