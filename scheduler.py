import random
import time
from collections import deque
import gymnasium as gym
from gymnasium.spaces import Discrete
import json
import os  # ADD THIS - missing import!

class DefaultScheduler:
    def random_sel(self, shvl):
        """Choose randomly"""
        print('Seeking random')
        svl = random.choice(shvl)
        print(svl)
        return svl
    
    def fixed(self, truck_id, num_trucks, shovels):
        """
        Fixed allocation scheduler using round-robin.
        - First call: Generates LUT for ALL trucks (round-robin allocation)
        - Subsequent calls: Looks up truck_id and returns assigned shovel
        """
        allocation_file = 'alloc.json'
        num_shovels = len(shovels)
        
        # Check if allocation LUT exists
        if not os.path.exists(allocation_file):
            # First call - Generate complete LUT for all trucks
            allocation = {}
            for truck_num in range(1, num_trucks + 1):
                shovel_idx = (truck_num - 1) % num_shovels
                allocation[str(truck_num)] = shovel_idx  # Store index (0-based)
            
            # Save LUT to file
            with open(allocation_file, 'w') as f:
                json.dump(allocation, f)
            print(f"Fixed scheduler: Created allocation LUT for {num_trucks} trucks and {num_shovels} shovels")
        else:
            # Subsequent calls - Load existing LUT
            with open(allocation_file, 'r') as f:
                allocation = json.load(f)
        
        # Look up shovel index for this specific truck
        shovel_idx = allocation.get(str(truck_id), 0)
        
        # Return the actual shovel OBJECT (not the index)
        allocated_shovel = shovels[shovel_idx]
        print(f"Truck {truck_id} assigned to {allocated_shovel.name()}")
        return allocated_shovel
    
    def shortest_queue(self, shovels):
        """
        Return shovel with shortest TOTAL queue (requesters + claimers).
        Uses random tie-breaking to prevent herding.
        """
        # Include both waiting trucks AND currently loading trucks
        def total_load(shovel):
            return len(shovel.requesters()) + len(shovel.claimers())
            
        # Find the minimum total load across all shovels
        min_load = min(total_load(s) for s in shovels)
        
        # Select all shovels that match the minimum load (for tie-breaking)
        best_shovels = [s for s in shovels if total_load(s) == min_load]
        
        # Randomly choose one of the best shovels
        selected = random.choice(best_shovels)
        
        # Print diagnostic logging
        print(f"Shortest queue selected: {selected.name()} "
              f"(load={total_load(selected)}, "
              f"requesters={len(selected.requesters())}, "
              f"claimers={len(selected.claimers())})")

        return selected