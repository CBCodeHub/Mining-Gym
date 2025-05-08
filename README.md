# Mining-Gym
Mining-Gym is an open-source, configurable benchmarking environment for optimizing truck dispatch scheduling in open-pit mining using Reinforcement Learning (RL).
The paper with all the details can be found here https://doi.org/10.48550/arXiv.2503.19195

## Repository Files

- `mgym_GymRun.py`: Main script for training and playing RL-based scheduling agents.
- `mgym_DefSchdRun.py`: Script to run classical/rule-based schedulers.
- `mgym_GymEnv.py`: OpenAI Gym-compatible environment setting that wraps the DES Mining site simulator.
- `mgym_DesEnv.py`: Script for the DES-based Mining Site Simulator.
- `scheduler.py`: Contains definitions for rule-based scheduling algorithms.
- `config_extend.txt`: Configuration file where you can customize simulation settings.
- `environment.yml`: Conda environment setup file to ensure reproducibility.
  
## Setup Instructions

1. **Create and activate the virtual environment:**  
conda env create -f environment.yml  

2. **To train a new RL policy network, run:**  
python mgym_GymRun.py train --num_episodes 10  
--num_episodes: Sets the number of training episodes.  
The RL algorithm is selected internally in the mgym_GymRun.py code.

3. **To run a classical scheduler (e.g., random scheduling), use:**  
python mgym_DefSchdRun.py --num_episodes 10 --algo_choice 1  
--num_episodes: Number of episodes to simulate.  
--algo_choice: Selects the scheduler algorithm as defined in scheduler.py.  
Example: 1 stands for random scheduling.

4. **To play using a pretrained model, run:**  
python mgym_GymRun.py play --num_episodes 5 --model_path <path_to_saved_model.zip>  
Replace <path_to_saved_model.zip> with the actual path to your saved model.

5. **To change configuration data:**  
You can modify the simulation settings by editing the config.extend.txt file. This allows you to adjust parameters such as environment details, scheduler settings, and other simulation-related options.


