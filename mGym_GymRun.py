'''
Main script for training and playing RL-based scheduling agents.
'''
import gymnasium as gym
import random
import time
import numpy as np
import os
import csv
import tensorboard
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.envs.registration import register
from datetime import datetime
from typing import Optional

# Will be set in main()

def register_minegym():
    """Register the Minegym environment with Gymnasium."""
    try:
        register(
            id='Minegym-v0',
            entry_point='mGym_GymEnv:Minegym',
        )
        print("Environment registered successfully!")
    except Exception as e:
        print(f"Failed to register environment: {e}")

def gen_seed(iteration, initial_seed=42, ax=1664525, cx=1013904223, mx=2**32):
    """
    Generate a seed based on the iteration using a Linear Congruential Generator (LCG).
    
    Args:
        iteration: The current iteration (episode)
        initial_seed: The starting seed
        ax, cx, mx: Constants for the Linear Congruential Generator
    
    Returns:
        The seed for the given iteration
    """
    epi_seed = initial_seed
    for tx in range(iteration):
        epi_seed = (ax * epi_seed + cx) % mx
    return epi_seed

class TrainingLoggerCallback(BaseCallback):
    """
    Custom callback for logging training progress and saving model checkpoints.
    """
    def __init__(
        self,
        model,
        save_dir,  # Add save_dir parameter
        verbose=1,
        log_file='training_log.txt',
        max_timesteps=10000000,
        max_episodes=2,
        save_interval=10
    ):
        super().__init__(verbose)
        self.model = model
        self.save_dir = save_dir  # Store save directory
        self.log_file = log_file
        self.max_timesteps = max_timesteps
        self.max_episodes = max_episodes
        self.save_interval = save_interval
        self.episode_count = 0
        
        # Initialize log file with full path
        self.log_file = os.path.join(save_dir, log_file)
        self.log_file_handle = open(self.log_file, 'a')

    def _write_log_header(self):
        """Write header to the log file."""
        header = "Timestamp,Episode,Step,Reward,Done\n"
        self.log_file_handle.write(header)
        self.log_file_handle.flush()

    def _on_step(self) -> bool:
        """
        Method called after each step of training.
        Returns False if training should be stopped.
        """
        # Print current timestep and episode info for each step
        if self.verbose > 0:
            print(f"Logging Step: {self.num_timesteps}, Done: {self.locals['dones']}, Episodes: {self.episode_count}")

        # Check if episode ended
        if any(self.locals['dones']):
            self.episode_count += 1
            
            # Save model at specified frequency
            if self.episode_count % self.save_interval == 0:
                model_path = os.path.join(
                    self.save_dir,  # Use instance save_dir
                    f"ppo_minegym_checkpoint_{self.episode_count}.zip"
                )
                try:
                    self.model.save(model_path)
                    if self.verbose > 0:
                        print(f"Checkpoint saved at episode {self.episode_count} to {model_path}")
                except Exception as e:
                    print(f"Error saving model checkpoint: {e}")

            # Log episode info
            self._log_step_info()

        # Check stopping conditions
        if self._should_stop_training():
            return False

        return True

    def _log_step_info(self):
        """Log training information to file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = (
            f"{timestamp},{self.episode_count},{self.num_timesteps},"
            f"{self.locals['rewards'][0]},{self.locals['dones'][0]}\n"
        )
        self.log_file_handle.write(log_entry)
        self.log_file_handle.flush()

    def _should_stop_training(self) -> bool:
        """Check if training should be stopped based on conditions."""
        if self.num_timesteps >= self.max_timesteps:
            print(f"Reached maximum timesteps of {self.max_timesteps}")
            return True
        if self.episode_count >= self.max_episodes:
            print(f"Reached maximum episode count of {self.max_episodes}")
            return True
        return False

    def close(self):
        """Clean up resources."""
        if hasattr(self, 'log_file_handle'):
            self.log_file_handle.close()

def main(choice, num_episodes, model_path=None):
    """
    Main function to either train a new model or play with an existing one.
    
    Args:
        choice: 'train' or 'play'
        num_episodes: Number of episodes to train or play
    """
    # Generate unique folder name only when training
    global MODEL_SAVE_DIR
   

    if choice == 'train':
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        MODEL_SAVE_DIR = f"saved_models_{timestamp}"
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        print(f"Created model directory: {MODEL_SAVE_DIR}")
    register_minegym()
    env = gym.make("Minegym-v0", render_mode="console")

    if choice == 'train':
        # Initialize the PPO model with the same hyperparameters
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=0.0007,
            n_steps=600,
            batch_size=64,
            n_epochs=20,
            gamma=0.995,
            clip_range=0.25,
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.02,
            vf_coef=0.7,
            max_grad_norm=0.5,
            target_kl=None,
            verbose=2,
            tensorboard_log="./ppo_tensorboard/",
            device='auto',
            _init_setup_model=True
        )

        # Create callback with proper parameters
        logger_callback = TrainingLoggerCallback(
            model=model,  # Pass model directly as in original
            save_dir =  MODEL_SAVE_DIR,
            verbose=1,
            max_episodes=num_episodes,
            save_interval=100
        )

        try:
            # Train the model
            model.learn(total_timesteps=100000, callback=logger_callback, tb_log_name="run_02")
            
            # Save final model
            final_model_path = os.path.join(MODEL_SAVE_DIR, "ppo_minegym_final.zip")
            model.save(final_model_path)
            
        finally:
            logger_callback.close()
            env.close()

    elif choice == 'play':
        try:
            if model_path is None:
                print("Error: Model path must be provided for playing.")
                return
            model = PPO.load(model_path, env=env)

            # Run the model for specified number of episodes
            for episode in range(num_episodes):
                epi_seed = gen_seed(episode)
                obs, info = env.reset(seed=49)
                done = False

                while not done:
                    action, _states = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = env.step(action)

                print(f"Episode {episode + 1} finished")
                #pvol.append(info['PVOL'])

        finally:
            #print(f"Average PVOL is {np.mean(pvol)} averaged over {num_episodes} separate episodes")
            env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or play a PPO model for the Minegym environment.")
    parser.add_argument(
        'choice',
        type=str,
        choices=['train', 'play'],
        help="Choose 'train' to train a new model or 'play' to load and play an existing model."
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=2,
        help="Number of episodes to train/play. Default is 2."
    )
    parser.add_argument(
        '--model_path',
        type=str,
        help="Path to the pre-trained model for playing. Required for 'play'."
)
    
    args = parser.parse_args()
    main(args.choice, args.num_episodes, args.model_path)