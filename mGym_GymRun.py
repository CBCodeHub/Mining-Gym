import gymnasium as gym
import random
import time
import numpy as np
import os
import csv
import tensorboard
import argparse
import pandas as pd
from stable_baselines3 import PPO
from scenario_loader import load_scenario
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.envs.registration import register
from read_config import ConfigSampler
from datetime import datetime
from typing import Optional, Dict, Any, List


# Define evaluation parameters directly in the script.
DEFAULT_EVAL_EPISODES = 0
DEFAULT_EVAL_INTERVAL = 0
# -------------------------------------


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

def gen_seed(iteration, initial_seed=44, ax=1664525, cx=1013904223, mx=2**32):
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

# =========================================================================
# Helper function for logging evaluation data to a CSV file 
# =========================================================================
def log_evaluation_csv(episode_count: int, eval_data: List[Dict[str, Any]]):
    """
    Logs the evaluation data to a cumulative CSV file in the 'interm_test_data' folder.
    """
    EVAL_DIR = 'interim_test_data'
    os.makedirs(EVAL_DIR, exist_ok=True)
    
    # Add training episode column to each row
    for row in eval_data:
        row['Training_Episode'] = episode_count
    
    df = pd.DataFrame(eval_data)
    
    # ADD THIS: Calculate summary row
    summary_row = {
        'Training_Episode': episode_count,
        'Eval_Run': 'Mean',
        'Seed': '-',
        'Total_Reward': df['Total_Reward'].mean(),
        'PVOL': df['PVOL'].iloc[0],  # Keep first PVOL (all should be same)
        'DivScore': df['DivScore'].mean(),
        'Total_Steps': df['Total_Steps'].mean()
    }
    
    # ADD THIS: Append summary row to dataframe
    df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
    
    # Use fixed filename for cumulative logging
    filename = os.path.join(EVAL_DIR, 'Evaluation_Results.csv')
    
    # Append to existing file or create new with headers
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
        print(f"Evaluation data appended to {filename}")
    else:
        df.to_csv(filename, mode='w', header=True, index=False)
        print(f"Evaluation data saved to new file: {filename}")


def run_deterministic_eval(model: PPO, env_id: str, num_eval_episodes: int, scenario_overrides: Optional[dict] = None) -> Dict[str, Any]:
    """
    Runs multiple deterministic episodes, aggregates results, and returns them, 
    collecting PVOL and Diversity Score.
    """
    results = []
    
    # CRITICAL CHECK: Don't run if num_eval_episodes is 0
    if num_eval_episodes <= 0:
        return {'individual_runs': [], 'mean_pvol': 0.0, 'mean_div_score': 0.0}

    for eval_num in range(num_eval_episodes):
        print(f"\n--- Starting Deterministic Evaluation Episode {eval_num + 1}/{num_eval_episodes} ---")
        
        # Use a fixed, but distinct, seed for each evaluation run for reproducibility
        base_seed = int(time.time() * 1000)
        eval_seed = base_seed + (eval_num * 10000)

        # Create a unique, temporary file path for this single evaluation run
        eval_csv_path = "./eval_shared.csv" 
        
        # Create a temporary, single environment and pass the temporary path
        eval_env = gym.make(env_id, render_mode="console", 
                            scenario_overrides=scenario_overrides,
                            csv_path=eval_csv_path) 
        
        obs, info = eval_env.reset(seed=eval_seed)
        done = False
        cumulative_reward = 0
        final_pvol = 0.0
        final_div_score = 0.0
        step_count = 0
        
        while not done:
            # Use deterministic=True to disable exploration
            action, _ = model.predict(obs, deterministic=False)
            action_scalar = action.item() 
            
            obs, reward, done, truncated, info = eval_env.step(action_scalar)
            cumulative_reward += reward
            step_count += 1
            
            if done:
                 # Capture the final metrics from the info dictionary 
                 final_pvol = info.get('PVOL', 0.0) 
                 final_div_score = info.get('DivScore', 0.0)
                 break
        
        # Store results for this single evaluation run
        results.append({
            'Eval_Run': eval_num + 1,
            'Seed': eval_seed,
            'Total_Reward': cumulative_reward,
            'PVOL': final_pvol,
            'DivScore': final_div_score,
            'Total_Steps': step_count
        })

        print(f"--- Evaluation Run {eval_num + 1} Finished. PVOL: {final_pvol:.2f}, DivScore: {final_div_score:.4f} ---")
        eval_env.close() # Clean up the temporary environment and DES process

    # Calculate mean metrics over the evaluation episodes
    mean_pvol = np.mean([r['PVOL'] for r in results])
    mean_div_score = np.mean([r['DivScore'] for r in results])

    return {'individual_runs': results, 'mean_pvol': mean_pvol, 'mean_div_score': mean_div_score}


class TrainingLoggerCallback(BaseCallback):
    """
    Custom callback for saving model checkpoints and managing episode limits.
    """
    def __init__(
        self,
        model,
        save_dir,
        eval_interval: int,
        verbose=1,
        max_timesteps=10000000,
        max_episodes=2,
        save_interval=2,
        eval_env_id: str = 'Minegym-v0',
        scenario_overrides: Optional[dict] = None,
        num_eval_episodes: int = 0,
        training_env = None,
    ):
        super().__init__(verbose)
        self.model = model
        self.save_dir = save_dir
        self.max_timesteps = max_timesteps
        self.max_episodes = max_episodes
        self.save_interval = save_interval
        self.episode_count = 0
    
        # Evaluation related attributes (kept for structure, though unused)
        self.eval_interval = eval_interval
        self.eval_env_id = eval_env_id
        self.scenario_overrides = scenario_overrides
        self.num_eval_episodes = num_eval_episodes
        self.last_eval_episode = 0
    
    
    def _on_step(self) -> bool:
        """
        Method called after each step of training.
        Returns False if training should be stopped.
        """
        # Check if episode ended
        if any(self.locals['dones']):
            self.episode_count += 1
            
            if self.verbose > 0:
                print(f"--- Training Episode {self.episode_count} finished at Timestep {self.num_timesteps} ---")

            # Save model at specified frequency
            if self.episode_count % self.save_interval == 0:
                model_path = os.path.join(
                    self.save_dir,
                    f"ppo_minegym_checkpoint_{self.episode_count}.zip"
                )
                try:
                    self.model.save(model_path)
                    if self.verbose > 0:
                        print(f"Checkpoint saved at episode {self.episode_count} to {model_path}")
                except Exception as e:
                    print(f"Error saving model checkpoint: {e}")


        # Check stopping conditions
        if self._should_stop_training():
            return False

        return True

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
        pass

# =========================================================================
# Main function
# =========================================================================
def main(choice, num_episodes, model_path=None, scenario=None, play_seed=49):
    """
    Main function to either train a new model, resume training from a checkpoint, or play with an existing one.
    """
    global MODEL_SAVE_DIR

    # Load scenario overrides
    scenario_overrides = load_scenario(scenario)
    
    register_minegym()

    # Calculate n_steps from config
    cfg_samp = ConfigSampler('config_extend_review.txt')
    shift_duration = cfg_samp.get_sampled_value('Sdur')
    n_steps_calculated = int(shift_duration)
    print(f"Calculated n_steps: {n_steps_calculated} (from shift_duration of {shift_duration})")

    if choice == 'train':
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # If resuming from a checkpoint, use the existing model directory timestamp
        if model_path:
            import re
            # Try to extract the timestamp from the model_path directory name
            dir_match = re.search(r'saved_models_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', model_path)
            if dir_match:
                timestamp = dir_match.group(1)
                print(f"Resuming training in existing model directory: saved_models_{timestamp}")
            else:
                print("WARNING: Could not extract timestamp from model path directory. Creating a new directory.")

        MODEL_SAVE_DIR = f"saved_models_{timestamp}"
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        print(f"Using model directory: {MODEL_SAVE_DIR}")

        # Define a custom CSV path for the permanent training environment
        TRAINING_CSV_PATH = "./train_shared.csv"
        env = gym.make("Minegym-v0", render_mode="console", 
                       scenario_overrides=scenario_overrides,csv_path=TRAINING_CSV_PATH)

        # Initialize the PPO model or load from a checkpoint
        if model_path:
            print(f"Loading model from checkpoint: {model_path}")
            model = PPO.load(
                model_path, 
                env=env, 
                verbose=2, 
                tensorboard_log="./ppo_tensorboard/"
            )
            
            # Extract last episode number from filename for callback initialization
            import re
            match = re.search(r'_(\d+)\.zip$', model_path)
            start_episode = int(match.group(1)) if match else 0
            print(f"Resuming training from episode: {start_episode}")
        else:
            print("Initializing a new PPO model.")
            model = PPO(
                "MultiInputPolicy",
                env,
                learning_rate= 0.0007, 
                n_steps=n_steps_calculated,
                batch_size=60,
                n_epochs=20,
                gamma= 0.995, 
                clip_range=0.25,
                clip_range_vf=None,
                normalize_advantage=True,
                ent_coef=0.09, 
                vf_coef=0.7,
                max_grad_norm=0.5,
                target_kl=None,
                verbose=2,
                tensorboard_log="./ppo_tensorboard/",
                device='auto',
                _init_setup_model=True
            )
            start_episode = 0 # Starting from episode 0 for new training

        # Create callback with proper parameters
        raw_env = env

        logger_callback = TrainingLoggerCallback(
            model=model,
            save_dir=MODEL_SAVE_DIR,
            verbose=1,
            # Set max_episodes to the sum of the starting point and the new desired run length
            max_episodes=num_episodes + start_episode, 
            save_interval=50,
            eval_interval=DEFAULT_EVAL_INTERVAL,
            num_eval_episodes=DEFAULT_EVAL_EPISODES,
            eval_env_id='Minegym-v0', 
            scenario_overrides=scenario_overrides,
            training_env=raw_env
        )
        
        # Set the episode counter for the callback instance to ensure correct naming and stopping logic
        logger_callback.episode_count = start_episode

        try:
            # Train the model (set log_interval=1 for max train log frequency)
            # The total_timesteps is large to allow training to be controlled by the episode limit in the callback
            model.learn(total_timesteps=500000, callback=logger_callback, tb_log_name="div_fun_2", log_interval=1)
            
            # Save final model
            final_model_path = os.path.join(MODEL_SAVE_DIR, "ppo_minegym_final.zip")
            model.save(final_model_path)
            
        finally:
            logger_callback.close()
            env.close()

    elif choice == 'play':

        TESTING_CSV_PATH = "./test_shared.csv" 
        
        # Create a new environment instance for testing
        env_test = gym.make("Minegym-v0", render_mode="console", 
                           scenario_overrides=scenario_overrides, 
                           csv_path=TESTING_CSV_PATH,
                           scenario_name=scenario,
                           play_seed=play_seed)
        
        # List to accumulate results for averaging
        all_results = []

        try:
            if model_path is None:
                print("Error: Model path must be provided for playing.")
                return
            
            # Load model with the separate testing environment
            model = PPO.load(model_path, env=env_test)

            print(f"\n--- Starting Play Mode for {num_episodes} Episodes ---")

            # Run the model for specified number of episodes
            for episode in range(num_episodes):
                # Using a fixed seed (49) per your original setup, but you could use epi_seed
                obs, info = env_test.reset(seed=play_seed) 
                done = False
                cumulative_reward = 0
                step_count = 0

                while not done:
                    action, _states = model.predict(obs, deterministic=False)
                    action_scalar = action.item()
                    obs, reward, done, truncated, info = env_test.step(action_scalar)
                    cumulative_reward += reward
                    step_count += 1
                    
                    if done or truncated:
                        # Capture and store metrics from the final info dictionary
                        final_pvol = info.get('PVOL', 0.0)
                        final_div_score = info.get('DivScore', 0.0)
                        
                        all_results.append({
                            'Episode': episode + 1,
                            'Reward': cumulative_reward,
                            'PVOL': final_pvol,
                            'DivScore': final_div_score,
                            'Steps': step_count
                        })
                        
                        print(f"Episode {episode + 1} finished. Reward: {cumulative_reward:.2f} | PVOL: {final_pvol:.2f}")
                        break

            # --- Report the final average statistics after the loop ---
            if all_results:
                df = pd.DataFrame(all_results)
                mean_reward = df['Reward'].mean()
                mean_pvol = df['PVOL'].mean()
                mean_div_score = df['DivScore'].mean()
                mean_steps = df['Steps'].mean()
                
                print("\n======================================")
                print(f"    Average Play Metrics ({num_episodes} Runs)   ")
                print("======================================")
                print(f"Mean Reward:        {mean_reward:.2f}")
                print(f"Mean PVOL:          {mean_pvol:.2f}")
                print(f"Mean DivScore:      {mean_div_score:.4f}")
                print(f"Mean Steps:         {mean_steps:.2f}")
                print("======================================")
                
        finally:
            env_test.close() # Close the test environment

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
    parser.add_argument(
        '--scenario', 
        type=str, 
        default=None, 
        help="Test scenario (A, B, C, D, E, F)")

    # ADD THIS NEW ARGUMENT
    parser.add_argument(
        '--play_seed',
        type=int,
        default=49,  # Keep the default for backward compatibility
        help="Seed used for environment reset in 'play' mode."
    )

    
    args = parser.parse_args()
    main(args.choice, args.num_episodes, args.model_path, args.scenario, args.play_seed)