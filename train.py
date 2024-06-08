import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from loop3 import ImageGlitchEnv
import numpy as np
import torch  # Import PyTorch to handle CUDA

# Path to the directory containing images to glitch
image_directory = './smalldogsdata'
log_directory = './logdir'
save_directory = './savedir5'

# Ensure the log and save directories exist
os.makedirs(log_directory, exist_ok=True)
os.makedirs(save_directory, exist_ok=True)

class CustomLoggerCallback(BaseCallback):
    def __init__(self, log_dir, save_dir, save_freq=20000, verbose=1):
        super(CustomLoggerCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.save_path = os.path.join(save_dir, 'ppo_imageglitch')
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, 'training_log.txt')

    def _on_step(self) -> bool:
        # Log statistics
        if self.n_calls % self.model.n_steps == 0:
            # Gather statistics
            ep_len_mean = np.mean([ep_info['episode']['l'] for ep_info in self.locals['infos'] if 'episode' in ep_info])
            ep_rew_mean = np.mean([ep_info['episode']['r'] for ep_info in self.locals['infos'] if 'episode' in ep_info])
            
            with open(self.log_file, 'a') as f:
                f.write(f"Time: {self.num_timesteps}, ")
                f.write(f"Iterations: {self.num_timesteps // self.model.n_steps}, ")
                f.write(f"Episode Length Mean: {ep_len_mean}, ")
                f.write(f"Episode Reward Mean: {ep_rew_mean}\n")

        # Save the model at specified intervals
        if self.num_timesteps % self.save_freq == 0:
            save_path = f"{self.save_path}_{self.num_timesteps}"
            self.model.save(save_path)
            if self.verbose > 0:
                print(f"Model saved at {save_path}")

        return True

# Wrap your environment with VecEnv for compatibility with Stable Baselines3
env = make_vec_env(lambda: Monitor(ImageGlitchEnv(image_directory)), n_envs=1)

# Path to the existing model file
model_path = './savedir4/ppo_imageglitch_40000.zip'

# Load the model on CPU first
if os.path.exists(model_path):
    model = PPO.load(model_path, env=env, device='cpu')
    print("Loaded existing model from", model_path)
else:
    model = PPO("MlpPolicy", env, device='cpu', verbose=1)
    print("Starting with a new model because no saved model was found.")

# If CUDA is available, transfer the model to GPU
if torch.cuda.is_available():
    model = PPO.load(model_path, env=env, device='cuda')
    print("Model loaded to CUDA.")
else:
    print("CUDA not available, continuing with CPU.")

# Print the device being used
print(f"Using device: {model.device}")

# Train the model with the custom callback
print("Starting training...")
callback = CustomLoggerCallback(log_dir=log_directory, save_dir=save_directory, save_freq=10000, verbose=1)
model.learn(total_timesteps=5000000, callback=callback)
print("Training completed.")

# Save the final model
model.save(os.path.join(save_directory, "ppo_imageglitch_final"))
print("Model saved as ppo_imageglitch_final.")

# To test the trained model
obs = env.reset()
print("Testing the model...")
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = env.step(action)
    print(f"Action taken: {action}, Reward received: {rewards}")
    env.render()
    if done:
        print("Episode finished.")
        obs = env.reset()
print("Testing completed.")
