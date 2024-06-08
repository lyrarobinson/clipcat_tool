#!/usr/bin/env python

import argparse
from stable_baselines3 import PPO
from loop3 import ImageGlitchEnv

def main(image_folder, output_folder):
    # Load the PPO model
    model = PPO.load("ppo_imageglitch_120000.zip")

    # Create an instance of the environment
    env = ImageGlitchEnv(image_folder=image_folder, output_folder=output_folder)

    while True:
        obs, _ = env.reset()
        if obs is None:  # Check if there are no more images to process
            break
        
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            env.render()  # Optional, to visualize the process
        
        env.save_processed_image()  # Save the processed image

    env.close()
    print("Processing completed for all images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images using a PPO model.")
    parser.add_argument("input_dir", type=str, help="Directory containing input images.")
    parser.add_argument("output_dir", type=str, help="Directory to save processed images.")

    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
