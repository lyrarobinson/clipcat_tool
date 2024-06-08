import argparse
from stable_baselines3 import PPO
from loop3 import ImageGlitchEnv

def main():
    parser = argparse.ArgumentParser(description="Process images using a PPO model.")

    parser.add_argument("image_folder", type=str, help="Directory containing input images.")
    parser.add_argument("output_folder", type=str, help="Directory to save processed images.")
    
    # Parsing arguments
    args = parser.parse_args()
    print(f"Image Folder: {args.image_folder}, Output Folder: {args.output_folder}")

    process_images(args.image_folder, args.output_folder)

def process_images(image_folder, output_folder):
    print(f"Processing images from {image_folder} to {output_folder}")
    model = PPO.load("ppo_imageglitch_120000.zip")
    
    env = ImageGlitchEnv(image_folder=image_folder, output_folder=output_folder)

    while True:
        obs, _ = env.reset()
        if obs is None:
            print("No more images to process.")
            break

        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            env.render()  #just kept this in incase its necessary, there's no visualisation
        env.save_processed_image()

    env.close()
    print("processed all images")

if __name__ == "__main__":
    main()
