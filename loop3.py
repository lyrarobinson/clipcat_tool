# In loop3.py
import os
import random
from PIL import Image
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from transformers import CLIPProcessor, CLIPModel
import torch

model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

def classify_image(img):
    img = img.resize((224, 224))
    global model, processor
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_labels = ["a cat", "a tree", "a dog", "a building", "a chair", "a horse", "a car", "a house", "a room", "a street", "a bed", "a table", "a kitchen"]
        text_inputs = processor(text=text_labels, return_tensors="pt", padding=True)
        text_features = model.get_text_features(**text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarities = torch.nn.functional.cosine_similarity(image_features, text_features)
        predicted_index = similarities.argmax().item()
        predicted_label = text_labels[predicted_index]
        max_similarity = similarities.max().item()
    return predicted_label, max_similarity

def color_channel_shift(img_array, shift_amount):
    height, width, channels = img_array.shape
    for channel in range(channels):
        img_array[:, :, channel] = np.roll(img_array[:, :, channel], shift_amount, axis=0)
    return img_array

def random_pixel_noise(img_array, noise_level=15):
    noise = np.random.randint(-noise_level, noise_level + 1, img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255)
    return img_array

def block_shift(img_array, block_size=150):
    height, width, _ = img_array.shape
    for start_row in range(0, height, block_size):
        for start_col in range(0, width, block_size):
            end_row = min(start_row + block_size, height)
            end_col = min(start_col + block_size, width)
            shift_amount_row = random.randint(-block_size//1, block_size//1)
            shift_amount_col = random.randint(-block_size//1, block_size//1)
            img_array[start_row:end_row, start_col:end_col] = np.roll(
                img_array[start_row:end_row, start_col:end_col],
                shift=(shift_amount_row, shift_amount_col),
                axis=(0, 1)
            )
    return img_array

def create_glitch_art(img, action):
    img = img.resize((224, 224)).convert('RGB')
    img_array = np.array(img, dtype=np.uint8)
    if action[0] > 0:
        img_array = color_channel_shift(img_array, action[0])
    if action[1] > 0:
        img_array = random_pixel_noise(img_array, action[1] * 3)
    if action[2] > 0:
        img_array = block_shift(img_array, action[2] * 30)
    return Image.fromarray(img_array.astype(np.uint8), 'RGB')

class ImageGlitchEnv(gym.Env):
    """Custom Environment that processes images with glitches and saves the output."""

    def __init__(self, image_folder, output_folder, image_size=(224, 224), max_steps=150):
        super(ImageGlitchEnv, self).__init__()
        self.image_folder = image_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        self.image_size = image_size
        self.image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith(('.png', '.jpg', '.jpeg'))]
        self.max_steps = max_steps
        self.current_step = 0
        self.action_space = spaces.MultiDiscrete([5, 5, 5])
        
        self.image_shape = (image_size[0], image_size[1], 3)
        self.observation_space = spaces.Box(low=0, high=255, shape=(np.prod(self.image_shape),), dtype=np.uint8)
        
        self.seed()
        self.current_image_index = -1
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        return [seed]

    def step(self, action):
        self.current_image = create_glitch_art(self.current_image, action)
        self.current_step += 1
        predicted_label, similarity_score = classify_image(self.current_image)
        done = predicted_label == "a cat" or self.current_step >= self.max_steps
        reward = similarity_score if predicted_label == "a cat" else 0
        print(f"Step {self.current_step}: Label={predicted_label}, Similarity={similarity_score:.2f}, Done={done}")
        
        flattened_image = np.array(self.current_image).flatten()
        print(f"Flattened image shape: {flattened_image.shape}")
        
        if flattened_image.shape[0] != np.prod(self.image_size) * 3:  # Assuming RGB images
            raise ValueError(f"Unexpected flattened image shape after step: {flattened_image.shape}, expected: {np.prod(self.image_size) * 3}")
        
        truncated = False
        return flattened_image, reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.current_image_index += 1
        if self.current_image_index >= len(self.image_files):
            print("All images processed.")
            return None, None
        
        # Load the next image
        self.current_image_path = self.image_files[self.current_image_index]  # Ensure this is set
        self.original_image = Image.open(self.current_image_path).resize(self.image_size)
        self.current_image = self.original_image.copy()
        self.current_step = 0
        
        flattened_image = np.array(self.original_image).flatten()
        print(f"Resetting environment. New image loaded from {self.current_image_path}")
        print(f"Flattened image shape: {flattened_image.shape}")
        
        if flattened_image.shape[0] != np.prod(self.image_size) * 3:  # Assuming RGB images
            raise ValueError(f"Unexpected flattened image shape: {flattened_image.shape}, expected: {np.prod(self.image_size) * 3}")
        
        return flattened_image, {}


    def save_processed_image(self):
        base_name = os.path.basename(self.current_image_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(self.output_folder, f"{name}_cat{ext}")
        self.current_image.save(output_path)
        print(f"Processed image saved to {output_path}")

    def render(self, mode='human'):
        if mode == 'human':
            print("Rendering current image...")
            self.current_image.show()


def color_channel_shift(img_array, shift_amount):
    height, width, channels = img_array.shape
    for channel in range(channels):
        img_array[:, :, channel] = np.roll(img_array[:, :, channel], shift_amount, axis=0)
    return img_array

def random_pixel_noise(img_array, noise_level=15):
    noise = np.random.randint(-noise_level, noise_level + 1, img_array.shape)  # Corrected range
    img_array = np.clip(img_array + noise, 0, 255)
    return img_array

def block_shift(img_array, block_size=150):
    height, width, _ = img_array.shape
    for start_row in range(0, height, block_size):
        for start_col in range(0, width, block_size):
            end_row = min(start_row + block_size, height)
            end_col = min(start_col + block_size, width)
            shift_amount_row = random.randint(-block_size//1, block_size//1)
            shift_amount_col = random.randint(-block_size//1, block_size//1)
            img_array[start_row:end_row, start_col:end_col] = np.roll(
                img_array[start_row:end_row, start_col:end_col],
                shift=(shift_amount_row, shift_amount_col),
                axis=(0, 1)
            )
    return img_array

def create_glitch_art(img, action):
    """Apply glitches to the image based on the action."""
    img = img.resize((224, 224)).convert('RGB') 
    img_array = np.array(img, dtype=np.uint8)
    # Color Channel Shift
    if action[0] > 0:
        img_array = color_channel_shift(img_array, action[0])
    # Pixel Noise
    if action[1] > 0:
        img_array = random_pixel_noise(img_array, action[1] * 3)  # Scale noise level
    # Block Shift
    if action[2] > 0:
        img_array = block_shift(img_array, action[2] * 30)  # Scale block size
    return Image.fromarray(img_array.astype(np.uint8), 'RGB')
