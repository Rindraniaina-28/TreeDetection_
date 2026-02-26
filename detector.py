from PIL import Image
import os

def run_tile_processing(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        if file.endswith(".png"):
            img = Image.open(os.path.join(input_dir, file))
            result = img.copy()  # Dummy processing
            result.save(os.path.join(output_dir, file))
