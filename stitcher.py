from PIL import Image
import os

def stitch_images_to_tiff(detected_dir, output_path, reference_image_path, tile_size, strides):
    base_img = Image.open(reference_image_path)
    base_w, base_h = base_img.size
    stitched_img = Image.new("RGB", (base_w, base_h))

    for file in os.listdir(detected_dir):
        if file.endswith(".png"):
            parts = file.replace("image_", "").replace(".png", "").split("_")
            y, x = map(int, parts)
            tile = Image.open(os.path.join(detected_dir, file))
            stitched_img.paste(tile, (x, y))

    stitched_img.save(output_path, "TIFF")
