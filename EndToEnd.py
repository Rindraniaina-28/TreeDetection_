import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from PIL import Image, ImageDraw
import rasterio
from rasterio.windows import Window
from rasterio.merge import merge
import fiona
from shapely.geometry import Polygon, mapping
from ultralytics import YOLO

def is_valid_tiff_filename(filename):
    if not (filename.endswith('.tif') or filename.endswith('.tiff')):
        return False
    parts = os.path.splitext(filename)[0].split('_')
    if len(parts) != 3 or parts[0] != 'tile':
        return False
    try:
        int(parts[1])
        int(parts[2])
    except ValueError:
        return False
    return True

def slice_geotiff_to_tiffs(input_path, output_dir, tile_width, tile_height):
    print(f"Slicing {input_path} into tiles...")
    os.makedirs(output_dir, exist_ok=True)
    with rasterio.open(input_path) as src:
        meta = src.meta.copy()
        for i in range(0, src.height, tile_height):
            for j in range(0, src.width, tile_width):
                window = Window(j, i, tile_width, tile_height)
                transform = rasterio.windows.transform(window, src.transform)
                win_width = min(tile_width, src.width - j)
                win_height = min(tile_height, src.height - i)
                meta.update({
                    "height": win_height,
                    "width": win_width,
                    "transform": transform
                })
                tile_data = src.read(window=window)
                out_path = os.path.join(output_dir, f"tile_{i}_{j}.tif")
                with rasterio.open(out_path, "w", **meta) as dst:
                    dst.write(tile_data)
    print(f"Slicing complete. Tiles saved to {output_dir}.")

def stitch_tiff_tiles(input_dir, output_path):
    print(f"Stitching tiles from {input_dir}...")
    tile_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if is_valid_tiff_filename(f)]
    if not tile_files:
        raise ValueError("No valid TIFF tiles found to stitch.")
    src_files_to_mosaic = [rasterio.open(path) for path in tile_files]
    mosaic, out_trans = merge(src_files_to_mosaic)
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
    })
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)
    for src in src_files_to_mosaic:
        src.close()
    print(f"Stitched TIFF saved to {output_path}")

def create_merged_shp(features, crs, output_shp_path):
    print(f"Creating Shapefile with {len(features)} features...")
    schema = {'geometry': 'Polygon', 'properties': {'id': 'int'}}
    with fiona.open(
        output_shp_path, 'w', driver='ESRI Shapefile', crs=crs, schema=schema
    ) as collection:
        collection.writerecords(features)
    print(f"Shapefile '{output_shp_path}' created successfully.")

def detect_boxes_with_yolo(model, image_path, conf=0.3):
    results = model.predict(source=image_path, conf=conf, save=False)
    boxes = []
    for result in results:
        for box in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box[:4])
            boxes.append([x1, y1, x2, y2])
    return boxes

def run_end_to_end(
    original_tiff_path,
    sliced_dir,
    detected_dir,
    output_tiff_path,
    output_shp_path,
    tile_size=(512, 512),
    model_path="last.pt",
    conf=0.3
):
    slice_geotiff_to_tiffs(
        input_path=original_tiff_path,
        output_dir=sliced_dir,
        tile_width=tile_size[1],
        tile_height=tile_size[0]
    )

    model = YOLO(model_path)

    os.makedirs(detected_dir, exist_ok=True)
    tile_files = [f for f in os.listdir(sliced_dir) if is_valid_tiff_filename(f)]

    all_geo_features = []
    feature_id = 0

    for tile_filename in tile_files:
        tile_path = os.path.join(sliced_dir, tile_filename)
        output_tile_path = os.path.join(detected_dir, tile_filename)

        # Read and convert to RGB if needed
        with rasterio.open(tile_path) as src:
            data = src.read()
            meta = src.meta.copy()
            if data.shape[0] > 3:
                data = data[:3, :, :]
            img = np.moveaxis(data, 0, -1).astype(np.uint8)
            pil_img = Image.fromarray(img)
            temp_jpg = os.path.join(detected_dir, f"{os.path.splitext(tile_filename)[0]}.jpg")
            pil_img.save(temp_jpg)

        # YOLO detection
        local_pixel_boxes = detect_boxes_with_yolo(model, temp_jpg, conf=conf)

        # Draw boxes
        draw = ImageDraw.Draw(pil_img)
        for box in local_pixel_boxes:
            draw.rectangle(box, outline="red", width=3)
        img_with_boxes = np.array(pil_img)
        data_out = np.moveaxis(img_with_boxes, -1, 0)
        # Update meta to match the number of bands
        meta.update(count=data_out.shape[0])
        with rasterio.open(output_tile_path, "w", **meta) as dst:
            dst.write(data_out)

        # Clean up temp file
        os.remove(temp_jpg)

        # Collect features for shapefile
        with rasterio.open(tile_path) as tile_src:
            transform = tile_src.transform
            for box in local_pixel_boxes:
                xmin_pix, ymin_pix, xmax_pix, ymax_pix = box
                top_left_geo = transform * (xmin_pix, ymin_pix)
                top_right_geo = transform * (xmax_pix, ymin_pix)
                bottom_right_geo = transform * (xmax_pix, ymax_pix)
                bottom_left_geo = transform * (xmin_pix, ymax_pix)
                poly_geom = Polygon([top_left_geo, top_right_geo, bottom_right_geo, bottom_left_geo])
                feature = {
                    'geometry': mapping(poly_geom),
                    'properties': {'id': feature_id},
                }
                all_geo_features.append(feature)
                feature_id += 1

    stitch_tiff_tiles(detected_dir, output_tiff_path)

    with rasterio.open(original_tiff_path) as src:
        original_crs = src.crs
    create_merged_shp(all_geo_features, original_crs, output_shp_path)

    print(f"End-to-end process complete. Outputs saved to {output_tiff_path} and {output_shp_path}")
