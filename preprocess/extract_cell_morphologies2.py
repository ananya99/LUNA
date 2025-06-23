# %%
import pandas as pd
df = pd.read_parquet("/scratch/anagupta/xenium_data/Xenium_V1_FFPE_TgCRND8_2_5_months_outs/nucleus_boundaries.parquet")

import tifffile

fullres_img_tiff = tifffile.imread(
    "/scratch/anagupta/xenium_data/Xenium_V1_FFPE_TgCRND8_2_5_months_outs/morphology_focus.ome.tif", is_ome=False, level=0)

output_dir = "/home/anagupta/luna/preprocess/cell_images"

# %%
import cv2
import numpy as np


# %%
def extract_cell_boundary_vertices(df, scale_factor=0.2125):
    """
    Extract x and y coordinates for all unique cell_ids.

    Parameters:
    df (pd.DataFrame): DataFrame containing cell data with 'cell_id', 'vertex_x', and 'vertex_y' columns.
    scale_factor (float): Scaling factor for coordinates.

    Returns:
    dict: A dictionary with cell_ids as keys and lists of (x, y) tuples as values.
    """
    cell_ids = df["cell_id"].unique()
    cell_coords = {}
    for i, cell_id in enumerate(cell_ids):
        if i % 1000 == 0:
            print(f"Processed {i} cell_boundary_vertices ...")
        cell_df = df[df["cell_id"] == cell_id]
        x_coords = np.ceil(cell_df["vertex_x"].values / scale_factor)
        y_coords = np.ceil(cell_df["vertex_y"].values / scale_factor)
        coords = list(zip(x_coords, y_coords))
        cell_coords[cell_id] = coords
    return cell_coords

#%%
def extract_cell_image_from_boundary_vertices(image, coords):
    """
    image: Input image as a NumPy array (e.g., from cv2.imread or PIL.Image)
    coords: List of (x, y) tuples defining the polygon (e.g., [(x1,y1), (x2,y2), ..., (xn,yn)])
    """
    if not coords:
        raise ValueError("Coordinates list is empty. Cannot extract cell image.")

    # Step 1: Create a mask of the same size as the image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)  # single channel

    # Step 2: Convert coords to numpy array of shape (n,1,2)
    polygon = np.array([coords], dtype=np.int32)

    # Step 3: Fill the polygon area with white (255)
    cv2.fillPoly(mask, polygon, 255)

    # Step 4: Bitwise-and mask with image
    masked = cv2.bitwise_and(image, image, mask=mask)

    # Step 5: Crop to bounding box of polygon (optional but cleaner)
    x, y, w, h = cv2.boundingRect(polygon)
    if w == 0 or h == 0:
        raise ValueError(f"Bounding box has zero width or height: (x={x}, y={y}, w={w}, h={h}). Check the input coordinates.")
    cropped = masked[y:y+h, x:x+w]

    return cropped


# %%
import os
import gc
import h5py
from multiprocessing import Pool, cpu_count

directories = [
    "/scratch/anagupta/xenium_data/Xenium_V1_FFPE_TgCRND8_2_5_months_outs/",
    "/scratch/anagupta/xenium_data/Xenium_V1_FFPE_TgCRND8_5_7_months_outs/",
    "/scratch/anagupta/xenium_data/Xenium_V1_FFPE_TgCRND8_17_9_months_outs/",
    "/scratch/anagupta/xenium_data/Xenium_V1_FFPE_wildtype_2_5_months_outs/",
    "/scratch/anagupta/xenium_data/Xenium_V1_FFPE_wildtype_5_7_months_outs/",
    "/scratch/anagupta/xenium_data/Xenium_V1_FFPE_wildtype_13_4_months_outs/"
    ]

# read nucleus_boundaries.parquet and morphology_focus.ome.tif for each directory
for directory in directories:
    sample_name = directory.split("_V1_FFPE_")[1].split("_months_outs")[0]

    df = pd.read_parquet(os.path.join(directory, "nucleus_boundaries.parquet"), engine="pyarrow")
    tiff = tifffile.imread(os.path.join(directory, "morphology_focus.ome.tif"), is_ome=False, level=0)
    print(f"Loaded {directory} with shape {tiff.shape} and {df.shape[0]} rows")

    # extract cell coordinates
    cell_boundary_vertices = extract_cell_boundary_vertices(df)
    print("cell_boundary_vertices", cell_boundary_vertices)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, sample_name + "_cell_images.hdf5")

    for i, (cell_id, cell_boundary_vertices) in enumerate(cell_boundary_vertices.items()):
        if i % 1000 == 0:
            print(f"Processed {i} cell_images...")
        try:
            # Extract the polygon region from the image
            cell_img = extract_cell_image_from_boundary_vertices(tiff, cell_boundary_vertices)
            print(f"Extracted cell image for cell_id {cell_id} with shape {cell_img.shape}")
            # save the image to a png file
            cv2.imwrite(os.path.join(output_dir, f"{cell_id}.png"), cell_img)
        except Exception as e:
            print(f"Error processing cell_id {cell_id} in {sample_name}: {e}. Skipping.")
            continue

    print(f"Finished saving all cells for {sample_name} to {output_file}")

        
