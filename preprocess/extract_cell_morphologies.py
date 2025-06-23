# %%
import pandas as pd
df = pd.read_parquet("/scratch/anagupta/xenium_data/Xenium_V1_FFPE_TgCRND8_2_5_months_outs/nucleus_boundaries.parquet")

import tifffile

fullres_img_tiff = tifffile.imread(
    "/scratch/anagupta/xenium_data/Xenium_V1_FFPE_TgCRND8_2_5_months_outs/morphology_focus.ome.tif", is_ome=False, level=0)

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
        if i >10:
            break
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
    polygon = np.array([coords], dtype=np.int32).reshape((-1, 1, 2))

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
# add padding to the images to make them 128 x 128 without resizing
def add_padding(image, target_size=(128, 128)):
    """
    Add padding to the image to make it the target size without resizing.
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size

    # Create a new blank image with the target size
    padded_image = np.zeros((target_h, target_w), dtype=image.dtype)

    # Calculate padding offsets
    y_offset = (target_h - h) // 2
    x_offset = (target_w - w) // 2

    # Place the original image in the center of the padded image
    padded_image[y_offset:y_offset + h, x_offset:x_offset + w] = image

    return padded_image

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
    cell_coords = extract_cell_boundary_vertices(df)
    output_dir = "cell_images"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, sample_name + "_cell_images.hdf5")
    
    cell_imgs = []
    cell_ids = []
    
    batch_size = 1000
    batch_imgs = []
    batch_ids = []
    batch_count = 0

    cell_image_shape = (128, 128)

    def process_cell(args):
        cell_id, coords = args
        try:
            # Extract the polygon region from the image
            cell_img = extract_cell_image_from_boundary_vertices(tiff, coords)
            # Add padding to the image
            cell_img = add_padding(cell_img, cell_image_shape)
            return cell_img, cell_id
        except Exception as e:
            print(f"Error processing cell_id {cell_id} in {sample_name}: {e}. Skipping.")
            return None

    with h5py.File(output_file, "w") as f:
        print(f"Creating HDF5 file: {output_file}")
        img_dataset = f.create_dataset("images", shape=(0, 128, 128), maxshape=(None, 128, 128), dtype=np.uint8, compression="gzip")
        id_dataset = f.create_dataset("cell_ids", shape=(0,), maxshape=(None,), dtype='S32', compression="gzip")

        # Use multiprocessing to process cells in parallel
        with Pool(cpu_count()) as pool:
            results = pool.map(process_cell, cell_coords.items())

        batch_imgs = []
        batch_ids = []
        batch_count = 0

        for result in results:
            if result is None:
                continue
            cell_img, cell_id = result
            batch_imgs.append(cell_img)
            batch_ids.append(cell_id)

            # Save in batches
            if len(batch_imgs) >= batch_size:
                batch_imgs = np.stack(batch_imgs)  # shape: (batch_size, 100, 100)
                img_dataset.resize(img_dataset.shape[0] + batch_imgs.shape[0], axis=0)
                img_dataset[-batch_imgs.shape[0]:] = batch_imgs

                id_dataset.resize(id_dataset.shape[0] + len(batch_ids), axis=0)
                id_dataset[-len(batch_ids):] = batch_ids

                print(f"Saved batch {batch_count + 1} with {len(batch_ids)} cells to {output_file}")
                batch_imgs = []
                batch_ids = []
                batch_count += 1

        # Save any remaining cells
        if batch_imgs:
            batch_imgs = np.stack(batch_imgs)
            img_dataset.resize(img_dataset.shape[0] + batch_imgs.shape[0], axis=0)
            img_dataset[-batch_imgs.shape[0]:] = batch_imgs

            id_dataset.resize(id_dataset.shape[0] + len(batch_ids), axis=0)
            id_dataset[-len(batch_ids):] = batch_ids

            print(f"Saved final batch with {len(batch_ids)} cells to {output_file}")

        print(f"Finished saving all cells for {sample_name} to {output_file}")

        # plot a few images from saved hdf5 file

        
