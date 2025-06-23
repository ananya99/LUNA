import pandas as pd
import tifffile
import cv2
import numpy as np
import tarfile
import io
import os
import matplotlib.pyplot as plt
import random

input_directories = [
    "/scratch/anagupta/xenium_data/Xenium_V1_FFPE_TgCRND8_2_5_months_outs/",
    "/scratch/anagupta/xenium_data/Xenium_V1_FFPE_TgCRND8_5_7_months_outs/",
    "/scratch/anagupta/xenium_data/Xenium_V1_FFPE_TgCRND8_17_9_months_outs/",
    "/scratch/anagupta/xenium_data/Xenium_V1_FFPE_wildtype_2_5_months_outs/",
    "/scratch/anagupta/xenium_data/Xenium_V1_FFPE_wildtype_5_7_months_outs/",
    "/scratch/anagupta/xenium_data/Xenium_V1_FFPE_wildtype_13_4_months_outs/"
    ]
output_dir = "/home/anagupta/luna/preprocess/cell_images22"

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


def extract_cell_image_from_boundary_vertices(image, coords):
    """
    image: Input image as a NumPy array (e.g., from cv2.imread or PIL.Image)
    coords: List of (x, y) tuples defining the vertices of the cell boundary polygon (e.g., [(x1,y1), (x2,y2), ..., (xn,yn)])
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


# add padding to the images to make them target size without resizing
def add_padding(image, target_size=(128, 128)):
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

#%%
# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the data and extract cell images
for directory in input_directories:
    sample_name = directory.split("_V1_FFPE_")[1].split("_months_outs")[0]

    df = pd.read_parquet(os.path.join(directory, "nucleus_boundaries.parquet"), engine="pyarrow")
    tiff = tifffile.imread(os.path.join(directory, "morphology_focus.ome.tif"), is_ome=False, level=0)
    print(f"Loaded {directory} with shape {tiff.shape} and {df.shape[0]} rows")
    cell_boundary_vertices = extract_cell_boundary_vertices(df)

    # Option to enable or disable batching
    enable_batching = True
    batch_size = 1000
    batch_tar_file_path_template = os.path.join(output_dir, f"{sample_name}_cell_images_batch_{{batch_idx}}.tar")
    tar_file_path = os.path.join(output_dir, f"{sample_name}_cell_images.tar")

    if enable_batching:
        batch_idx = 0
        with tarfile.open(batch_tar_file_path_template.format(batch_idx=batch_idx), "w") as tar:
            for i, (cell_id, cell_boundary_vertices) in enumerate(cell_boundary_vertices.items()):
                if i % 1000 == 0 and i > 0:
                    # Close the current tar file and start a new one
                    tar.close()
                    batch_idx += 1
                    tar = tarfile.open(batch_tar_file_path_template.format(batch_idx=batch_idx), "w")
                    print(f"Started new tar file: {batch_tar_file_path_template.format(batch_idx=batch_idx)}")

                try:
                    # Extract the polygon region from the image
                    cell_img = extract_cell_image_from_boundary_vertices(tiff, cell_boundary_vertices)
                    
                    # Add padding to the image
                    cell_img_padded = add_padding(cell_img, target_size=(64, 64))
                    
                    # Save the image to an in-memory buffer
                    img_buffer = io.BytesIO()
                    np.save(img_buffer, cell_img_padded)
                    img_buffer.seek(0)
                    
                    # Add the image to the tar file
                    tar_info = tarfile.TarInfo(name=f"{cell_id}.npy")
                    tar_info.size = img_buffer.getbuffer().nbytes
                    tar.addfile(tarinfo=tar_info, fileobj=img_buffer)
                    
                except Exception as e:
                    print(f"Error processing cell_id {cell_id} in {sample_name}: {e}. Skipping.")
                    continue

            # Ensure the last tar file is closed
            tar.close()

        print(f"Finished saving all cells for {sample_name} in batches to {output_dir}")
    else:
        with tarfile.open(tar_file_path, "w") as tar:
            for cell_id, cell_boundary_vertices in cell_boundary_vertices.items():
                try:
                    # Extract the polygon region from the image
                    cell_img = extract_cell_image_from_boundary_vertices(tiff, cell_boundary_vertices)
                    
                    # Add padding to the image
                    cell_img_padded = add_padding(cell_img, target_size=(64, 64))
                    
                    # Save the image to an in-memory buffer
                    img_buffer = io.BytesIO()
                    np.save(img_buffer, cell_img_padded)
                    img_buffer.seek(0)
                    
                    # Add the image to the tar file
                    tar_info = tarfile.TarInfo(name=f"{cell_id}.npy")
                    tar_info.size = img_buffer.getbuffer().nbytes
                    tar.addfile(tarinfo=tar_info, fileobj=img_buffer)
                    
                except Exception as e:
                    print(f"Error processing cell_id {cell_id} in {sample_name}: {e}. Skipping.")
                    continue

        print(f"Finished saving all cells for {sample_name} to {tar_file_path}")

    print(f"Finished saving all cells for {sample_name} to {tar_file_path}")


# %%
# load the tar file and plot randomly 10 cell images from each sample
output_dir = "/home/anagupta/luna/preprocess/cell_images"
sample_names = [directory.split("_V1_FFPE_")[1].split("_months_outs")[0] for directory in input_directories]

for sample_name in sample_names:
    tar_file_path = os.path.join(output_dir, f"{sample_name}_cell_images.tar")
    with tarfile.open(tar_file_path, "r") as tar:
        members = tar.getmembers()
        random_members = random.sample(members, 10)
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        for ax, member in zip(axes.flatten(), random_members):
            # Extract the image from the tar file
            img_buffer = tar.extractfile(member).read()
            cell_img = np.load(io.BytesIO(img_buffer))
            ax.imshow(cell_img, cmap="gray")
            ax.set_title(f"Cell ID: {member.name}")
            ax.axis("off")
        plt.tight_layout()
        plt.show()
