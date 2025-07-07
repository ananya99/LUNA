import pandas as pd
import os

import sys
sys.path.append(os.path.abspath('/mlbio_scratch/anagupta/luna/LUNA'))

from metrics.evaluation_plot import plot_scatter_visualization, plot_scatter_visualization_custom
from align import process_directories, align_point_clouds2

from utils.data.load import (
    cell_class_decoding,
    compute_distance,
    position_normalize,
    to_dataframe,
)

import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
import os

from scipy.optimize import linear_sum_assignment
import ot

from metrics.evaluation_statistics import (
    compute_contact,
    compute_RSSD,
    align_point_clouds,
    compute_spearman_correlation,
)

def append_to_dataframe(log_dict, file_path):
    """
    Appends log data to an existing DataFrame or creates a new one.
    """
    df = pd.DataFrame([log_dict])

    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        updated_df = pd.concat([existing_df, df], ignore_index=True)
    else:
        updated_df = df

    updated_df.to_csv(file_path, index=False)

def perform_evaluation(
    metadata_true, metadata_pred_normalized, file_path
):
    """
    Evaluates model predictions using various metrics and generates a log dictionary.
    """

    spr_v, spr_p, spr_avg, spr_median = compute_spearman_correlation(
        metadata_true, metadata_pred_normalized
    )
    sum_rssd, mean_rssd, absolute_rssd = compute_RSSD(
        metadata_true, metadata_pred_normalized
    )

    distances_true = compute_distance(metadata_true)
    distances_pred = compute_distance(metadata_pred_normalized)
    precision, f1 = compute_contact(distances_true, distances_pred, percentile=0.1)

    log_dict = {
        "test/precision": precision,
        "test/Spearman's Rank Correlation (Median)": spr_median,
        "test/Spearman's Rank Correlation (Average)": spr_avg,
        "test/F1": f1,
        "test/rssd_absolute": absolute_rssd,
        "test/mean_rssd": mean_rssd,
        "test/sum_rssd": sum_rssd,
        "num_cells": len(metadata_true),
    }

    append_to_dataframe(log_dict, file_path)
    
    return log_dict

def sample_points_from_tissue_image(image, num_samples, dir):
    sigma = 5  # Controls the spatial extent of each cell's "influence"
    min_distance = 5  # Minimum distance between real detected peaks
    intensity_threshold = 10  # Adjust this to ignore low-intensity noise

    # Step 1: Detect bright nuclei-like peaks
    coordinates = peak_local_max(
        image,
        min_distance=min_distance,
        threshold_abs=intensity_threshold
    )

    print(f"Detected {len(coordinates)} bright spots (putative nuclei)")

    # Step 2: Create density map (impulses at nuclei locations)
    density_map = np.zeros_like(image, dtype=np.float32)
    for y, x in coordinates:
        density_map[y, x] = 1.0

    # Step 3: Smooth with Gaussian to create spatial probability map
    prob_map = gaussian_filter(density_map, sigma=sigma)

    # Step 4: Normalize to get a proper probability distribution
    prob_map /= prob_map.sum()

    # Step 5: Flatten and sample pixel indices based on the probability
    flat_probs = prob_map.ravel()
    sampled_indices = np.random.choice(
        len(flat_probs), size=num_samples, replace=True, p=flat_probs
    )

    # Step 6: Convert flat indices to 2D coordinates
    H, W = image.shape
    ys, xs = np.unravel_index(sampled_indices, (H, W))
    coordinates = np.array([[x, y] for x, y in zip(xs, ys)])
    point_cloud = np.zeros((8192, 8192))
    point_cloud[ys, xs] = 1

    # plot thes
    plt.scatter(coordinates[:, 0], coordinates[:, 1], marker='x', color='red', s=5)
    plt.savefig(dir+"/sampled_points_from_tissue_image.png")
    
    # Save the sampled coordinates as a CSV file
    coords_df = pd.DataFrame(coordinates, columns=['coord_X', 'coord_Y'])
    coords_df = position_normalize(coords_df)
    coords_df.to_csv(os.path.join(dir, 'metadata_shape.csv'))
    
    return coordinates

def optimal_transport(target_df, input_df):
    target_coords = target_df[['coord_X', 'coord_Y']].to_numpy()
    input_coords = input_df[['coord_X', 'coord_Y']].to_numpy()
    
    
    # Cost matrix
    M = ot.dist(input_coords[:, :2], target_coords[:, :2], metric='euclidean')**2

    # One-to-one assignment
    row_ind, col_ind = linear_sum_assignment(M)

    # Map each predicted point to one target point
    matched_pts = []
    for row, col in zip(row_ind, col_ind):
        matched_pts.append((input_df.iloc[row]['cell_class'], target_coords[col, 0], target_coords[col, 1]))
        # create final dataframe with the matched points
        final_df = pd.DataFrame(matched_pts, columns=['cell_class', 'coord_X', 'coord_Y'])
    
    return final_df

def main():
    data_dir = '/mlbio_scratch/anagupta/xenium_preprocessed/sliced_data'
    images = np.load(os.path.join(data_dir, 'slice_images.npz'), allow_pickle=True)
    print("Loaded images: ", images.keys())

    directories = ['/mlbio_scratch/anagupta/luna/runs/baseline_sliced/2025-07-02_16-12-52/test_results/test/model_2025-07-02_epoch_999/TgCRND8_5_7__4_0',
                   '/mlbio_scratch/anagupta/luna/runs/baseline_sliced/2025-07-02_16-12-52/test_results/test/model_2025-07-02_epoch_999/TgCRND8_5_7__8_0',
                   '/mlbio_scratch/anagupta/luna/runs/baseline_sliced/2025-07-02_16-12-52/test_results/test/model_2025-07-02_epoch_999/TgCRND8_5_7__9_0',
                   '/mlbio_scratch/anagupta/luna/runs/baseline_sliced/2025-07-02_16-12-52/test_results/test/model_2025-07-02_epoch_999/TgCRND8_5_7__10_0',
                   '/mlbio_scratch/anagupta/luna/runs/baseline_sliced/2025-07-02_16-12-52/test_results/test/model_2025-07-02_epoch_999/TgCRND8_5_7__11_0']
    

    for directory in directories:
        slice_name = '_'.join(directory.rstrip('/').split('/')[-1].split('_')[:-1])
        print(slice_name)
        
        pred_df = pd.read_csv(directory + '/metadata_pred.csv', index_col=0)
        true_df = pd.read_csv(directory + '/metadata_true.csv', index_col=0)
        pred_coords = pred_df[['coord_X', 'coord_Y']].values
        num_samples = pred_coords.shape[0]
        print("num_samples: ", num_samples)
        print("pred_coords: ", pred_coords)
        
        img = images[slice_name]
        shape_guided_coords = sample_points_from_tissue_image(img, num_samples, directory)
        
        # shape_df = pd.read_csv(directory + '/metadata_shape.csv', index_col=0)
    
        
    process_directories(directories)
    
    results_file_path = os.path.join("/mlbio_scratch/anagupta/luna/runs/baseline_sliced/2025-07-02_16-12-52/test_results/test/model_2025-07-02_epoch_999", "test_results3.csv")
    
    for directory in directories:
        slice_name = '_'.join(directory.rstrip('/').split('/')[-1].split('_')[:-1])
        print(slice_name)          
        
        pred_df = pd.read_csv(directory + '/metadata_pred.csv', index_col=0)
        true_df = pd.read_csv(directory + '/metadata_true.csv', index_col=0)
        shape_df = pd.read_csv(directory + '/metadata_shape.csv', index_col=0)
        aligned_df = pd.read_csv(directory + '/metadata_pred_aligned.csv', index_col=0)
        
        plot_scatter_visualization_custom(shape_df, pred_df, directory, "Shape-guided", "Predicted")

        aligned_df = position_normalize(aligned_df)
        plot_scatter_visualization_custom(shape_df, aligned_df, directory, "Shape-guided", "Aligned")

        plot_scatter_visualization_custom(true_df, aligned_df, directory, "Ground-Truth", "Aligned")
        
        final_df = optimal_transport(shape_df, aligned_df)
        final_df = position_normalize(final_df)
        
        # Save the sampled coordinates as a CSV file
        final_df.to_csv(os.path.join(directory, 'metadata_final.csv'))
        
        plot_scatter_visualization_custom(shape_df, final_df, directory, "Shape-guided", "Final")
        
        plot_scatter_visualization_custom(true_df, final_df, directory, "Ground-Truth", "Final")
        
        plot_scatter_visualization_custom(true_df, pred_df, directory, "Ground-Truth", "Predicted")
        
        perform_evaluation(true_df, final_df, results_file_path)


if __name__ == "__main__":
    main()