import numpy as np
import pandas as pd
from scipy.linalg import svd

def load_data(directory):
    """ Load groundtruth and predicted data from specified directory. """
    metadata_true = pd.read_csv(f"{directory}/metadata_shape.csv")
    metadata_pred_unnormalized = pd.read_csv(f"{directory}/metadata_pred.csv", index_col=0)
    return metadata_true, metadata_pred_unnormalized

def align_point_clouds(base, target):
    """ Align target to base using Procrustes analysis (rotation only). """
    # Ensure data are centered at the origin
    base_centered = base - np.mean(base, axis=0)
    target_centered = target - np.mean(target, axis=0)

    # SVD for rotation matrix
    U, _, Vt = svd(np.dot(target_centered.T, base_centered))
    R = np.dot(U, Vt)  # Calculate the rotation matrix

    # Apply rotation to the target
    aligned_target = np.dot(target_centered, R)
    return aligned_target + np.mean(base, axis=0)  # Re-add the mean of the base

import numpy as np
from scipy.linalg import svd

def align_point_clouds2(base, target):
    """Align target to base using Procrustes analysis with optional x/y axis flips."""
    base_centered = base - np.mean(base, axis=0)
    target_centered = target - np.mean(target, axis=0)

    best_error = np.inf
    best_aligned = None

    # Define all flip combinations
    flip_options = [
        np.array([1, 1]),    # no flip
        np.array([-1, 1]),   # flip x
        np.array([1, -1]),   # flip y
        np.array([-1, -1])   # flip both
    ]

    for flip in flip_options:
        flipped_target = target_centered * flip  # Apply flipping
        U, _, Vt = svd(np.dot(flipped_target.T, base_centered))
        R = np.dot(U, Vt)  # Rotation matrix
        aligned = np.dot(flipped_target, R)

        error = np.linalg.norm(base_centered - aligned)
        if error < best_error:
            best_error = error
            best_aligned = aligned

    return best_aligned + np.mean(base, axis=0)

def align_point_clouds_with_reflection_option(base, target):
    """
    Aligns target to base by trying both with and without horizontal flip.
    Returns the best alignment based on mean squared error.
    """
    def align(base, target):
        base_mean = base.mean(0)
        target_mean = target.mean(0)
        base_centered = base - base_mean
        target_centered = target - target_mean

        H = target_centered.T @ base_centered
        U, _, Vt = svd(H)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt

        aligned = target_centered @ R + base_mean
        return aligned

    aligned_normal = align(base, target)
    aligned_flipped = align(base, target * np.array([-1, 1]))  # horizontal flip

    error_normal = np.mean(np.linalg.norm(aligned_normal - base, axis=1))
    error_flipped = np.mean(np.linalg.norm(aligned_flipped - base, axis=1))

    if error_flipped < error_normal:
        return aligned_flipped
    else:
        return aligned_normal


def save_aligned_data(aligned_pred, directory):
    """ Save the aligned predicted data to CSV. """
    aligned_pred.to_csv(f"{directory}/metadata_pred_aligned.csv", index=True)

def process_directories(directories):
    """ Process each directory, aligning predicted point clouds to groundtruth. """
    for directory in directories:
        metadata_true, metadata_pred_unnormalized = load_data(directory)
        
        # Extract coordinates
        true_coords = metadata_true[['coord_X', 'coord_Y']].to_numpy()
        pred_coords = metadata_pred_unnormalized[['coord_X', 'coord_Y']].to_numpy()
        
        # Align coordinates
        aligned_pred_coords = align_point_clouds(true_coords, pred_coords)
        
        # Save the aligned coordinates back to a DataFrame and then to CSV
        aligned_pred_df = pd.DataFrame(aligned_pred_coords, columns=['coord_X', 'coord_Y'], index=metadata_pred_unnormalized.index)
        aligned_pred_df['cell_class'] = metadata_pred_unnormalized['cell_class']
        save_aligned_data(aligned_pred_df, directory)
        print(f"Processed and saved aligned predictions for {directory}")