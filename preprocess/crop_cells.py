from PIL import Image, ImageDraw
import numpy as np
from scipy.ndimage import label, find_objects
import os
from skimage.filters import threshold_otsu

# Increase max image pixels to something very high
Image.MAX_IMAGE_PIXELS = None

# Create a directory to save the cropped images if needed
output_dir = 'scimg_total'
os.makedirs(output_dir, exist_ok=True)

# Directory containing the images
input_dir = 'data/png_images'
image_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]

total_image = 0

for count, image_file in enumerate(image_files):
    # Load the grayscale image
    image_path = os.path.join(input_dir, image_file)
    if not os.path.isfile(image_path):
        print('No file', image_path)
        continue

    image = Image.open(image_path).convert('L')

    # downsample the image to 1/16 of the original size
    image = image.resize((image.width // 4, image.height // 4), Image.LANCZOS)

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Apply a threshold to the image to separate the cells from the background
    threshold_value = threshold_otsu(image_array)
    binary_image = image_array > threshold_value
    
    # Label the binary image
    labeled_array, num_features = label(binary_image)

    # Find the objects in the image
    objects = find_objects(labeled_array)

    # Initialize a list to store the positions of the cells
    cell_positions = []

    # Loop through each object to find the cell positions
    for obj in objects:
        # Get the bounding box of the object
        y_slice, x_slice = obj
        
        if x_slice.start == 0 or x_slice.stop == image_array.shape[1] or y_slice.start == 0 or y_slice.stop == image_array.shape[0]:
            continue
        elif (x_slice.stop - x_slice.start) < 10 and (y_slice.stop - y_slice.start) < 10:
            continue
        else:
            x_center = (x_slice.start + x_slice.stop) // 2
            y_center = (y_slice.start + y_slice.stop) // 2
            cell_positions.append((x_center, y_center))

    # Define the size of the cropped images
    crop_size = 128
    half_crop_size = crop_size // 2

    # Process each cell
    for idx, (x_center, y_center) in enumerate(cell_positions):
        
        # Calculate the crop box coordinates
        left = x_center - half_crop_size
        top = y_center - half_crop_size
        right = x_center + half_crop_size
        bottom = y_center + half_crop_size
        
        # Crop the image
        cropped_image = image_array[top:bottom, left:right]
        
        # Create a mask for the current cell
        mask = np.zeros_like(cropped_image, dtype=bool)
        mask[
            (labeled_array[top:bottom, left:right] == labeled_array[y_center, x_center])
        ] = True
        
        # Apply the mask to remove other cells
        cropped_image = cropped_image * mask
        if np.any(cropped_image):  # Check if there are any non-zero pixels
            cropped_image = cropped_image.astype(np.float32)
            cropped_image = 255 * (cropped_image - cropped_image.min()) / (cropped_image.max() - cropped_image.min())
            cropped_image = cropped_image.astype(np.uint8)
        
            # Convert the cropped image back to a PIL Image
            cropped_pil_image = Image.fromarray(cropped_image)
            
            # Ensure the cropped image is exactly 128x128 by adding padding if necessary
            padded_image = Image.new("L", (crop_size, crop_size))
            padded_image.paste(cropped_pil_image, (0, 0))
            
            # Save the cropped image
            padded_image.save(os.path.join(output_dir, f'cell_{count}_{idx}.png'))
            total_image += 1
            print(count, total_image)
