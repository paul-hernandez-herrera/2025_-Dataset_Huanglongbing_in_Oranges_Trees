# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 10:59:05 2025

@author: paulh
"""

import numpy as np
from skimage import io, filters, morphology
from skimage.measure import label
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes
from sklearn.decomposition import PCA
from PIL import Image
from pathlib import Path
from util import display_step_standarization
import pandas as pd
 
    

# Input folder containing images or a image file to be processed
input_folder = r"path_to_folder_or_file"


input_folder = Path(input_folder)
flag_display_detailed_step = True

# Valid file extensions for images
valid_extensions = {'.png', '.jpeg', '.jpg', '.tif'}

# Determine input files and folder structure
if input_folder.is_dir():
    image_files = [file.name for file in input_folder.iterdir() if file.suffix in valid_extensions]
    root_folder = input_folder
else:
    image_files = [input_folder.name]
    input_folder = input_folder.parent
    
# Define output paths
output_folder_images = Path(input_folder,"preprocessed_images")
output_folder_visualizations = Path(input_folder,"visualizations") 
output_folder_steps = Path(input_folder,"steps")    
output_csv_path = Path(input_folder, "segmentation_thresholds.csv")

# Create output directories if they don't exist
output_folder_images.mkdir(parents=True, exist_ok=True)
output_folder_visualizations.mkdir(parents=True, exist_ok=True)
if flag_display_detailed_step:
    output_folder_steps.mkdir(parents=True, exist_ok=True)    

# Initialize thresholds DataFrame
if output_csv_path.exists():
    Data = pd.read_csv(output_csv_path)
    #thresholds_df = pd.read_csv(output_csv_path).set_index('file_name', drop=False)
    thresholds_df = pd.DataFrame(columns = ["file_name","Hue_L", "Hue_H", "Saturation_L", "Saturation_H", "Brightness_L","Brightness_H", "Method"])
    for key in Data.index:
        file_name = Data.loc[key]["file_name"]
        thresholds_df.loc[file_name] = Data.loc[key]
else:
    thresholds_df = pd.DataFrame(columns = ["file_name","Hue_L", "Hue_H", "Saturation_L", "Saturation_H", "Brightness_L","Brightness_H", "Method"])

# Process each image
for file_name in image_files:
    print(f"Processing: {file_name}")
    
    # Read and convert image to HSV
    image = io.imread(Path(input_folder,file_name))
    img_hsv = np.array(Image.fromarray(image).convert('HSV'))
    
    # get each channel HSV
    hue = img_hsv[:,:,0]
    saturation = img_hsv[:,:,1]
    value = img_hsv[:,:,2]    
    
    # Set thresholds values for current image
    thresholds = {
        "file_name": file_name,
        "Hue_L": 30,
        "Hue_H": int(filters.threshold_otsu(img_hsv[:, :, 0])),
        "Saturation_L": 0,
        "Saturation_H": 255,
        "Brightness_L": 0,
        "Brightness_H": int(filters.threshold_otsu(img_hsv[:, :, 2])),
        "Method": "Automatic"
    }
    
    # Optionally adjust thresholds manually. To fix incorrect segmentations.
    manual_adjustment = False
    if manual_adjustment:
        thresholds.update({"Hue_L": 20,"Hue_H": 130, 
                           "Saturation_L": 105, "Saturation_H": 255,
                           "Brightness_L": 60, "Brightness_H": 233, 
                           "Method": "Manual"})
        
    #print(thresholds_df.loc[file_name])
    thresholds_df.loc[file_name] = thresholds
    #print("Done\n\n")
    
    # Segment the image based on thresholds    
    segmented_image_raw = (
        (hue >= thresholds["Hue_L"]) & (hue <= thresholds["Hue_H"]) &
        (saturation >= thresholds["Saturation_L"]) & (saturation <= thresholds["Saturation_H"]) &
        (value >= thresholds["Brightness_L"]) & (value <= thresholds["Brightness_H"])
    )   
    
    
    segmented_image_fill_holes = binary_fill_holes(segmented_image_raw)
    
    # Label connected components and keep the largest one
    labeled_image = label(segmented_image_fill_holes)
    
    # Find the size of each connected component
    component_sizes = np.bincount(labeled_image.ravel())
    
    # Exclude the background
    component_sizes[0] = 0
    
    # Find the largest connected component
    largest_component = component_sizes.argmax()
    
    # Create a mask for the largest connected component
    segmented_image = (labeled_image == largest_component)  
    
    
    # Extract principal component for rotation
    coordinates = np.column_stack(np.where(segmented_image >0))
    pca = PCA();
    pca.fit(coordinates)
    eigenvector1 = pca.components_[0]
    
    # Calculate rotation angle
    theta = np.arctan2(eigenvector1[1],eigenvector1[0])* 180 / np.pi

    # Prepare segmented image for rotation
    segmented_rgb = np.zeros_like(image)
    for channel in range(image.shape[2]):
        # remove background
        segmented_rgb[:, :, channel] = image[:, :, channel] * segmented_image

    
    # Calculate center of rotation
    center = tuple(np.mean(coordinates,axis = 0))
    
    # ROTAR IMAGEN
    rotated_image = Image.fromarray(segmented_rgb).rotate(
        angle = -theta+90,  # add 90 degrees to align image horizontal
        center = center[::-1], resample  = Image.Resampling.BILINEAR, expand = True)
    
    rotated_image = np.array(rotated_image)
    
    # Crop rotated image
    coordinates = np.column_stack(np.where(rotated_image[:,:,1] >0))
    x_min,x_max = coordinates[:,0].min(), coordinates[:,0].max()
    y_min,y_max = coordinates[:,1].min(), coordinates[:,1].max()
    
    cropped_image = rotated_image[x_min:x_max, y_min:y_max, :]

    # Save visualization and processed image
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image)
    contour = morphology.binary_dilation(segmented_image).astype(np.uint8) - segmented_image
    contour_coords = np.column_stack(np.where(contour > 0))
    axes[0].plot(contour_coords[:, 1], contour_coords[:, 0], 'r.')
    axes[1].imshow(cropped_image)
    
    plt.title(file_name)
    plt.savefig(Path(output_folder_visualizations, file_name+ ".jpeg"))
    plt.show()
    
    Image.fromarray(cropped_image).save(Path(output_folder_images,Path(file_name).stem+ ".png"))    

    if flag_display_detailed_step:
        display_step_standarization(image, img_hsv, segmented_image_raw, 
                                    segmented_image_fill_holes, segmented_rgb, 
                                    rotated_image, cropped_image,
                                    center, pca,
                                    Path(output_folder_steps, file_name))
        
thresholds_df.to_csv(output_csv_path, index=False, encoding="utf-8")
print(f"Resultados guardados en: {output_csv_path}")