# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 16:52:21 2025

@author: paulh
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.patches import Rectangle

flag_high_quality = True

def display_step_standarization(raw_img, hsv_image, raw_seg, fill_seg, rgb_seg, rgb_rotate, rgb_cropped, center, pca, path_output):

    # Create the main figure and gridspec
    fig = plt.figure(figsize=(10, 10))
    outer_gs = gridspec.GridSpec(3, 3, figure=fig)
    
    # Create main axes
    ax00 = fig.add_subplot(outer_gs[0, 0])  # First row, first column
    ax10 = fig.add_subplot(outer_gs[1, 0])  # Second row, first column
    ax11 = fig.add_subplot(outer_gs[1, 1])  # Second row, second column
    ax12 = fig.add_subplot(outer_gs[1, 2])  # Second row, third column
    ax20 = fig.add_subplot(outer_gs[2, 0])  # Third row, first column
    ax21 = fig.add_subplot(outer_gs[2, 1])  # Third row, second column
    ax22 = fig.add_subplot(outer_gs[2, 2])  # Third row, third column
    
    # Create a nested GridSpec for ax01_02 (spanning 0,1 and 0,2)
    nested_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[0, 1:3])
    
    # Add subplots inside ax01_02
    nested_ax1 = fig.add_subplot(nested_gs[0, 0])  # First column of the nested grid
    nested_ax2 = fig.add_subplot(nested_gs[0, 1])  # Second column of the nested grid
    nested_ax3 = fig.add_subplot(nested_gs[0, 2])  # Third column of the nested grid
    
    # Add RGB IMAGE
    
    ax00.imshow(raw_img)   
    ax00.axis('off')  
    ax00.set_title("RGB image")
    
    # Add HSV IMAGE
    plot_hue = nested_ax1.imshow(hsv_image[:,:,0], cmap='hsv')
    nested_ax1.set_title("Hue Channel")
    nested_ax1.axis('off')
    fig.colorbar(plot_hue, ax=nested_ax1, orientation='horizontal', pad=0.04)    

    
    plot_sat = nested_ax2.imshow(hsv_image[:,:,1], cmap='gray')
    nested_ax2.set_title("Step 1: HSV Image\n\nSaturation Channel")
    nested_ax2.axis('off')
    fig.colorbar(plot_sat, ax=nested_ax2, orientation='horizontal', pad=0.04)    
    
    plot_value = nested_ax3.imshow(hsv_image[:,:,2], cmap='gray')
    nested_ax3.set_title("Value Channel")
    nested_ax3.axis('off')
    fig.colorbar(plot_value, ax=nested_ax3, orientation='horizontal', pad=0.04)    
    
    # Add Segmentation
    plot_value = ax10.imshow(raw_seg, cmap='gray')
    ax10.set_title("Step 2: Segmentation")
    ax10.axis('off')    
    
    # Fill holes Segmentation
    plot_value = ax11.imshow(fill_seg, cmap='gray')
    ax11.set_title("Step 3: Fill holes")
    ax11.axis('off')   

    # PCA
    
    plot_value = ax12.imshow(fill_seg, cmap='gray')
    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_    

    ax12.scatter(center[1], center[0], color='cyan', label="Center", zorder=10)
    c = ["blue", "red"]
    for i in range(len(eigenvalues)):
        vector = 2*np.sqrt(eigenvalues[i])*eigenvectors[:,i]
        ax12.arrow(center[1], center[0], vector[1], vector[0],
                  head_width=np.sqrt(eigenvalues[i])/3, head_length=np.sqrt(eigenvalues[i])/3, fc=c[i], ec= c[i])    
    ax12.set_title("Step 4: Principal Orientation")
    ax12.axis('off') 
    
    # RGB segmented
    plot_value = ax20.imshow(rgb_seg, cmap='gray')
    ax20.set_title("Step 5: RGB Background removal")
    ax20.axis('off')      
    


    # RGB segmented
    plot_value = ax21.imshow(rgb_rotate)
    ax21.set_title("Step 6: RGB aligned")
    ax21.axis('off')     
    
    # get the new principal directions from the rotated image
    coordinates = np.column_stack(np.where(rgb_rotate[:,:,1] >0))
    pca.fit(coordinates)    
    
    # Calculate center of rotation
    center = tuple(np.mean(coordinates,axis = 0))    
    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_ 

    # Find the bounding box of the non-zero region
    non_zero_indices = np.argwhere(rgb_rotate[:,:,0] > 0)
    min_row, min_col = non_zero_indices.min(axis=0)
    max_row, max_col = non_zero_indices.max(axis=0)    

    # Add a bounding box
    rect = Rectangle((min_col, min_row),  # Bottom-left corner
                     max_col - min_col,  # Width
                     max_row - min_row,  # Height
                     linewidth=2, edgecolor='yellow', facecolor='none')
    ax21.add_patch(rect)
    
    ax21.scatter(center[1], center[0], color='cyan', label="Center", zorder=10)
    
    c = ["blue", "red"]
    for i in range(len(eigenvalues)):
        vector = 2*np.sqrt(eigenvalues[i])*eigenvectors[:,i]
        ax21.arrow(center[1], center[0], vector[1], vector[0],
                  head_width=np.sqrt(eigenvalues[i])/3, head_length=np.sqrt(eigenvalues[i])/3, fc=c[i], ec= c[i])     

    # Final Image
    plot_value = ax22.imshow(rgb_cropped)
    ax22.set_title("Step 7: Cropped image/Final")
    ax22.axis('off')        
    
    # Adjust layout
    plt.tight_layout(pad=2.0) 
    # Save the plot as a high-quality image
    if flag_high_quality:
        plt.savefig(path_output.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.1)
    else:
        plt.savefig(path_output.with_suffix(".jpeg"))
        
    plt.show()