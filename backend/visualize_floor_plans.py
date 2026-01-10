"""
Visualization script for floor plan drawings from PNG images.
Displays PNG images from a data directory and its subdirectories.
Annotates rooms based on RGB color values.
"""

import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy import ndimage
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# Room color mapping
ROOM_COLORS = {
    (0, 0, 0): "Walls/Openings",
    (230, 230, 230): "Hall",
    (200, 230, 230): "Corridor",
    (230, 200, 230): "Living Room",
    (230, 230, 200): "Kitchen",
    (230, 200, 200): "Bathroom",
    (200, 200, 230): "Single Bedroom",
    (170, 170, 230): "Master Bedroom",
    (230, 170, 170): "Private Bathroom",
    (140, 140, 230): "Double Bedroom A",
    (110, 110, 230): "Double Bedroom B",
    (230, 170, 230): "Living Room with Kitchenette",
}


def find_pngs_recursive(data_dir):
    """Recursively find all PNG files in data_dir and subdirectories."""
    png_files = []
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.png'):
                png_files.append(os.path.join(root, file))
    
    return sorted(png_files)


def get_room_type(rgb_tuple, tolerance=5):
    """Find the room type for an RGB color with tolerance."""
    for color, room_type in ROOM_COLORS.items():
        if all(abs(rgb_tuple[i] - color[i]) <= tolerance for i in range(3)):
            return room_type
    return None


def find_room_regions(img_array):
    """Find contiguous regions of similar colors representing rooms."""
    # Convert to uint8 if needed
    if img_array.dtype != np.uint8:
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)
    
    # Get RGB channels
    if len(img_array.shape) != 3 or img_array.shape[2] < 3:
        return []
    
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    
    regions = []
    
    # Process each known room color
    for color, room_type in ROOM_COLORS.items():
        if room_type == "Walls/Openings":
            continue  # Skip walls
        
        # Create mask for this color (with tolerance)
        tolerance = 5
        mask = (np.abs(r.astype(int) - color[0]) <= tolerance) & \
               (np.abs(g.astype(int) - color[1]) <= tolerance) & \
               (np.abs(b.astype(int) - color[2]) <= tolerance)
        
        if not mask.any():
            continue
        
        # Label connected components
        labeled, num_features = ndimage.label(mask)
        
        # Get properties of each component
        for label_idx in range(1, num_features + 1):
            component_mask = labeled == label_idx
            points = np.where(component_mask)
            area = len(points[0])
            
            # Only keep regions large enough to be actual rooms
            if area > 100:
                cy, cx = np.mean(points[0]), np.mean(points[1])
                regions.append({
                    'room': room_type,
                    'x': cx,
                    'y': cy,
                    'area': area
                })
    
    return regions


def visualize_png(png_path):
    """Visualize a single PNG image with room annotations."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    img = mpimg.imread(png_path)
    ax.imshow(img)
    
    # Find and label rooms
    regions = find_room_regions(img)
    
    # Add text labels for each room
    for region in regions:
        ax.text(region['x'], region['y'], region['room'], 
                fontsize=8, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                color='black', weight='bold')
    
    filename = os.path.basename(png_path)
    relative_path = os.path.relpath(png_path)
    ax.set_title(f"{filename}\n({relative_path})", fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Add legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                              markersize=8, label=room) 
                      for room in sorted(set(r['room'] for r in regions))]
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    return fig


def visualize_all_pngs(data_dir):
    """Loop through and visualize all PNG images one at a time."""
    png_files = find_pngs_recursive(data_dir)
    
    if not png_files:
        print(f"No PNG files found in {data_dir} or its subdirectories")
        return
    
    print(f"\nLoading {len(png_files)} PNG floor plans from {data_dir}...")
    print("Close the window to move to the next floor plan.\n")
    
    for idx, png_path in enumerate(png_files):
        filename = os.path.basename(png_path)
        relative_path = os.path.relpath(png_path)
        print(f"[{idx + 1}/{len(png_files)}] Displaying: {relative_path}")
        fig = visualize_png(png_path)
        plt.show()


def dataset_statistics(data_dir):
    """Print statistics about the dataset."""
    png_files = find_pngs_recursive(data_dir)
    
    print(f"Dataset Statistics for: {data_dir}")
    print(f"=" * 50)
    print(f"Total PNG files found: {len(png_files)}")
    
    # Group by subdirectory
    dir_counts = {}
    for png_path in png_files:
        png_dir = os.path.dirname(png_path)
        rel_path = os.path.relpath(png_dir, data_dir)
        dir_counts[rel_path] = dir_counts.get(rel_path, 0) + 1
    
    print(f"\nPNG files by directory:")
    for dir_path in sorted(dir_counts.keys()):
        print(f"  {dir_path}: {dir_counts[dir_path]} images")
    
    # Sample image sizes
    if png_files:
        print(f"\nSample image sizes:")
        for png_path in png_files[:5]:
            try:
                img = mpimg.imread(png_path)
                filename = os.path.relpath(png_path, data_dir)
                print(f"  {filename}: {img.shape}")
            except Exception as e:
                print(f"  {os.path.basename(png_path)}: Error - {e}")


if __name__ == "__main__":
    # Set data directory (scans subdirectories for PNGs)
    data_dir = "/Users/josephjia/Documents/lotiv-ai/data-new/FPD_10_NO_COMPACTNESS"
    
    print("Floor Plan PNG Visualization")
    print("=" * 50)
    
    # Print statistics
    dataset_statistics(data_dir)
    
    # Visualize all PNGs
    print("\nStarting PNG visualization loop...")
    visualize_all_pngs(data_dir)
