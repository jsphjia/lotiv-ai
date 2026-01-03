import pickle
import matplotlib.pyplot as plt
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union
from shapely.validation import make_valid
import os

# Load the ResPlan.pkl file
def load_resplan_dataset(pkl_file_path):
    with open(pkl_file_path, 'rb') as file:
        data = pickle.load(file)
    return data

# Ensure the output directory exists
output_dir = "output_plots_detailed"
os.makedirs(output_dir, exist_ok=True)

# Plot bounding boxes for each floor plan
def plot_bounding_boxes(data):
    for i, floor_plan in enumerate(data):
        print(f"Processing floor plan {i + 1}...")

        # Combine all geometries in the floor plan to calculate the bounding box
        all_geometries = []
        for _, value in floor_plan.items():
            if isinstance(value, MultiPolygon):
                all_geometries.extend(value.geoms)
            elif isinstance(value, Polygon):
                all_geometries.append(value)

        if all_geometries:
            # Calculate the bounding box
            combined = MultiPolygon(all_geometries)
            minx, miny, maxx, maxy = combined.bounds
            bounding_box = box(minx, miny, maxx, maxy)

            # Plot the bounding box
            plt.figure()
            x, y = bounding_box.exterior.xy
            plt.plot(x, y, color='black', linewidth=2)

            # Hide axis and labels
            plt.axis('off')

            # Save the plot to the output directory
            plot_path = os.path.join(output_dir, f"floor_plan_{i + 1}.png")
            plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            print(f"Saved plot for floor plan {i + 1} to {plot_path}")
        else:
            print(f"No geometries found in floor plan {i + 1}.")

def create_detailed_bounding_box(floor_plan):
    """
    Create a detailed bounding box based on the edges of the rooms in the floor plan.

    Args:
        floor_plan (dict): A dictionary containing room geometries.

    Returns:
        Polygon: A detailed bounding box polygon.
    """
    all_geometries = []
    for _, value in floor_plan.items():
        if isinstance(value, MultiPolygon):
            all_geometries.extend(value.geoms)
        elif isinstance(value, Polygon):
            all_geometries.append(value)

    if all_geometries:
        # Validate and fix invalid geometries
        valid_geometries = []
        for geom in all_geometries:
            if not geom.is_valid:
                try:
                    geom = make_valid(geom)  # Attempt to fix invalid geometry
                except Exception as e:
                    print(f"Failed to fix geometry: {e}")
                    continue
            valid_geometries.append(geom)

        # Combine all valid geometries into a single geometry
        if valid_geometries:
            combined_geometry = unary_union(valid_geometries)
            return combined_geometry.convex_hull  # Create a detailed bounding box using the convex hull
        else:
            print("No valid geometries found.")
            return None
    else:
        return None

def plot_detailed_bounding_boxes(data):
    for i, floor_plan in enumerate(data):
        print(f"Processing floor plan {i + 1}...")

        # Create a detailed bounding box
        detailed_bounding_box = create_detailed_bounding_box(floor_plan)

        if detailed_bounding_box:
            # Plot the detailed bounding box
            plt.figure()
            x, y = detailed_bounding_box.exterior.xy
            plt.plot(x, y, color='black', linewidth=2, label='Detailed Bounding Box')

            # Hide axis and labels
            plt.axis('off')

            # Show the plot instead of saving it
            # plt.show()

            # Save the plot to the output directory (commented out for now)
            plot_path = os.path.join(output_dir, f"detailed_floor_plan_{i + 1}.png")
            plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            print(f"Displayed detailed plot for floor plan {i + 1}")
        else:
            print(f"No geometries found in floor plan {i + 1}.")

if __name__ == "__main__":
    dataset_path = 'ResPlan.pkl'
    data = load_resplan_dataset(dataset_path)
    # plot_bounding_boxes(data)
    plot_detailed_bounding_boxes(data)