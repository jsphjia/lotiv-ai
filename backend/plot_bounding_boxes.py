import pickle
import matplotlib.pyplot as plt
from shapely.geometry import MultiPolygon, Polygon, box

# Load the ResPlan.pkl file
def load_resplan_dataset(pkl_file_path):
    with open(pkl_file_path, 'rb') as file:
        data = pickle.load(file)
    return data

# Plot bounding boxes for each floor plan
def plot_bounding_boxes(data):
    for i, floor_plan in enumerate(data):
        print(f"Processing floor plan {i + 1}...")

        # Combine all geometries in the floor plan to calculate the bounding box
        all_geometries = []
        for key, value in floor_plan.items():
            if isinstance(value, MultiPolygon):
                all_geometries.extend(value.geoms)
            elif isinstance(value, Polygon):
                all_geometries.append(value)

        if all_geometries:
            # Calculate the bounding box
            combined = MultiPolygon(all_geometries)
            minx, miny, maxx, maxy = combined.bounds
            bounding_box = box(minx, miny, maxx, maxy)

            # USED FOR DEBUGGING PURPOSES
            # Plot the floor plan geometries
            # plt.figure()
            # for geom in all_geometries:
            #     if isinstance(geom, Polygon):
            #         x, y = geom.exterior.xy
            #         plt.plot(x, y, color='blue', linewidth=1, label='Floor Plan Geometry')

            # Plot the bounding box
            x, y = bounding_box.exterior.xy
            plt.plot(x, y, color='black', linewidth=2, label='Bounding Box')

            # Set plot title and labels
            plt.title(f'Bounding Box and Floor Plan {i + 1}')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            print(f"No geometries found in floor plan {i + 1}.")

if __name__ == "__main__":
    # Replace 'ResPlan.pkl' with the actual path to your dataset
    dataset_path = 'ResPlan.pkl'
    data = load_resplan_dataset(dataset_path)
    plot_bounding_boxes(data)