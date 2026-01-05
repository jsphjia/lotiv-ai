import os
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import pickle

def process_png_image(image_path):
    """
    Process a PNG image to extract features using edge detection.

    Args:
        image_path (str): Path to the PNG image file.

    Returns:
        np.ndarray: Extracted image features as a numpy array.
    """
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale

    # Apply edge detection (Canny)
    edges = cv2.Canny(image, threshold1=100, threshold2=200)

    # Resize the edges image to a fixed size (e.g., 224x224)
    edges_resized = cv2.resize(edges, (224, 224))

    # Flatten the image into a 1D array
    image_features = edges_resized.flatten()

    return image_features

def generate_npy_files(image_folder, output_folder):
    """
    Generate .npy files for all PNG images in the specified folder.

    Args:
        image_folder (str): Path to the folder containing PNG images.
        output_folder (str): Path to the folder where .npy files will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)

    for image_file in os.listdir(image_folder):
        if image_file.endswith(".png"):
            image_path = os.path.join(image_folder, image_file)
            features = process_png_image(image_path)

            # Save the features as a .npy file
            output_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_features.npy")
            np.save(output_path, features)
            print(f"Processed {image_file} and saved features to {output_path}")

def prepare_gnn_input(bedrooms, bathrooms, floorplan_features):
    """
    Prepare input data for training a GNN model by combining numerical and image features.

    Args:
        bedrooms (int): Number of bedrooms.
        bathrooms (int): Number of bathrooms.
        floorplan_features (np.ndarray): Extracted features from the floorplan image.

    Returns:
        dict: Combined input data for the GNN model.
    """
    # Combine features into a single input vector
    combined_features = np.concatenate([
        [bedrooms, bathrooms],  # Numerical features without normalization
        floorplan_features  # Image features
    ])

    return {
        "features": combined_features
    }

def extract_bedrooms_bathrooms(dataset_path):
    """
    Extract the number of bedrooms and bathrooms from the dataset.

    Args:
        dataset_path (str): Path to the dataset file (e.g., ResPlan.pkl).

    Returns:
        list of dict: A list of dictionaries containing bedrooms and bathrooms for each floor plan.
    """
    with open(dataset_path, 'rb') as file:
        data = pickle.load(file)

    extracted_data = []
    for floor_plan in data:
        # Debugging: Print the keys in the floor plan to verify structure
        print("Floor plan keys:", floor_plan.keys())

        # Assuming the dataset contains keys 'bedrooms' and 'bathrooms'
        bedrooms = floor_plan.get('bedrooms', None)  # Default to None if not found
        bathrooms = floor_plan.get('bathrooms', None)  # Default to None if not found

        # Debugging: Print extracted values
        print(f"Extracted bedrooms: {bedrooms}, bathrooms: {bathrooms}")

        extracted_data.append({
            'bedrooms': bedrooms,
            'bathrooms': bathrooms
        })

    return extracted_data

if __name__ == "__main__":
    # Example usage
    dataset_path = "ResPlan.pkl"  # Replace with the actual dataset path

    # Extract bedrooms and bathrooms from the dataset
    extracted_data = extract_bedrooms_bathrooms(dataset_path)
    print("Extracted data:", extracted_data)

    # Generate .npy files for original floor plans
    image_folder = "output_plots"  # Replace with the folder containing PNG images
    output_folder = "gnn_data"  # Updated to save numpy objects in gnn_data folder
    generate_npy_files(image_folder, output_folder)

    for i, floor_plan in enumerate(extracted_data):
        bds = floor_plan['bedrooms']
        bths = floor_plan['bathrooms']
        features = np.load(os.path.join(output_folder, f"floor_plan_{i + 1}_features.npy"))

        gnn_input = prepare_gnn_input(bds, bths, features)
        print(f"Prepared GNN input for floor plan {i + 1}:", gnn_input)

    # Generate .npy files for detailed floor plans
    image_folder = "output_plots_detailed"  # Replace with the folder containing PNG images
    output_folder = "gnn_data"  # Updated to save numpy objects in gnn_data folder
    generate_npy_files(image_folder, output_folder)

    for i, floor_plan in enumerate(extracted_data):
        bds = floor_plan['bedrooms']
        bths = floor_plan['bathrooms']
        features = np.load(os.path.join(output_folder, f"floor_plan_{i + 1}_features_detailed.npy"))

        gnn_input = prepare_gnn_input(bds, bths, features)
        print(f"Prepared GNN input for floor plan {i + 1}:", gnn_input)