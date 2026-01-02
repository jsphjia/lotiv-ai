import os
from torchvision import models, transforms
from PIL import Image
import torch
import numpy as np
from torchvision.transforms.functional import to_tensor
import cv2

def process_png_image(image_path):
    """
    Process a PNG image to extract features using a pre-trained ResNet model.
    Applies edge detection to emphasize the bounding box.

    Args:
        image_path (str): Path to the PNG image file.

    Returns:
        np.ndarray: Extracted image features as a numpy array.
    """
    # Load pre-trained ResNet
    resnet = models.resnet18(pretrained=True)
    resnet.fc = torch.nn.Identity()  # Remove the classification layer

    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale

    # Apply edge detection (Canny)
    edges = cv2.Canny(image, threshold1=100, threshold2=200)

    # Convert edges to 3-channel image for ResNet
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    # Convert to PIL Image for torchvision transforms
    edges_pil = Image.fromarray(edges_rgb)

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_tensor = transform(edges_pil).unsqueeze(0)

    # Extract features
    with torch.no_grad():
        image_features = resnet(image_tensor).squeeze().numpy()

    return image_features

if __name__ == "__main__":
    # Example usage
    image_folder = "output_plots"  # Replace with the folder containing PNG images
    output_folder = "processed_images"
    os.makedirs(output_folder, exist_ok=True)

    for image_file in os.listdir(image_folder):
        if image_file.endswith(".png"):
            image_path = os.path.join(image_folder, image_file)
            features = process_png_image(image_path)

            # Save the features as a .npy file
            output_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_features.npy")
            np.save(output_path, features)
            print(f"Processed {image_file} and saved features to {output_path}")