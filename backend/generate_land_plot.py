import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import math

def normalize_coordinates(coordinates):
    # Use the first coordinate as the origin (0, 0)
    origin_x, origin_y = coordinates[0]
    normalized = [(x - origin_x, y - origin_y) for x, y in coordinates]
    return normalized

def calculate_distance(point1, point2):
    # Calculate the Euclidean distance between two points
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def plot_coordinates(coordinates):
    # Normalize the coordinates
    normalized = normalize_coordinates(coordinates)

    # Extract x and y values
    x_values = [x for x, y in normalized]
    y_values = [y for x, y in normalized]

    # Close the shape by appending the first point at the end
    normalized.append(normalized[0])
    x_values.append(x_values[0])
    y_values.append(y_values[0])

    # Create the plot
    fig, ax = plt.subplots()
    ax.plot(x_values, y_values, color='black')  # Draw black lines connecting the points

    # Ensure equal scaling for both axes
    ax.set_aspect('equal', adjustable='datalim')

    # Label the lengths of the sides with 'm' (meters)
    for i in range(len(normalized) - 1):
        point1 = normalized[i]
        point2 = normalized[i + 1]
        mid_x = (point1[0] + point2[0]) / 2
        mid_y = (point1[1] + point2[1]) / 2
        distance = calculate_distance(point1, point2)
        ax.text(mid_x, mid_y, f"{distance:.2f} m", color='red', fontsize=8, ha='center', va='center')

    # Set white background and remove axes
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.axis('off')

    # Save the plot as an image
    plt.savefig('plot.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # Close the figure to free resources