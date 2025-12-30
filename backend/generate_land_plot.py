import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

def normalize_coordinates(coordinates):
    # Use the first coordinate as the origin (0, 0)
    origin_x, origin_y = coordinates[0]
    normalized = [(x - origin_x, y - origin_y) for x, y in coordinates]
    return normalized

def plot_coordinates(coordinates):
    # Normalize the coordinates
    normalized = normalize_coordinates(coordinates)

    # Extract x and y values
    x_values = [x for x, y in normalized]
    y_values = [y for x, y in normalized]

    # Close the shape by appending the first point at the end
    x_values.append(x_values[0])
    y_values.append(y_values[0])

    # Create the plot
    fig, ax = plt.subplots()
    ax.plot(x_values, y_values, color='black')  # Draw black lines connecting the points

    # Set white background and remove axes
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.axis('off')

    # Save the plot as an image
    plt.savefig('plot.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # Close the figure to free resources