import pickle
import matplotlib.pyplot as plt
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import networkx as nx
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# used for data visualization
def visualize_floor_plan_data(pkl_file_path):
    # Load the ResPlan.pkl file
    with open(pkl_file_path, 'rb') as file:
        data = pickle.load(file)

    if isinstance(data, list) and data:
        # Iterate through the first 5-10 floor plans
        for i, floor_plan in enumerate(data[:10]):
            print(f"Visualizing floor plan {i + 1}...")

            if 'graph' in floor_plan:
                nx_graph = floor_plan['graph']  # Extract the NetworkX graph

                # Relabel nodes with integers
                mapping = {node: i for i, node in enumerate(nx_graph.nodes)}
                relabeled_graph = nx.relabel_nodes(nx_graph, mapping)

                # Debugging: Print the graph structure
                # print("Graph structure:")
                # print(f"Number of nodes: {nx_graph.number_of_nodes()}")
                # print(f"Number of edges: {nx_graph.number_of_edges()}")
                # print("Graph nodes with attributes:", nx_graph.nodes(data=True))
                # print("Graph edges:", list(nx_graph.edges))

                # Validate and extract node features and positions
                for node in nx_graph.nodes:
                    # Use 'type' and 'area' fields for node features
                    if 'type' in nx_graph.nodes[node] and 'area' in nx_graph.nodes[node]:
                        room_type = nx_graph.nodes[node]['type']
                        room_area = nx_graph.nodes[node]['area']
                        nx_graph.nodes[node]['feature'] = [room_type, room_area]  # Combine type and area as features
                    else:
                        print(f"Node {node} is missing 'type' or 'area'. Adding default feature.")
                        nx_graph.nodes[node]['feature'] = ["unknown", 0]  # Default feature

                    if 'geometry' in nx_graph.nodes[node]:
                        # Place the node at the center of its polygon geometry
                        polygon = nx_graph.nodes[node]['geometry']
                        if isinstance(polygon, Polygon):
                            center = polygon.centroid
                            nx_graph.nodes[node]['pos'] = (center.x, center.y)
                        else:
                            print(f"Node {node} has invalid geometry. Adding random position.")
                            nx_graph.nodes[node]['pos'] = (torch.rand(1).item(), torch.rand(1).item())
                    elif 'pos' not in nx_graph.nodes[node]:
                        print(f"Node {node} is missing 'pos'. Adding random position.")
                        nx_graph.nodes[node]['pos'] = (torch.rand(1).item(), torch.rand(1).item())

                # Construct the entire floor plan with black borders for rooms and filled polygons
                for key, value in floor_plan.items():
                    if isinstance(value, MultiPolygon):
                        for polygon in value.geoms:
                            x, y = polygon.exterior.xy
                            plt.fill(x, y, edgecolor='black', alpha=0.5, label=key.capitalize())
                    elif isinstance(value, Polygon):
                        x, y = value.exterior.xy
                        plt.fill(x, y, edgecolor='black', alpha=0.5, label=key.capitalize())

                # Overlay graph nodes with their positions and label them with their types
                node_positions = {node: nx_graph.nodes[node]['pos'] for node in nx_graph.nodes}
                x, y = zip(*node_positions.values())
                node_labels = {node: nx_graph.nodes[node]['type'] for node in nx_graph.nodes if 'type' in nx_graph.nodes[node]}

                plt.scatter(x, y, c='red', s=50, zorder=3, label='Graph Nodes')

                # Annotate nodes with their types
                for node, (x_pos, y_pos) in node_positions.items():
                    label = node_labels.get(node, "unknown")
                    plt.text(x_pos, y_pos, str(label), fontsize=8, ha='right', color='black')

                # Plot graph edges
                for edge in nx_graph.edges:
                    start_pos = node_positions[edge[0]]
                    end_pos = node_positions[edge[1]]
                    plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], c='gray', linewidth=1, zorder=2)

                # Print the number of bedrooms and bathrooms for the current floor plan
                bedrooms = len(floor_plan['bedroom'].geoms) if isinstance(floor_plan.get('bedroom'), MultiPolygon) else 0
                bathrooms = len(floor_plan['bathroom'].geoms) if isinstance(floor_plan.get('bathroom'), MultiPolygon) else 0
                print(f"Floor Plan {i + 1}: {bedrooms} bedrooms, {bathrooms} bathrooms")

                plt.title(f'Floor Plan {i + 1} with Graph Overlay')
                plt.xlabel('X Position')
                plt.ylabel('Y Position')
                plt.grid(True)
                plt.show()
            else:
                print(f"No graph found in floor plan {i + 1}.")
    else:
        print("The data format is not a list or is empty.")

def prepare_data(pkl_file_path):
    # Load the ResPlan.pkl file
    with open(pkl_file_path, 'rb') as file: 
        data = pickle.load(file)

    dataset = []

    # Iterate over all floor plans
    for floor_plan in data:
        if isinstance(floor_plan, dict) and 'graph' in floor_plan:
            nx_graph = floor_plan['graph']  # Extract the NetworkX graph

            # Relabel nodes with integers
            mapping = {node: i for i, node in enumerate(nx_graph.nodes)}
            relabeled_graph = nx.relabel_nodes(nx_graph, mapping)

            # Debugging: Print the graph structure
            print("Graph structure:")
            print(f"Number of nodes: {nx_graph.number_of_nodes()}")
            print(f"Number of edges: {nx_graph.number_of_edges()}")
            print("Graph nodes with attributes:", nx_graph.nodes(data=True))
            print("Graph edges:", list(nx_graph.edges))

            # Validate and extract node features and positions
            for node in nx_graph.nodes:
                # Use 'type' and 'area' fields for node features
                if 'type' in nx_graph.nodes[node] and 'area' in nx_graph.nodes[node]:
                    room_type = nx_graph.nodes[node]['type']
                    room_area = nx_graph.nodes[node]['area']
                    nx_graph.nodes[node]['feature'] = [room_type, room_area]  # Combine type and area as features
                else:
                    print(f"Node {node} is missing 'type' or 'area'. Adding default feature.")
                    nx_graph.nodes[node]['feature'] = ["unknown", 0]  # Default feature

                if 'geometry' in nx_graph.nodes[node]:
                    # Place the node at the center of its polygon geometry
                    polygon = nx_graph.nodes[node]['geometry']
                    if isinstance(polygon, Polygon):
                        center = polygon.centroid
                        nx_graph.nodes[node]['pos'] = (center.x, center.y)
                    else:
                        print(f"Node {node} has invalid geometry. Adding random position.")
                        nx_graph.nodes[node]['pos'] = (torch.rand(1).item(), torch.rand(1).item())
                elif 'pos' not in nx_graph.nodes[node]:
                    print(f"Node {node} is missing 'pos'. Adding random position.")
                    nx_graph.nodes[node]['pos'] = (torch.rand(1).item(), torch.rand(1).item())

            # Convert the relabeled graph to PyTorch Geometric Data
            edge_index = torch.tensor(list(relabeled_graph.edges), dtype=torch.long).t().contiguous()
            node_features = torch.tensor([relabeled_graph.nodes[node].get('feature', [0]) for node in relabeled_graph.nodes], dtype=torch.float)

            graph_data = Data(x=node_features, edge_index=edge_index)

            # Extract additional data (e.g., number of bedrooms and bathrooms)
            bedrooms = len(floor_plan['bedroom'].geoms) if isinstance(floor_plan.get('bedroom'), MultiPolygon) else 0
            bathrooms = len(floor_plan['bathroom'].geoms) if isinstance(floor_plan.get('bathroom'), MultiPolygon) else 0

            # Print the number of bedrooms and bathrooms for the current floor plan
            print(f"Floor Plan {i + 1}: {bedrooms} bedrooms, {bathrooms} bathrooms")

            # Append the graph and metadata to the dataset
            dataset.append({
                'graph': graph_data,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms
            })

    return dataset

if __name__ == "__main__":
    visualize_floor_plan_data('ResPlan.pkl')
    # data = prepare_data('ResPlan.pkl')
