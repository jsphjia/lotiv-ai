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

            # Append the graph and metadata to the dataset
            dataset.append({
                'graph': graph_data,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms
            })

    return dataset

class ResidentialFloorPlanGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ResidentialFloorPlanGNN, self).__init__()
        # Define GCN layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

        # Additional layers for processing metadata inputs
        self.metadata_fc = torch.nn.Linear(3, hidden_dim)  # For bedrooms, bathrooms, and floors
        self.boundary_fc = torch.nn.Linear(hidden_dim, hidden_dim)  # For boundary image features

        # Final layers for generating floor plan and graph
        self.floor_plan_fc = torch.nn.Linear(hidden_dim, output_dim)
        self.graph_fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, metadata, boundary_image, data):
        # Process metadata (bedrooms, bathrooms, floors)
        metadata_features = F.relu(self.metadata_fc(metadata))

        # Process boundary image features
        boundary_features = F.relu(self.boundary_fc(boundary_image))

        # Combine metadata and boundary features
        combined_features = metadata_features + boundary_features

        # Process graph data
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        # Generate floor plan and graph outputs
        floor_plan_output = self.floor_plan_fc(combined_features)
        graph_output = self.graph_fc(x)

        return floor_plan_output, graph_output

# Example usage:
# Define the model
# model = ResidentialFloorPlanGNN(input_dim=2, hidden_dim=16, output_dim=3)
# metadata = torch.tensor([[3, 2, 1]])  # Example: 3 bedrooms, 2 bathrooms, 1 floor
# boundary_image = torch.rand((1, 16))  # Example boundary image features
# output_floor_plan, output_graph = model(metadata, boundary_image, data)

if __name__ == "__main__":
    visualize_floor_plan_data('ResPlan.pkl')
    # data = prepare_data('ResPlan.pkl')
