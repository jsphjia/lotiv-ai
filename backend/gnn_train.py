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

    # Extract one floor plan (assuming it's a list of 17000 floor plans)
    if isinstance(data, list) and len(data) > 0:
        floor_plan = data[0]  # Get the first floor plan

        # Debugging: Print the structure of the first floor plan
        print("First floor plan structure:", floor_plan)

        # Create a plot
        fig, ax = plt.subplots()
        patches = []

        # Iterate over the keys and plot geometries
        for key, value in floor_plan.items():
            if isinstance(value, (MultiPolygon, Polygon)):
                if isinstance(value, MultiPolygon):
                    for polygon in value.geoms:  # Iterate over individual polygons in MultiPolygon
                        mpl_polygon = MplPolygon(list(polygon.exterior.coords), closed=True)
                        patches.append(mpl_polygon)
                elif isinstance(value, Polygon):
                    mpl_polygon = MplPolygon(list(value.exterior.coords), closed=True)
                    patches.append(mpl_polygon)

        # Add patches to the plot
        patch_collection = PatchCollection(patches, edgecolor='black', facecolor='lightblue', alpha=0.5)
        ax.add_collection(patch_collection)

        # Overlay the graph representation if available
        if 'graph' in floor_plan:
            graph = floor_plan['graph']

            # Relabel nodes with integers for processing
            mapping = {node: i for i, node in enumerate(graph.nodes)}
            relabeled_graph = nx.relabel_nodes(graph, mapping)

            # Ensure all nodes have valid positions
            num_nodes = len(relabeled_graph.nodes)
            for node in relabeled_graph.nodes:
                if 'pos' not in relabeled_graph.nodes[node] or not isinstance(relabeled_graph.nodes[node]['pos'], (tuple, list)):
                    relabeled_graph.nodes[node]['pos'] = (torch.rand(1).item(), torch.rand(1).item())

            # Extract edges and node positions from the relabeled graph
            edge_index = torch.tensor(list(relabeled_graph.edges), dtype=torch.long).t().numpy()
            node_positions = {node: relabeled_graph.nodes[node]['pos'] for node in relabeled_graph.nodes}

            # Plot edges
            for edge in edge_index.T:
                start, end = edge
                x_coords = [node_positions[start][0], node_positions[end][0]]
                y_coords = [node_positions[start][1], node_positions[end][1]]
                ax.plot(x_coords, y_coords, 'k-', alpha=0.5)

            # Plot nodes
            x, y = zip(*node_positions.values())
            ax.scatter(x, y, c='red', s=50, zorder=3, label='Graph Nodes')

        # Set plot limits and aspect ratio
        ax.autoscale()
        ax.set_aspect('equal')
        plt.title('Floor Plan Visualization with Graph Overlay')
        plt.legend()
        plt.show()
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

if __name__ == "__main__":
    visualize_floor_plan_data('ResPlan.pkl')
    # data = prepare_data('ResPlan.pkl')
