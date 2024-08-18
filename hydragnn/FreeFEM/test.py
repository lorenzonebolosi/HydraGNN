import os, json
import logging
import sys
import random

import numpy as np
from mpi4py import MPI
import argparse
#from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import StandardScaler
#from sklearn.utils import resample
#from sklearn.model_selection import train_test_split

import torch
from torch_geometric.transforms import RadiusGraph, Distance

import hydragnn
#sys.path.append(os.path.dirname(os.path.abspath(__file__))+'thesis_work')
from hydragnn.utils import comm_reduce

#import seaborn as sns

from hydragnn.utils.freefemdataset import GraphDataset
from hydragnn.utils.time_utils import Timer
from hydragnn.utils.config_utils import get_log_name_config
from hydragnn.utils.model import print_model, tensor_divide
from hydragnn.utils.pickledataset import SimplePickleWriter, SimplePickleDataset
from hydragnn.preprocess.load_data import split_dataset
from hydragnn.preprocess.utils import gather_deg
import hydragnn.utils.tracer as tr
import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))

#from raw data, get a dataset and normalize it
def __normalize_dataset(not_normalized_dataset):
    print("Start normalization")
    """Performs the normalization on Data objects and returns the normalized dataset."""
    num_node_features = 3#len(not_normalized_dataset.node_feature_dim)
    num_graph_features = 0#len(not_normalized_dataset.graph_feature_dim)

    not_normalized_dataset.minmax_graph_feature = np.full((2, num_graph_features), np.inf)
    # [0,...]:minimum values; [1,...]: maximum values
    not_normalized_dataset.minmax_node_feature = np.full((2, num_node_features), np.inf)
    not_normalized_dataset.minmax_graph_feature[1, :] *= -1
    not_normalized_dataset.minmax_node_feature[1, :] *= -1
    for data in not_normalized_dataset.dataset:
        # find maximum and minimum values for graph level features
        g_index_start = 0
        #It shouldn't use this
        for ifeat in range(num_graph_features):
            g_index_end = g_index_start + not_normalized_dataset.graph_feature_dim[ifeat]
            not_normalized_dataset.minmax_graph_feature[0, ifeat] = min(
                torch.min(data.y[g_index_start:g_index_end]),
                not_normalized_dataset.minmax_graph_feature[0, ifeat],
            )
            not_normalized_dataset.minmax_graph_feature[1, ifeat] = max(
                torch.max(data.y[g_index_start:g_index_end]),
                not_normalized_dataset.minmax_graph_feature[1, ifeat],
            )
            g_index_start = g_index_end

        # find maximum and minimum values for node level features
        n_index_start = 0
        for ifeat in range(num_node_features):
            n_index_end = n_index_start + config["Dataset"]["node_features"]["dim"][ifeat]
            not_normalized_dataset.minmax_node_feature[0, ifeat] = min(
                torch.min(data.x[:, n_index_start:n_index_end]),
                not_normalized_dataset.minmax_node_feature[0, ifeat],
            )
            not_normalized_dataset.minmax_node_feature[1, ifeat] = max(
                torch.max(data.x[:, n_index_start:n_index_end]),
                not_normalized_dataset.minmax_node_feature[1, ifeat],
            )
            n_index_start = n_index_end
    print("Max and min values found")
    print("Minmax node feature: ", not_normalized_dataset.minmax_node_feature)
    ## Gather minmax in parallel
    not_normalized_dataset.minmax_graph_feature[0, :] = comm_reduce(
        not_normalized_dataset.minmax_graph_feature[0, :], torch.distributed.ReduceOp.MIN
    )
    not_normalized_dataset.minmax_graph_feature[1, :] = comm_reduce(
        not_normalized_dataset.minmax_graph_feature[1, :], torch.distributed.ReduceOp.MAX
    )
    not_normalized_dataset.minmax_node_feature[0, :] = comm_reduce(
        not_normalized_dataset.minmax_node_feature[0, :], torch.distributed.ReduceOp.MIN
    )
    not_normalized_dataset.minmax_node_feature[1, :] = comm_reduce(
        not_normalized_dataset.minmax_node_feature[1, :], torch.distributed.ReduceOp.MAX
    )

    for data in not_normalized_dataset.dataset:
        g_index_start = 0
        for ifeat in range(num_graph_features):
            g_index_end = g_index_start + not_normalized_dataset.graph_feature_dim[ifeat]
            data.y[g_index_start:g_index_end] = tensor_divide(
                (
                    data.y[g_index_start:g_index_end]
                    - not_normalized_dataset.minmax_graph_feature[0, ifeat]
                ),
                (
                    not_normalized_dataset.minmax_graph_feature[1, ifeat]
                    - not_normalized_dataset.minmax_graph_feature[0, ifeat]
                ),
            )
            g_index_start = g_index_end
        n_index_start = 0
        for ifeat in range(num_node_features):
            n_index_end = n_index_start + config["Dataset"]["node_features"]["dim"][ifeat]
            data.x[:, n_index_start:n_index_end] = tensor_divide(
                (
                    data.x[:, n_index_start:n_index_end]
                    - not_normalized_dataset.minmax_node_feature[0, ifeat]
                ),
                (
                    not_normalized_dataset.minmax_node_feature[1, ifeat]
                    - not_normalized_dataset.minmax_node_feature[0, ifeat]
                ),
            )
            n_index_start = n_index_end
    print("Normalization done")



def visualize_graph(edge_index, x):
    # Create a PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index)

    # Convert to NetworkX graph
    def to_networkx(data, node_attrs=None, edge_attrs=None):
        G = nx.DiGraph() if data.is_directed() else nx.Graph()

        node_attrs = [] if node_attrs is None else node_attrs
        edge_attrs = [] if edge_attrs is None else edge_attrs

        G.add_nodes_from(range(data.num_nodes))
        for key in node_attrs:
            for i, val in enumerate(data[key]):
                G.nodes[i][key] = val

        edges = data.edge_index.t().tolist()
        for i, (u, v) in enumerate(edges):
            G.add_edge(u, v)
            for key in edge_attrs:
                G[u][v][key] = data[key][i]

        return G

    G = to_networkx(data)

    # Plot the graph
    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_size=700, node_color='skyblue', font_size=16, font_color='black', font_weight='bold')
    plt.show()



def plot_iterations(data_objects, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, data in enumerate(data_objects):
        plt.figure(figsize=(10, 8))

        # Extracting the node features
        u1 = data.x[:, 0]
        u2 = data.x[:, 1]
        w = data.x[:, 2]

        # Extracting the node positions
        pos_x = data.pos[:, 0]
        pos_y = data.pos[:, 1]

        # Creating a scatter plot for node features w
        plt.subplot(3, 1, 1)
        plt.scatter(pos_x, pos_y, c=w, cmap='viridis', s=50, edgecolor='k')
        plt.colorbar(label='w')
        plt.xlabel('x-coordinate')
        plt.ylabel('y-coordinate')
        #plt.title(f'Iteration {i + 1}: Node features w, Compliance: ' + str(data.y.item()))

        # Creating a scatter plot for u1
        plt.subplot(3, 1, 2)
        plt.scatter(pos_x, pos_y, c=u1, cmap='coolwarm', s=50, edgecolor='k')
        plt.colorbar(label='u1')
        plt.xlabel('x-coordinate')
        plt.ylabel('y-coordinate')
        plt.title(f'Iteration {i + 1}: Displacement field u1')

        # Creating a scatter plot for u2
        plt.subplot(3, 1, 3)
        plt.scatter(pos_x, pos_y, c=u2, cmap='coolwarm', s=50, edgecolor='k')
        plt.colorbar(label='u2')
        plt.xlabel('x-coordinate')
        plt.ylabel('y-coordinate')
        plt.title(f'Iteration {i + 1}: Displacement field u2')

        plt.tight_layout()
        # Save the plot
        filename = os.path.join(output_dir, f'iteration_{i + 1}.png')
        plt.savefig(filename)
        plt.close()


def normalize_columns(x):
    """
    Normalize each column of the tensor so that the values in each column range from 0 to 1.

    Parameters:
    x (torch.Tensor): A tensor of shape (451, 3) to be normalized.

    Returns:
    torch.Tensor: The normalized tensor with values in each column ranging from 0 to 1.
    """
    x_min = x.min(dim=0, keepdim=True).values  # Minimum value of each column
    x_max = x.max(dim=0, keepdim=True).values  # Maximum value of each column

    # Normalize each column
    x_normalized = (x - x_min) / (x_max - x_min)

    return x_normalized


def plot_matrix_densities(objects_list):
        """
        Plots the density distributions for the three columns of matrices stored in the 'x' attribute
        of a list of objects using Matplotlib.

        Parameters:
        objects_list (list): A list of objects, each with an attribute 'x' containing a matrix.
        """
        # Initialize lists to store data from each column
        column_1_data = []
        column_2_data = []
        column_3_data = []

        # Extract data from each object
        for obj in objects_list:
            matrix = getattr(obj, 'x', None)  # Safely get the 'x' attribute, or None if it doesn't exist
            if matrix is not None:
                #and matrix.shape == (451, 3):  # Check if the matrix is valid and has the correct shape
                column_1_data.extend(matrix[:, 0])  # Extract the first column
                column_2_data.extend(matrix[:, 1])  # Extract the second column
                column_3_data.extend(matrix[:, 2])  # Extract the third column
            else:
                print(f"Skipping an object due to invalid or missing matrix.")

        # Now plot the densities using Matplotlib
        plt.figure(figsize=(15, 5))

        # Plot density for column 1
        plt.subplot(1, 3, 1)
        plt.hist(column_1_data, bins=30, density=True, alpha=0.6, color='blue')
        plt.title('Density Plot for ux')
        plt.xlabel('Value')
        plt.ylabel('Density')

        #Plot density for column 2
        plt.subplot(1, 3, 2)
        plt.hist(column_2_data, bins=30, density=True, alpha=0.6, color='green')
        plt.title('Density Plot for uy')
        plt.xlabel('Value')
        plt.ylabel('Density')

        # Plot density for column 3
        plt.subplot(1, 3, 3)
        plt.hist(column_3_data, bins=30, density=True, alpha=0.6, color='red')
        plt.title('Density Plot for w')
        plt.xlabel('Value')
        plt.ylabel('Density')

        plt.tight_layout()
        # Save the plot
        filename = os.path.join('plots', 'densities.png')
        plt.savefig(filename)
        plt.close()


def plot_separate_densities(objects_list):
    """
    Plots the density distributions for the three columns of matrices stored in the 'x' attribute
    of a list of objects using Matplotlib. Each plot is saved as 'densities+i.png' where i is the iteration number.

    Parameters:
    objects_list (list): A list of objects, each with an attribute 'x' containing a matrix.
    """
    # Create a directory to store the plots if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Iterate over each object in the list
    for i, obj in enumerate(objects_list, start=1):
        matrix = getattr(obj, 'x', None)  # Safely get the 'x' attribute, or None if it doesn't exist

        if matrix is not None:
            plt.figure(figsize=(15, 5))

            # Plot density for column 1
            plt.subplot(1, 3, 1)
            plt.hist(matrix[:, 0], bins=30, density=True, alpha=0.6, color='blue')
            plt.title('Density Plot for ux')
            plt.xlabel('Value')
            plt.ylabel('Density')

            # Plot density for column 2
            plt.subplot(1, 3, 2)
            plt.hist(matrix[:, 1], bins=30, density=True, alpha=0.6, color='green')
            plt.title('Density Plot for uy')
            plt.xlabel('Value')
            plt.ylabel('Density')

            # Plot density for column 3
            plt.subplot(1, 3, 3)
            plt.hist(matrix[:, 2], bins=30, density=True, alpha=0.6, color='red')
            plt.title('Density Plot for w')
            plt.xlabel('Value')
            plt.ylabel('Density')

            plt.tight_layout()

            # Save the plot with a unique filename
            filename = os.path.join('plots', f'densities_{i}.png')
            plt.savefig(filename)
            plt.close()
        else:
            print(f"Skipping object {i} due to invalid or missing matrix.")
#I want to normalize all for making it gaussian
def gaussian_normalize_matrices(objects, transform=None):
    """
    Normalize the first two columns of the 'x' attribute (3x451 matrices) of each object
    in the 'objects' list independently to have a Gaussian-like distribution using Z-score normalization.

    Parameters:
    - objects: list of objects, each containing an attribute 'x' that is a 3x451 matrix.
    - transform: str or None, optional (default=None). If 'boxcox', apply Box-Cox transformation
      before normalization. If 'yeojohnson', apply Yeo-Johnson transformation.

    Returns:
    - means: list of two elements, the means of the first two columns.
    - stds: list of two elements, the standard deviations of the first two columns.
    - lmbdas: list of two elements, the lambda values for the transformations (if applied).
    - transforms: list of two elements, the type of transformation applied to each column.
    """
    means = [None, None]
    stds = [None, None]
    lmbdas = [None, None]
    transforms = [None, None]

    for col in range(2):  # Only normalize the first two columns
        # Step 1: Flatten the column data across all matrices and concatenate into one tensor
        col_data = torch.cat([obj.x[:, col].flatten() for obj in objects])

        # Convert tensor to numpy array for transformation
        col_data_np = col_data.numpy()

        # Step 2: Optionally apply a Gaussian transformation (e.g., Box-Cox or Yeo-Johnson)
        transforms[col] = transform
        if transform == 'boxcox':
            col_data_np, lmbdas[col] = stats.boxcox(col_data_np + np.abs(np.min(col_data_np)) + 1)
        elif transform == 'yeojohnson':
            col_data_np, lmbdas[col] = stats.yeojohnson(col_data_np)

        # Step 3: Calculate and store mean and std for reverting
        means[col] = np.mean(col_data_np)
        stds[col] = np.std(col_data_np)

        # Step 4: Normalize using Z-Score (Standardization)
        normalized_data_np = (col_data_np - means[col]) / stds[col]

        # Convert back to tensor
        normalized_data = torch.tensor(normalized_data_np, dtype=col_data.dtype)

        # Step 5: Reshape normalized data back to the original shape and assign it to the appropriate column
        start = 0
        for obj in objects:
            num_elements = obj.x.shape[0]  # Number of rows (451)
            obj.x[:, col] = normalized_data[start:start + num_elements].reshape(obj.x[:, col].shape)
            start += num_elements
    return means, stds, lmbdas, transforms


# def apply_smoter(objects, random_state=42):
#     """
#     Apply SMOTER to balance the dataset in a list of objects.
#
#     Parameters:
#     - objects: list of ExampleObject instances. Each object should have 'x' as a (451, 3) tensor, with columns 0 and 1 as target features and column 2 as input features.
#     - random_state: int, optional (default=42). The random seed for SMOTER.
#
#     Returns:
#     - Updated list of objects with resampled data.
#     """
#     # Step 1: Combine tensors from the list of objects into a single dataset
#     X_list = []
#     y_list = []
#
#     for obj in objects:
#         X_list.append(obj.x[:, 2].numpy())  # Extract the input feature (column 2) and convert to numpy array
#         y_list.append(obj.x[:, :2].numpy())  # Extract target features (columns 0 and 1) and convert to numpy array
#
#     X_combined = np.concatenate(X_list)  # Combine all input features
#     y_combined = np.concatenate(y_list)  # Combine all target features
#
#     # Step 2: Normalize the input feature data
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_combined.reshape(-1, 1))  # Reshape to 2D for scaling
#
#     # Step 3: Apply SMOTER for regression
#     smoter = SMOTER(random_state=random_state)
#     X_resampled, y_resampled = smoter.fit_resample(X_scaled, y_combined)
#
#     # Step 4: Update the objects with resampled data
#     num_samples = len(X_resampled)
#     num_objects = len(objects)
#     rows_per_object = num_samples // num_objects
#
#     for i, obj in enumerate(objects):
#         start_idx = i * rows_per_object
#         end_idx = (i + 1) * rows_per_object
#         # Reshape back to original dimensions
#         X_resampled_tensor = torch.tensor(X_resampled[start_idx:end_idx], dtype=torch.float32).reshape(-1, 1)
#         y_resampled_tensor = torch.tensor(y_resampled[start_idx:end_idx], dtype=torch.float32)
#         # Combine the input feature and target features into one tensor
#         obj.x = torch.cat((y_resampled_tensor, X_resampled_tensor), dim=1)
#
#     return objects

# def apply_random_oversampling(objects, target_rows=451, random_state=42):
#     """
#     Apply global resampling to balance the dataset across all objects to create a more uniform distribution of target values.
#
#     Parameters:
#     - objects: list of ExampleObject instances. Each object should have 'x' as a (451, 3) tensor, with columns 0 and 1 as target features and column 2 as input features.
#     - target_rows: int, optional (default=451). The desired number of rows for each object after resampling.
#     - random_state: int, optional (default=42). The random seed for resampling.
#
#     Returns:
#     - Updated list of objects with resampled data.
#     """
#     # Step 1: Combine data from all objects
#     X_list = []
#     y_list = []
#
#     for obj in objects:
#         X_list.append(obj.x[:, 2].numpy())  # Extract input feature (column 2)
#         y_list.append(obj.x[:, :2].numpy())  # Extract target features (columns 0 and 1)
#
#     X_combined = np.concatenate(X_list)  # Combined input features across all objects
#     y_combined = np.concatenate(y_list)  # Combined target features across all objects
#
#     # Step 2: Normalize the input feature data
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_combined.reshape(-1, 1))  # Normalize input features
#
#     # Step 3: Combine the normalized input features with the targets
#     combined = np.hstack((y_combined, X_scaled))
#
#     # Step 4: Stratify by target values
#     stratified_data = []
#     tmp = len(np.unique(y_combined, axis=0))
#     num_samples_per_stratum = len(objects) * target_rows // len(np.unique(y_combined, axis=0))
#
#     for target_value in np.unique(y_combined, axis=0):
#         mask = np.all(y_combined == target_value, axis=1)
#         stratum = combined[mask]
#
#         # Step 5: Resample each stratum
#         resampled_stratum = resample(stratum, replace=True, n_samples=num_samples_per_stratum,
#                                      random_state=random_state)
#         stratified_data.append(resampled_stratum)
#
#     # Step 6: Combine stratified, resampled data
#     combined_resampled = np.vstack(stratified_data)
#
#     # Step 7: Redistribute the resampled data back to the objects
#     for i, obj in enumerate(objects):
#         start_idx = i * target_rows
#         end_idx = start_idx + target_rows
#         obj_data = combined_resampled[start_idx:end_idx]
#
#         # Split back into y and X
#         y_resampled = obj_data[:, :2]
#         X_resampled = obj_data[:, 2:]
#
#         # Convert back to tensors and update the object
#         y_resampled_tensor = torch.tensor(y_resampled, dtype=torch.float32)
#         X_resampled_tensor = torch.tensor(X_resampled, dtype=torch.float32)
#
#         obj.x = torch.cat((y_resampled_tensor, X_resampled_tensor), dim=1)

def plot_density_map(matrix, method='histogram', bins=50, cmap='Blues', bw_adjust=0.5):
    """
    Plots the density of points based on the first two columns of a matrix.

    Parameters:
    - matrix (numpy.ndarray): A 2D array where the first two columns represent x and y coordinates.
    - method (str): The method to visualize density. Options are 'histogram' or 'kde'.
    - bins (int): Number of bins for the histogram (only used if method='histogram').
    - cmap (str): The color map to use for the plot.
    - bw_adjust (float): Bandwidth adjustment for KDE plot (only used if method='kde').
    """

    # Extract the first two columns (x and y coordinates)
    x = np.array(matrix[:, 0])
    y = np.array(matrix[:, 1])

    # Plot based on the selected method
    plt.figure(figsize=(10, 8))

    if method == 'histogram':
        # 2D Histogram (Heatmap)
        plt.hist2d(x, y, bins=bins, cmap=cmap)
        plt.colorbar(label='Number of points')
        plt.xlabel('X position')
        plt.ylabel('Y position')
        plt.title('Density of Points on Map (2D Histogram)')


    else:
        raise ValueError("Method must be either 'histogram' or 'kde'.")

    plt.tight_layout()
    # Save the plot
    filename = os.path.join('plots', 'densities_map.png')
    plt.savefig(filename)
    plt.close()


def extract_elements_from_bins(matrix, bins=30):
    """
    Extracts one element from each populated bin after dividing the plot into bins.

    Parameters:
    - matrix (numpy.ndarray): A 2D array where the first two columns represent x and y coordinates.
    - bins (int): Number of bins along each axis.

    Returns:
    - selected_points (numpy.ndarray): An array of selected points, one from each populated bin.
    """

    # Extract the first two columns (x and y coordinates)
    matrix = np.array(matrix)
    x = matrix[:, 0]
    y = matrix[:, 1]

    # Calculate the 2D histogram bins
    hist, x_edges, y_edges = np.histogram2d(x, y, bins=bins)

    selected_points = []

    # Iterate over each bin
    for i in range(bins):
        for j in range(bins):
            # Identify points within the current bin
            x_bin_mask = (x >= x_edges[i]) & (x < x_edges[i + 1])
            y_bin_mask = (y >= y_edges[j]) & (y < y_edges[j + 1])
            bin_points = matrix[x_bin_mask & y_bin_mask]

            if bin_points.shape[0] > 0:
                # Select one point from this bin
                selected_point = bin_points[0]  # You can choose a different strategy here
                selected_points.append(selected_point)

    # Convert the selected points list to a numpy array
    selected_points = np.array(selected_points)

    return torch.tensor(selected_points)


def select_with_distances(graph, distance_threshold = 0.0001):
    """
    Selects a uniform subset of points based on the distance between points in two columns,
    then extends the selection to the other columns, and plots the selected points.

    Parameters:
    - matrix (numpy.ndarray): A matrix with shape (n, d).
    - col_indices (tuple): A tuple of two integers representing the indices of the two columns to use for selection.
    - distance_threshold (float): Minimum allowed distance between any two selected points.

    Returns:
    - selected_matrix (numpy.ndarray): The subset of the matrix corresponding to the selected points.
    """
    # Extract the columns to be used for selection
    # Extract first two columns from graph.x and store as an np array
    selected_columns = np.array(graph.x[:, :2])

    # Initialize the list of selected indices
    selected_indices = []

    # Iterate over each point in the selected columns
    for i, point in enumerate(selected_columns):
        if len(selected_indices) == 0:
            selected_indices.append(i)
        else:
            distances = np.linalg.norm(selected_columns[selected_indices] - point, axis=1)
            if np.all(distances >= distance_threshold):
                selected_indices.append(i)

    # Filter the original matrix based on the selected indices
    selected_graph = Data()
    selected_graph.x = graph.x[selected_indices, :]
    selected_graph.pos = graph.pos[selected_indices, :]
    create_graph_fromXYZ = RadiusGraph(r=5, max_num_neighbors=50)
    compute_edge_lengths = Distance(norm=False, cat=True)
    # first_data.edge_index = read_edges()
    selected_graph = create_graph_fromXYZ(selected_graph)
    selected_graph = compute_edge_lengths(selected_graph)
    plot_matrix_densities([selected_graph])
    return selected_graph

def plot_columns_independently(matrix):
    """
    Plots each column of a matrix independently as a function of the row index.

    Parameters:
    - matrix (numpy.ndarray): A matrix with shape (n, d), where each column is plotted independently.
    """
    # Get the number of rows and columns in the matrix
    n_rows, n_cols = matrix.shape

    # Generate the x-axis values (assuming the row indices as x values)
    x_values = np.arange(n_rows)

    # Create a plot with a subplot for each column
    plt.figure(figsize=(4 * n_cols, 6))

    for i in range(n_cols):
        plt.subplot(1, n_cols, i + 1)
        plt.scatter(x_values, matrix[:, i], label=f'Column {i + 1}', color='b', s=5)
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title(f'Column {i + 1}')
        plt.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Save the plot
    filename = os.path.join('plots', 'values_distribution.png')
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    #exec(os.path.dirname(os.path.abspath(__file__))+ "/freefemdataset.py")
    # Get the current working directory
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    print("Current working directory:", ROOT_DIR)
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, help="batch_size", default=None)
    parser.add_argument(
        "--loadexistingsplit",
        action="store_true",
        help="loading from existing pickle/adios files with train/test/validate splits",
    )
    parser.add_argument(
        "--preonly",
        action="store_true",
        help="preprocess only. Adios or pickle saving and no train",
    )
    parser.add_argument("--inputfile", help="input file", type=str, default="custom.json")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--pickle",
        help="Pickle dataset",
        action="store_const",
        dest="format",
        const="pickle",
    )
    parser.set_defaults(format="pickle")

    args = parser.parse_args()
    dirpwd = os.path.dirname(os.path.abspath(__file__))
    input_filename = os.path.join(dirpwd, args.inputfile)
    with open(input_filename, "r") as f:
        config = json.load(f)
    hydragnn.utils.setup_log(get_log_name_config(config))
    ##################################################################################################################
    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.setup_ddp()
    ##################################################################################################################
    comm = MPI.COMM_WORLD
    ## Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%%(levelname)s (rank %d): %%(message)s" % (rank),
        datefmt="%H:%M:%S",
    )

    datasetname = config["Dataset"]["name"]
    config["Dataset"]["name"] = "%s_%d" % (datasetname, rank)
    modelname = "FreeFEM"
    node_feature_names = ["ux", "uy", "w"]
    graph_feature_names = []
    graph_feature_dims = []
    node_feature_dims = [1, 1, 1]
    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(dirpwd, "../../hydragnn/utils/thesis_work/dataset")
    # Configurable run choices (JSON file that accompanies this example script).
    with open(input_filename, "r") as f:
        config = json.load(f)
    verbosity = config["Verbosity"]["level"]
    var_config = config["NeuralNetwork"]["Variables_of_interest"]
    var_config["graph_feature_names"] = graph_feature_names
    var_config["graph_feature_dims"] = graph_feature_dims
    var_config["node_feature_names"] = node_feature_names
    var_config["node_feature_dims"] = node_feature_dims
    if args.batch_size is not None:
        config["NeuralNetwork"]["Training"]["batch_size"] = args.batch_size

    if not args.loadexistingsplit:
        radius = config["NeuralNetwork"]["Architecture"]["radius"]
        max_neighbours = config["NeuralNetwork"]["Architecture"]["max_neighbours"]
        total = GraphDataset(
            os.path.dirname(os.path.abspath(__file__))+"/freeFEM_results/", radius, max_neighbours)  # dirpwd + "/dataset/VASP_calculations/binaries", config, dist=True)
        #__normalize_dataset(total)
        data_objects = total.dataset  # Replace with your actual data objects
        #plot_iterations(data_objects, 'plots')
        #plot_matrix_densities(total.dataset)
        #total.dataset = apply_random_oversampling(total.dataset)
        #gaussian_normalize_matrices(total.dataset, transform='yeojohnson')
        #total.dataset = (total.dataset)
        #plot_matrix_densities(total.dataset)
        ##########
        # This is for testing purposes, with a single graph
        #random_graph = total.dataset[10]
        #plot_columns_independently(random_graph.x)
        #random_graph = select_with_distances(random_graph)
        #plot_columns_independently(random_graph.x)
        #plot_density_map(random_graph.x, bins = 100)
        #random_graph.x = extract_elements_from_bins(random_graph.x, 100)
        #plot_density_map(random_graph.x)
        #random_graph.x = normalize_columns(random_graph.x)
        plot_matrix_densities(total.dataset)
        total.dataset = [select_with_distances(graph) for graph in total.dataset]
        plot_separate_densities(total.dataset)
        plot_matrix_densities(total.dataset)
        #plot_iterations([random_graph], 'plots')

        #random_graph.x = torch.unique(random_graph.x[:,0], dim=0)
        #total.dataset = [random_graph] * 4000
        #plot_matrix_densities(total.dataset)
        #random_graph.x = torch.rand(451, 3)
        #total.dataset = [random_graph] * 4000
        ##########


        # Example usage with a list of data objects
        data_objects = [total.dataset[0]]  # Replace with your actual data objects
        plot_iterations(data_objects, 'plots')
        #plot_matrix_densities(total.dataset)
        print(len(total))
        trainset, valset, testset = split_dataset(
            dataset=total,
            perc_train=config["NeuralNetwork"]["Training"]["perc_train"],
            stratify_splitting=False,
        )

        print(len(total), len(trainset), len(valset), len(testset))
        deg = gather_deg(trainset)
        config["pna_deg"] = deg
        setnames = ["trainset", "valset", "testset"]
        ## pickle
        if args.format == "pickle":
            basedir = os.path.join(os.getcwd()+ "/dataset", "%s.pickle" % modelname)

            attrs = dict()
            attrs["pna_deg"] = deg
            SimplePickleWriter(
                trainset,
                basedir,
                "trainset",
                #minmax_node_feature=total.minmax_node_feature,
                #minmax_graph_feature=total.minmax_graph_feature,
                use_subdir=True,
                attrs=attrs,
            )
            SimplePickleWriter(
                valset,
                basedir,
                "valset",
                #minmax_node_feature=total.minmax_node_feature,
                #minmax_graph_feature=total.minmax_graph_feature,
                use_subdir=True,
            )
            SimplePickleWriter(
                testset,
                basedir,
                "testset",
                #minmax_node_feature=total.minmax_node_feature,
                #minmax_graph_feature=total.minmax_graph_feature,
                use_subdir=True,
            )
        sys.exit(0)

    tr.initialize()
    tr.disable()
    timer = Timer("load_data")
    timer.start()

    timer = Timer("load_data")
    timer.start()

    if args.format == "pickle":
        info("Pickle load")
        basedir = os.path.join(os.getcwd()+ "/dataset", "%s.pickle" % modelname)
        trainset = SimplePickleDataset(basedir=basedir, label="trainset", var_config=var_config)
        valset = SimplePickleDataset(basedir=basedir, label="valset", var_config=var_config)
        testset = SimplePickleDataset(basedir=basedir, label="testset", var_config=var_config)

        #Print after reading from pickle
        #data_objects = [trainset[0]]  # Replace with your actual data objects
        #plot_iterations(data_objects, 'plots_after_pkl_read')
        minmax_node_feature = trainset.minmax_node_feature
        minmax_graph_feature = trainset.minmax_graph_feature
        pna_deg = trainset.pna_deg
    else:
        raise ValueError("Unknown data format: %d" % args.format)
    ## Set minmax
    config["NeuralNetwork"]["Variables_of_interest"][
        "minmax_node_feature"
    ] = trainset.minmax_node_feature
    config["NeuralNetwork"]["Variables_of_interest"][
        "minmax_graph_feature"
    ] = trainset.minmax_graph_feature

    info(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
    )

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )
    timer.stop()

    config = hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)
    config["NeuralNetwork"]["Variables_of_interest"].pop("minmax_node_feature", None)
    config["NeuralNetwork"]["Variables_of_interest"].pop("minmax_graph_feature", None)

    verbosity = config["Verbosity"]["level"]
    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )
    if rank == 0:
        print_model(model)
    comm.Barrier()

    model = hydragnn.utils.get_distributed_model(model, verbosity)

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    log_name = get_log_name_config(config)
    writer = hydragnn.utils.get_summary_writer(log_name)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    hydragnn.utils.save_config(config, log_name)

    hydragnn.train.train_validate_test(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        writer,
        scheduler,
        config["NeuralNetwork"],
        log_name,
        verbosity,
        create_plots=True,
        plot_hist_solution=True,
    )

    hydragnn.utils.save_model(model, optimizer, log_name)
    hydragnn.utils.print_timers(verbosity)

    sys.exit(0)
