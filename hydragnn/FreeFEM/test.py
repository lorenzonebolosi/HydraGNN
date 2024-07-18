import os, json
import logging
import sys

import numpy as np
from mpi4py import MPI
import argparse

import torch

import hydragnn
#sys.path.append(os.path.dirname(os.path.abspath(__file__))+'thesis_work')
from hydragnn.utils import comm_reduce

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
        plt.scatter(pos_x, pos_y, c=w, cmap='viridis', s=10, edgecolor='k')
        plt.colorbar(label='w')
        plt.xlabel('x-coordinate')
        plt.ylabel('y-coordinate')
        plt.title(f'Iteration {i + 1}: Node features w, Compliance: ' + str(data.y.item()))

        # Creating a scatter plot for u1
        plt.subplot(3, 1, 2)
        plt.scatter(pos_x, pos_y, c=u1, cmap='coolwarm', s=10, edgecolor='k')
        plt.colorbar(label='u1')
        plt.xlabel('x-coordinate')
        plt.ylabel('y-coordinate')
        plt.title(f'Iteration {i + 1}: Displacement field u1')

        # Creating a scatter plot for u2
        plt.subplot(3, 1, 3)
        plt.scatter(pos_x, pos_y, c=u2, cmap='coolwarm', s=10, edgecolor='k')
        plt.colorbar(label='u2')
        plt.xlabel('x-coordinate')
        plt.ylabel('y-coordinate')
        plt.title(f'Iteration {i + 1}: Displacement field u2')

        plt.tight_layout()
        # Save the plot
        filename = os.path.join(output_dir, f'iteration_{i + 1}.png')
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
        total = GraphDataset(
            os.path.dirname(os.path.abspath(__file__))+"/freeFEM_results/")  # dirpwd + "/dataset/VASP_calculations/binaries", config, dist=True)
        #__normalize_dataset(total)
        # Example usage with a list of data objects
        data_objects = total  # Replace with your actual data objects
        #plot_iterations(data_objects, 'plots')
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
                # minmax_node_feature=total.minmax_node_feature,
                # minmax_graph_feature=total.minmax_graph_feature,
                use_subdir=True,
                attrs=attrs,
            )
            SimplePickleWriter(
                valset,
                basedir,
                "valset",
                # minmax_node_feature=total.minmax_node_feature,
                # minmax_graph_feature=total.minmax_graph_feature,
                use_subdir=True,
            )
            SimplePickleWriter(
                testset,
                basedir,
                "testset",
                # minmax_node_feature=total.minmax_node_feature,
                # minmax_graph_feature=total.minmax_graph_feature,
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


        # minmax_node_feature = trainset.minmax_node_feature
        # minmax_graph_feature = trainset.minmax_graph_feature
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
    )

    hydragnn.utils.save_model(model, optimizer, log_name)
    hydragnn.utils.print_timers(verbosity)

    sys.exit(0)
