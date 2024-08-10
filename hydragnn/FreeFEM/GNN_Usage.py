

#Look at inference.py for the usage of a pre-trained model

##############################################################################
# Copyright (c) 2021, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

##############################################################################
# I want the neural network to be continuely running and updating the displacement
# whenever new values are written on the distribution file
##############################################################################
import json, os
#import tensorflow as tf
#tfk = tf.keras
import sys
import logging
import pickle
import time as t
from datetime import datetime

from torch_geometric.data import Data
from torch_geometric.transforms import RadiusGraph, Distance
from tqdm import tqdm
from mpi4py import MPI
import argparse

import torch
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import hydragnn
from hydragnn.utils.time_utils import Timer
from hydragnn.utils.distributed import get_device
from hydragnn.utils.model import load_existing_model
from hydragnn.utils.pickledataset import SimplePickleDataset
from hydragnn.utils.config_utils import (
    update_config,
)
from hydragnn.models.create import create_model_config
from hydragnn.preprocess import create_dataloaders

from scipy.interpolate import griddata

try:
    from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 16})

def convert_w_to_tensor(iteration_number):

    ################################
    #READ AND CONVERT W VALUES
    ################################
    dirpwd = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(dirpwd, input_folder)
    with open(input_dir+"/input_iteration_"+str(iteration_number)+".txt", 'r') as f:
        w_values = f.read().split('\n')

    #Need to remove the last row that is always empty
    end_index = len(w_values) - 1
    w_values = w_values[:end_index]
    w_values = [list(map(float, line.split())) for line in w_values]
    print("Converted: " + str(len(w_values)) + " w values to tensor")
    # Convert the list of lists into a PyTorch tensor
    w_tensor = torch.tensor(w_values, dtype=torch.float32)
    #torch.unique(w_tensor, dim=0)
    return w_tensor



#Mesh tensor only once
def convert_mesh_tensor(input_dir):

   dirpwd = os.path.dirname(os.path.abspath(__file__))
   input_dir = os.path.join(dirpwd, input_folder)
   with open(input_dir + "/mesh.msh", 'r') as f:
      mesh = f.read().split('\n')

   num_nodes, num_elements, _ = map(int, mesh[0].split())

   # Extract node information
   nodes = mesh[1:num_nodes + 1]

   # Parse node information into a tensor
   node_values = [list(map(float, line.split()[:2])) for line in nodes]  # Only take x and y coordinates
   pos = torch.tensor(node_values, dtype=torch.float32)

   # Extract element (triangle) information
   elements = mesh[num_nodes + 1:num_nodes + 1 + num_elements]

   # Parse element information to get edge indices
   edge_index = []
   for line in elements:
       indices = list(map(int, line.split()[:3]))
       edge_index.append([indices[0] - 1, indices[1] - 1])  # Convert to 0-based index
       edge_index.append([indices[1] - 1, indices[2] - 1])
       edge_index.append([indices[2] - 1, indices[0] - 1])

   edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # Transpose to get the correct shape

   # Create the PyTorch Geometric Data object
   data = Data(pos=pos, edge_index=edge_index)

   return data



def get_log_name_config(config):
    return (
        config["NeuralNetwork"]["Architecture"]["model_type"]
        + "-r-"
        + str(config["NeuralNetwork"]["Architecture"]["radius"])
        + "-ncl-"
        + str(config["NeuralNetwork"]["Architecture"]["num_conv_layers"])
        + "-hd-"
        + str(config["NeuralNetwork"]["Architecture"]["hidden_dim"])
        + "-ne-"
        + str(config["NeuralNetwork"]["Training"]["num_epoch"])
        + "-lr-"
        + str(config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"])
        + "-bs-"
        + str(config["NeuralNetwork"]["Training"]["batch_size"])
        + "-node_ft-"
        + "".join(
            str(x)
            for x in config["NeuralNetwork"]["Variables_of_interest"][
                "input_node_features"
            ]
        )
        + "-task_weights-"
        + "".join(
            str(weigh) + "-"
            for weigh in config["NeuralNetwork"]["Architecture"]["task_weights"]
        )
    )

class data_and_model:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.iteration_number = 0

    def get_model(self):
        return self.model

    def get_data(self):
        return self.data

    def get_iteration_number(self):
        return self.iteration_number

    def set_iteration_number(self, iteration_number):
        self.iteration_number = iteration_number

class Watcher:


    def __init__(self):
        self.observer = Observer()
        dirpwd = os.path.dirname(os.path.abspath(__file__))
        input_dir = os.path.join(dirpwd, input_folder)
        self.DIRECTORY_TO_WATCH = input_dir

    def run(self, model):
        event_handler = Handler(model)
        self.observer.schedule(event_handler, self.DIRECTORY_TO_WATCH, recursive=False)
        self.observer.start()
        try:
            while True:
                t.sleep(1)
        except KeyboardInterrupt:
            self.observer.stop()
        self.observer.join()

class Handler(FileSystemEventHandler):
    def __init__(self, data_and_model):
        self.data_and_model = data_and_model
    def on_created(self, event):
        #I want to discard the event generated from the output of the neural network
        if event.is_directory or "input_iteration" not in event.src_path :
            print(f"Received created event - {event.src_path}, but won't process it")
            return None
        else:
            # Here you can add your own action to be performed on the new file
            print(f"Received created event - {event.src_path}")
            data = Data()
            data.pos = data_and_model.get_data().pos
            data.edge_index = data_and_model.get_data().edge_index
            data.edge_attr = data_and_model.get_data().edge_attr

            # Wait until the file is fully written
            self.wait_until_file_is_fully_written(event.src_path)

            # Load the file
            data.x = convert_w_to_tensor(data_and_model.get_iteration_number())

            predicted = self.data_and_model.get_model()(data.to(get_device()))
            predicted_u1 = predicted[0].flatten()
            predicted_u2 = predicted[1].flatten()
            #print on file
            dirpwd = os.path.dirname(os.path.abspath(__file__))
            input_dir = os.path.join(dirpwd, input_folder)
            np.savetxt(input_dir + "/predicted_u1_" + str(data_and_model.get_iteration_number()) + ".txt", predicted_u1.detach().numpy())
            np.savetxt(input_dir + "/predicted_u2_" + str(data_and_model.get_iteration_number()) + ".txt", predicted_u2.detach().numpy())
            data_and_model.set_iteration_number(data_and_model.get_iteration_number() + 1)

    #I need to wait until the file is completely written
    def wait_until_file_is_fully_written(self, file_path, target_row_count=4961):
        while True:
            with open(file_path, 'r') as file:
                current_row_count = sum(1 for _ in file)  # Count the number of rows
            if current_row_count >= target_row_count:
                break
            t.sleep(0.001)  # Wait a moment before checking again

    #This implementation works but sometimes python reads faster than what freefem writes
    # def wait_until_file_is_fully_written(self, file_path):
    #     last_size = -1
    #     while True:
    #         current_size = os.path.getsize(file_path)
    #         if current_size == last_size:
    #             break
    #         last_size = current_size
    #         t.sleep(0.001)

#Input folder for data from FreeFEM
input_folder  = "online_data"

if __name__ == "__main__":

    modelname = "PNA-r-5-ncl-6-hd-200-ne-15-lr-0.001-bs-3-data-tensors-node_ft-2-task_weights-1-1-"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile", help="input file", type=str, default="logs/" + modelname + "/config.json"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--adios",
        help="Adios gan_dataset",
        action="store_const",
        dest="format",
        const="adios",
    )
    group.add_argument(
        "--pickle",
        help="Pickle gan_dataset",
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


    comm.Barrier()

    timer = Timer("load_data")
    timer.start()
    model = create_model_config(
        config=config["NeuralNetwork"],
        verbosity=config["Verbosity"]["level"],
    )

    #Load the model and the data

    #Load model
    model = torch.nn.parallel.DistributedDataParallel(model)
    load_existing_model(model, modelname, path="./logs/")

    #Create data adjency matrix only
    create_graph_fromXYZ = RadiusGraph(r=5.0)
    compute_edge_lengths = Distance(norm=False, cat=True)
    dirpwd = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(dirpwd, input_folder)
    # Create the edge only once
    first_data = convert_mesh_tensor(input_dir)
    first_data = compute_edge_lengths(first_data)
    data_and_model = data_and_model(model, first_data)
    model.eval()
    print("Model loaded")
    w = Watcher()
    w.run(data_and_model)







