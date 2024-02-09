import torch
from mpi4py import MPI
import numpy as np
from torch_geometric.transforms import RadiusGraph, Distance
from hydragnn.utils.data_object import Data_object


#Used for splitting the inputs between all the processes
def split( a, n):
    k, m = divmod(len(a), n)
    final_list = (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
    return list(final_list)

def parallel_processing(data):

    tensordictionary = []
    graph_list = []
    create_graph_fromXYZ = RadiusGraph(r=5.0)
    compute_edge_lengths = Distance(norm=False, cat=True)

    #I need to discard all the excess values that have been transformed in -1
    for value in data:
        tensore = torch.load('/Users/lorenzonebolosi/Desktop/HydraGNN/hydragnn/utils/tensors/' + value)
        tensordictionary.append(tensore)

    #Create the edge only once
    first_data = Data_object(tensordictionary[0][0])
    first_data = create_graph_fromXYZ(first_data)
    first_data = compute_edge_lengths(first_data)
    for tensorList in tensordictionary:
        for tensor in tensorList:
            #print(str(type(tensor)))
            # I need to pass a data that has the discussed structure
            data = Data_object(tensor)
            #data.pos = tensor[:, :2]
            data.set_graphs(first_data.edge_index, first_data.edge_attr)
            graph_list.append(data)
    return graph_list
