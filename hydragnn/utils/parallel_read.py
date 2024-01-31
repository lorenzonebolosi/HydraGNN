import torch
from mpi4py import MPI
import numpy as np
from torch_geometric.transforms import RadiusGraph, Distance
from hydragnn.utils.data_object import Data_object


def parallel_processing(data):
    comm = MPI.COMM_WORLD
    size = comm.Get_size() # new: gives number of ranks in comm
    rank = comm.Get_rank()
    numDataPerRank = int(len(data) / size) #Each process gets a value
    tensordictionary = []
    graph_list = []
    recvbuf = np.empty(numDataPerRank, dtype='s')  # allocate space for recvbuf
    comm.Scatter(data, recvbuf, root=0)
    create_graph_fromXYZ = RadiusGraph(r=5.0)
    compute_edge_lengths = Distance(norm=False, cat=True)
    print('Rank: ', rank, ', recvbuf received: ', recvbuf)
    for value in recvbuf:
        tensore = torch.load('tensors/' + value)
        tensordictionary.append(tensore)

    #Create the edge only once
    first_data = Data_object(tensordictionary[0][0])
    first_data = create_graph_fromXYZ(first_data)
    first_data = compute_edge_lengths(first_data)
    for tensorList in tensordictionary:
        for tensor in tensorList:
            print(str(type(tensor)))
            # I need to pass a data that has the discussed structure
            data = Data_object(tensor)
            #data.pos = tensor[:, :2]
            data.set_graphs(first_data.edge_index, first_data.edge_attr)
            graph_list.append(data)

    print('Rank: ', rank, ', sendbuf: ', graph_list)

    recvbuf = None
    if rank == 0:
        recvbuf = np.empty(numDataPerRank*size, dtype='d')

    comm.Gather(graph_list, recvbuf, root=0)

    if rank == 0:
        print('Rank: ',rank, ', recvbuf received: ',recvbuf)
        return recvbuf

