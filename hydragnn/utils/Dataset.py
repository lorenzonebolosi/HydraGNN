import os
import time

import numpy as np
from mpi4py import MPI

import hydragnn
from hydragnn.utils.distributed import nsplit

from hydragnn.utils.abstractbasedataset import AbstractBaseDataset
from hydragnn.utils.parallel_read import parallel_processing, split


class Dataset(AbstractBaseDataset):

    def __init__(self):
        #time.sleep(2000)
        # Always initialize for multi-rank training.

        comm = MPI.COMM_WORLD
        size, rank = hydragnn.utils.setup_ddp()

        values =[]
        # Read all the saved tensors
        self.tensordictionary = []
        self.graph_list = []  # Contains all the graphs created
        # Get the length of the result folder
        input_path = "/Users/lorenzonebolosi/Desktop/HydraGNN/hydragnn/utils/tensors"
        values = os.listdir(input_path)
        local_values = list(nsplit(values, size))[rank]
        print("Process "+str(rank)+" has data: "+str(len(local_values)))
        parallel_graphs = parallel_processing(local_values)
        # Each file has a tensor for every iteration. So each file represent a complete run of the FreeFem code.
        MPI.COMM_WORLD.Barrier()

        sendbuf = parallel_graphs
        print("Process "+str(rank)+" has produced: "+str(len(sendbuf))+" graphs")
        # Collect local array sizes using the high-level mpi4py gather
        # sendcounts = np.array(comm.gather(len(sendbuf), root = 0))
        # if rank == 0:
        #     print("sendcounts: {}, total: {}".format(sendcounts, sum(sendcounts)))
        #     recvbuf = [None] * (sum(sendcounts)) #Create an empylist of sendcounts elements
        # else:
        #     recvbuf = None

        gathered_data = comm.gather(sendbuf, root = 0)
        MPI.COMM_WORLD.Barrier()

        #comm.Gatherv(sendbuf=sendbuf, recvbuf=(recvbuf, sendcounts), root=0)

        if rank == 0:
            print("Gathered array of: " +str(len(gathered_data)))

        #comm.Gatherv(parallel_graphs, root=0)

        if(rank == 0):
            # I now have a list of lists that i need to flatten
            flattened_list = [item for sublist in gathered_data for item in sublist]

            self.graph_list = flattened_list
            print("Converted input tensors into " + str(len(self.graph_list)) + " graphs")



    def get(self, idx):
        """
        Return a dataset at idx
        """
        return self.graph_list[idx]
        pass

    def len(self):
        """
        Total number of dataset.
        If data is distributed, it should be the global total size.
        """
        return len(self.graph_list)
        pass
