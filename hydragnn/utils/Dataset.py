import os
import time

import numpy as np
from mpi4py import MPI

from hydragnn.utils.abstractbasedataset import AbstractBaseDataset
from hydragnn.utils.parallel_read import parallel_processing, split


class Dataset(AbstractBaseDataset):

    def __init__(self):
        #time.sleep(2000)
        comm = MPI.COMM_WORLD
        size = comm.Get_size()  # new: gives number of ranks in comm
        rank = comm.Get_rank()
        values =[]
        # Read all the saved tensors
        self.tensordictionary = []
        self.graph_list = []  # Contains all the graphs created
        # Get the length of the result folder
        #I want only one process to get the data

        ################################################################################
        # Invece che tutto sto casino potevo contare quanti elementi ho e fare un array
        # su quel numero tanto son sequenziali
        ################################################################################
        if(rank == 0):

            input_path = "/Users/lorenzonebolosi/Desktop/HydraGNN/hydragnn/utils/tensors"
            values = os.listdir(input_path)
            #This is necessary since scatterv doenst work with strings,
            #So i trunkate before and after and save the ints
            new_list = [str(i).removeprefix('random_v_') for i in values]
            new_list = [str(i).removesuffix('.pt') for i in new_list]
            values = np.array(new_list, dtype=np.dtype('int'))
            values_length = len(values)
        else:
            values = None
            values_length = None

        values_length = comm.bcast(values_length, root=0)

        if rank == 0:
            divided_data = np.empty(int((len(values)/size) + (len(values)%size)), dtype=np.dtype('int'))
        else:
            # I have no idea why but I need to put the +1 and store extra values otherwise it will give a Message
            # Truncated error
            print("Received a values_length of: "+str(values_length))
            divided_data = np.empty(int((values_length / size)) +1, dtype=np.dtype('int'))


        comm.Scatterv(values, divided_data, root=0)

        # I want to wait that all the process have the data and call them in parallel
        MPI.COMM_WORLD.Barrier()
        # I first need to eliminate random values that are there only because of memory space
        divided_data[divided_data > values_length] = -1
        # I don't know why it assigns two 0 files so I need to discard them
        if rank != 0:
            divided_data[divided_data == 0] = -1
        # Now I need to re-add the random_v_ and .pt to the numbers

        values = np.array(divided_data, dtype=np.dtype('str'))
        new_list = [ str(i) + '.pt' for i in values]
        new_list = ['random_v_' + str(i) for i in new_list]

        print("Process "+str(rank)+" has data: "+str(new_list))

        parallel_graphs = parallel_processing(new_list)
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
