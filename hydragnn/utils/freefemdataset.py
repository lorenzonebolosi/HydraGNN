import os

import torch
from mpi4py import MPI

from hydragnn.utils.distributed import nsplit

from hydragnn.utils.abstractbasedataset import AbstractBaseDataset
from hydragnn.utils.parallel_read import parallel_processing


class GraphDataset(AbstractBaseDataset):

    def __init__(self, results_path):
        super().__init__()
        self.world_size = torch.distributed.get_world_size()
        self.rank = torch.distributed.get_rank()
        # Always initialize for multi-rank training.
        # Read all the saved tensors
        self.tensordictionary = []
        self.graph_list = []  # Contains all the graphs created
        # Get the length of the result folder
        input_path = results_path#"/Users/lorenzonebolosi/Desktop/Tesi/FreeFem_code/results/"
        values = os.listdir(input_path)
        values.remove('.DS_Store')
        values.remove('mesh.msh')
        local_values = list(nsplit(values, self.world_size))[self.rank]
        print("Process "+str(self.rank)+" has data: "+str(len(local_values)))
        #I receive different folders, each representing a different run of the FreeFem code
        for local_value in local_values:
            self.dataset.extend(parallel_processing(local_value))
        # Each file has a tensor for every iteration. So each file represent a complete run of the FreeFem code.
        MPI.COMM_WORLD.Barrier()

        print("Process "+str(self.rank)+" has produced: "+str(len(self.dataset))+" graphs")

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset[idx]