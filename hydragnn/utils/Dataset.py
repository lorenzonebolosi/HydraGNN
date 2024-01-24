import os
from hydragnn.utils.abstractbasedataset import AbstractBaseDataset
from hydragnn.utils.parallel_read import parallel_processing


class Dataset(AbstractBaseDataset):

    def __init__(self):
        # Read all the saved tensors
        self.tensordictionary = []
        self.graph_list = []  # Contains all the graphs created
        # Get the length of the result folder
        input_path = "tensors/"
        values = os.listdir(input_path)
        values.remove('.DS_Store')
        self.graph_list = parallel_processing(values)
        # Each file has a tensor for every iteration. So each file represent a complete run of the FreeFem code.

        #print("Size of a single tensor is: " + str(tensordictionary[0][0].size()))
        #print("Number of files converted: " + str(len(tensordictionary)))

        print("Converted " + str(len(self.tensordictionary)) + " input tensors into " + str(len(self.graph_list)) + " graphs")

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
