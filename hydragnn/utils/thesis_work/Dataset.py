import os
import pickle

from mpi4py import MPI

import hydragnn
from hydragnn.preprocess.utils import gather_deg
from hydragnn.utils import iterate_tqdm
from hydragnn.utils.distributed import nsplit

from hydragnn.utils.abstractbasedataset import AbstractBaseDataset
from hydragnn.utils.thesis_work.parallel_read import parallel_processing


class GraphDataset(AbstractBaseDataset):

    def __init__(self, results_path):
        # Always initialize for multi-rank training.
        size, rank = hydragnn.utils.setup_ddp()

        # Read all the saved tensors
        self.tensordictionary = []
        self.parallel_graphs = []
        self.graph_list = []  # Contains all the graphs created
        # Get the length of the result folder
        input_path = results_path#"/Users/lorenzonebolosi/Desktop/Tesi/FreeFem_code/results/"
        values = os.listdir(input_path)
        values.remove('.DS_Store')
        values.remove('mesh.msh')
        local_values = list(nsplit(values, size))[rank]
        print("Process "+str(rank)+" has data: "+str(len(local_values)))
        #I receive different folders, each representing a different run of the FreeFem code
        for local_value in local_values:
            self.parallel_graphs.append(parallel_processing(local_value))
        # Each file has a tensor for every iteration. So each file represent a complete run of the FreeFem code.
        MPI.COMM_WORLD.Barrier()

        print("Process "+str(rank)+" has produced: "+str(len(self.parallel_graphs))+" graphs")
        #deg = gather_deg(parallel_graphs)

        ## pickle
        basedir = os.path.join(os.path.dirname(__file__), "dataset", "pickle")
        attrs = dict()
        attrs["pna_deg"] = 10 ##CHIEDERE CHE PARAMETRO E'
        SimplePickleWriter(
            self.parallel_graphs,
            basedir,
            "trainset",
            use_subdir=False,
            attrs=attrs,
        )
        #gathered_data = comm.gather(sendbuf, root=0)
        # if rank == 0:
        #     print("Gathered array of: " +str(len(gathered_data)))
        #
        #
        # if(rank == 0):
        #     # I now have a list of lists that i need to flatten
        #     flattened_list = [item for sublist in gathered_data for item in sublist]
        #
        #     self.graph_list = flattened_list
        #     print("Converted input tensors into " + str(len(self.graph_list)) + " graphs")
        #

        MPI.COMM_WORLD.Barrier()


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

class SimplePickleWriter:
    """SimplePickleWriter class to write Torch Geometric graph data"""

    def __init__(
        self,
        dataset,
        basedir,
        label="total",
        minmax_node_feature=None,
        minmax_graph_feature=None,
        use_subdir=False,
        nmax_persubdir=10_000,
        comm=MPI.COMM_WORLD,
        attrs=dict(),
    ):
        """
        Parameters
        ----------
        dataset: locally owned dataset (should be iterable)
        basedir: basedir
        label: label
        nmax: nmax in case of subdir
        minmax_node_feature: minmax_node_feature
        minmax_graph_feature: minmax_graph_feature
        comm: MPI communicator
        """

        self.dataset = dataset
        if not isinstance(dataset, list):
            raise Exception("Unsuppored data type yet.")

        self.basedir = basedir
        self.label = label
        self.use_subdir = use_subdir
        self.nmax_persubdir = nmax_persubdir
        self.comm = comm
        self.rank = comm.Get_rank()

        self.minmax_node_feature = minmax_node_feature
        self.minmax_graph_feature = minmax_graph_feature

        ns = self.comm.allgather(len(self.dataset))
        noffset = sum(ns[: self.rank])
        ntotal = sum(ns)

        if self.rank == 0:
            if not os.path.exists(basedir):
                os.makedirs(basedir)
            fname = os.path.join(basedir, "%s-meta.pkl" % (label))
            with open(fname, "wb") as f:
                pickle.dump(self.minmax_node_feature, f)
                pickle.dump(self.minmax_graph_feature, f)
                pickle.dump(ntotal, f)
                pickle.dump(use_subdir, f)
                pickle.dump(nmax_persubdir, f)
                pickle.dump(attrs, f)
        comm.Barrier()

        if use_subdir:
            ## Create subdirs first
            subdirs = set()
            for i in range(len(self.dataset)):
                subdirs.add(str((noffset + i) // nmax_persubdir))
            for k in subdirs:
                subdir = os.path.join(basedir, k)
                os.makedirs(subdir, exist_ok=True)

        for i, data in iterate_tqdm(
            enumerate(self.dataset),
            2,
            total=len(self.dataset),
            desc="Pickle write %s" % self.label,
        ):
            fname = "%s-%d.pkl" % (label, noffset + i)
            dirfname = os.path.join(basedir, fname)
            if use_subdir:
                subdir = str((noffset + i) // nmax_persubdir)
                dirfname = os.path.join(basedir, subdir, fname)
            with open(dirfname, "wb") as f:
                pickle.dump(data, f)
