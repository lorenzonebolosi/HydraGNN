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
        input_path = "hydragnn/utils/thesis_work/tensors"
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

        deg = gather_deg(parallel_graphs)
        ## pickle
        basedir = os.path.join(os.path.dirname(__file__), "dataset", "pickle")
        attrs = dict()
        attrs["pna_deg"] = deg
        SimplePickleWriter(
            parallel_graphs,
            basedir,
            "trainset",
            use_subdir=False,
            attrs=attrs,
        )


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
