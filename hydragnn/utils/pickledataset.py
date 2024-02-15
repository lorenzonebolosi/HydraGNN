import os
import pickle

import torch
from mpi4py import MPI

from .print_utils import print_distributed, log, iterate_tqdm

from hydragnn.utils.abstractbasedataset import AbstractBaseDataset
from hydragnn.preprocess import update_predicted_values, update_atom_features

import hydragnn.utils.tracer as tr


class SimplePickleDataset(AbstractBaseDataset):
    """Simple Pickle Dataset"""

    def __init__(self, basedir, label, subset=None, preload=False, var_config=None):
        """
        Parameters
        ----------
        basedir: basedir
        label: label
        subset: a list of index to subset
        """
        super().__init__()

        self.basedir = basedir
        self.label = label
        self.subset = subset
        self.preload = preload
        self.var_config = var_config

        if self.var_config is not None:
            self.input_node_features = self.var_config["input_node_features"]
            self.variables_type = self.var_config["type"]
            self.output_index = self.var_config["output_index"]
            self.graph_feature_dim = self.var_config["graph_feature_dims"]
            self.node_feature_dim = self.var_config["node_feature_dims"]

        fname = os.path.join(basedir, "%s-meta.pkl" % label)
        with open(fname, "rb") as f:
            self.minmax_node_feature = pickle.load(f)
            self.minmax_graph_feature = pickle.load(f)
            self.ntotal = pickle.load(f)
            self.use_subdir = pickle.load(f)
            self.nmax_persubdir = pickle.load(f)
            self.attrs = pickle.load(f)
        log("Pickle files:", self.label, self.ntotal)
        if self.attrs is None:
            self.attrs = dict()
        for k in self.attrs:
            setattr(self, k, self.attrs[k])

        if self.subset is None:
            self.subset = list(range(self.ntotal))

        if self.preload:
            for i in range(self.ntotal):
                data = self.read(i)
                self.update_data_object(data)
                self.dataset.append(data)

    def len(self):
        return len(self.subset)

    @tr.profile("get")
    def get(self, i):
        k = self.subset[i]
        if self.preload:
            return self.dataset[k]
        else:
            return self.read(k)

    def setsubset(self, subset):
        self.subset = subset

    def read(self, k):
        """
        Read from disk
        """
        fname = "%s-%d.pkl" % (self.label, k)
        dirfname = os.path.join(self.basedir, fname)
        if self.use_subdir:
            subdir = str(k // self.nmax_persubdir)
            dirfname = os.path.join(self.basedir, subdir, fname)
        with open(dirfname, "rb") as f:
            data_object = pickle.load(f)
            self.update_data_object(data_object)
        return data_object

    def setsubset(self, subset):
        self.subset = subset

    def update_data_object(self, data_object):
        if self.var_config is not None:
            update_predicted_values(
                self.variables_type,
                self.output_index,
                self.graph_feature_dim,
                self.node_feature_dim,
                data_object,
            )
            update_atom_features(self.input_node_features, data_object)


