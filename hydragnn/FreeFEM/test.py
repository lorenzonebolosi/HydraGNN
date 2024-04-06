import os, json
import logging
import sys

from mpi4py import MPI
import argparse

import torch

import hydragnn
#sys.path.append(os.path.dirname(os.path.abspath(__file__))+'thesis_work')

from hydragnn.utils.freefemdataset import GraphDataset
from hydragnn.utils.time_utils import Timer
from hydragnn.utils.config_utils import get_log_name_config
from hydragnn.utils.model import print_model
from hydragnn.utils.pickledataset import SimplePickleWriter, SimplePickleDataset
from hydragnn.preprocess.load_data import split_dataset
from hydragnn.preprocess.utils import gather_deg
import hydragnn.utils.tracer as tr


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))

if __name__ == "__main__":
    #exec(os.path.dirname(os.path.abspath(__file__))+ "/freefemdataset.py")
    # Get the current working directory
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    print("Current working directory:", ROOT_DIR)
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, help="batch_size", default=None)
    parser.add_argument(
        "--loadexistingsplit",
        action="store_true",
        help="loading from existing pickle/adios files with train/test/validate splits",
    )
    parser.add_argument(
        "--preonly",
        action="store_true",
        help="preprocess only. Adios or pickle saving and no train",
    )
    parser.add_argument("--inputfile", help="input file", type=str, default="custom.json")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--pickle",
        help="Pickle dataset",
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
    ## Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%%(levelname)s (rank %d): %%(message)s" % (rank),
        datefmt="%H:%M:%S",
    )

    datasetname = config["Dataset"]["name"]
    config["Dataset"]["name"] = "%s_%d" % (datasetname, rank)
    modelname = "FreeFEM"
    node_feature_names = ["ux", "uy", "w"]
    graph_feature_names = []
    graph_feature_dims = []
    node_feature_dims = [1, 1, 1]
    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(dirpwd, "../../hydragnn/utils/thesis_work/dataset")
    # Configurable run choices (JSON file that accompanies this example script).
    with open(input_filename, "r") as f:
        config = json.load(f)
    verbosity = config["Verbosity"]["level"]
    var_config = config["NeuralNetwork"]["Variables_of_interest"]
    var_config["graph_feature_names"] = graph_feature_names
    var_config["graph_feature_dims"] = graph_feature_dims
    var_config["node_feature_names"] = node_feature_names
    var_config["node_feature_dims"] = node_feature_dims
    if args.batch_size is not None:
        config["NeuralNetwork"]["Training"]["batch_size"] = args.batch_size

    if not args.loadexistingsplit:
        total = GraphDataset(
            os.path.dirname(os.path.abspath(__file__))+"/freeFEM_results/")  # dirpwd + "/dataset/VASP_calculations/binaries", config, dist=True)
        print(len(total))
        trainset, valset, testset = split_dataset(
            dataset=total,
            perc_train=config["NeuralNetwork"]["Training"]["perc_train"],
            stratify_splitting=False,
        )
        print(len(total), len(trainset), len(valset), len(testset))
        deg = gather_deg(trainset)
        config["pna_deg"] = deg
        setnames = ["trainset", "valset", "testset"]
        ## pickle
        if args.format == "pickle":
            basedir = os.path.join(
                os.path.dirname(__file__), "../../hydragnn/utils/thesis_work/dataset", "%s.pickle" % modelname
            )
            attrs = dict()
            attrs["pna_deg"] = deg
            SimplePickleWriter(
                trainset,
                basedir,
                "trainset",
                # minmax_node_feature=total.minmax_node_feature,
                # minmax_graph_feature=total.minmax_graph_feature,
                use_subdir=True,
                attrs=attrs,
            )
            SimplePickleWriter(
                valset,
                basedir,
                "valset",
                # minmax_node_feature=total.minmax_node_feature,
                # minmax_graph_feature=total.minmax_graph_feature,
                use_subdir=True,
            )
            SimplePickleWriter(
                testset,
                basedir,
                "testset",
                # minmax_node_feature=total.minmax_node_feature,
                # minmax_graph_feature=total.minmax_graph_feature,
                use_subdir=True,
            )
        sys.exit(0)

    tr.initialize()
    tr.disable()
    timer = Timer("load_data")
    timer.start()

    timer = Timer("load_data")
    timer.start()

    if args.format == "pickle":
        info("Pickle load")
        basedir = os.path.join(os.getcwd()+ "/dataset", "%s.pickle" % modelname)
        trainset = SimplePickleDataset(basedir=basedir, label="trainset", var_config=var_config)
        valset = SimplePickleDataset(basedir=basedir, label="valset", var_config=var_config)
        testset = SimplePickleDataset(basedir=basedir, label="testset", var_config=var_config)
        # minmax_node_feature = trainset.minmax_node_feature
        # minmax_graph_feature = trainset.minmax_graph_feature
        pna_deg = trainset.pna_deg
    else:
        raise ValueError("Unknown data format: %d" % args.format)
    ## Set minmax
    config["NeuralNetwork"]["Variables_of_interest"][
        "minmax_node_feature"
    ] = trainset.minmax_node_feature
    config["NeuralNetwork"]["Variables_of_interest"][
        "minmax_graph_feature"
    ] = trainset.minmax_graph_feature

    info(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
    )

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )
    timer.stop()

    config = hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)
    config["NeuralNetwork"]["Variables_of_interest"].pop("minmax_node_feature", None)
    config["NeuralNetwork"]["Variables_of_interest"].pop("minmax_graph_feature", None)

    verbosity = config["Verbosity"]["level"]
    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )
    if rank == 0:
        print_model(model)
    comm.Barrier()

    model = hydragnn.utils.get_distributed_model(model, verbosity)

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    log_name = get_log_name_config(config)
    writer = hydragnn.utils.get_summary_writer(log_name)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    hydragnn.utils.save_config(config, log_name)

    hydragnn.train.train_validate_test(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        writer,
        scheduler,
        config["NeuralNetwork"],
        log_name,
        verbosity,
        create_plots=True,
    )

    hydragnn.utils.save_model(model, optimizer, log_name)
    hydragnn.utils.print_timers(verbosity)

    sys.exit(0)
