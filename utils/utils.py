import os
from random import shuffle
from tqdm import tqdm

import torch
from torch import nn
from torch_geometric.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_utils.serialized_dataset_loader import (
    SerializedDataLoader,
)
from data_utils.raw_dataset_loader import RawDataLoader
from data_utils.dataset_descriptors import (
    AtomFeatures,
    StructureFeatures,
    Dataset,
)
from utils.models_setup import generate_model
from utils.visualizer import Visualizer


def train_validate_test_normal(
    model,
    optimizer,
    train_loader,
    val_loader,
    test_loader,
    writer,
    scheduler,
    config,
    model_with_config_name,
):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        """
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        """
    model.to(device)
    num_epoch = config["num_epoch"]
    for epoch in range(0, num_epoch):
        train_mae = train(train_loader, model, optimizer, config["output_dim"])
        val_mae = validate(val_loader, model, config["output_dim"])
        test_rmse = test(test_loader, model, config["output_dim"])
        scheduler.step(val_mae)
        writer.add_scalar("train error", train_mae, epoch)
        writer.add_scalar("validate error", val_mae, epoch)
        writer.add_scalar("test error", test_rmse[0], epoch)

        print(
            f"Epoch: {epoch:02d}, Train MAE: {train_mae:.8f}, Val MAE: {val_mae:.8f}, "
            f"Test RMSE: {test_rmse[0]:.8f}"
        )
    # At the end of training phase, do the one test run for visualizer to get latest predictions
    visualizer = Visualizer(model_with_config_name)
    test_rmse, true_values, predicted_values = test(
        test_loader, model, config["output_dim"]
    )
    visualizer.add_test_values(
        true_values=true_values, predicted_values=predicted_values
    )
    visualizer.create_scatter_plot()


def train(loader, model, opt, output_dim):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    total_error = 0
    model.train()
    for data in tqdm(loader):
        data = data.to(device)
        opt.zero_grad()
        pred = model(data)
        loss = model.loss_rmse(pred, data.y)
        loss.backward()
        total_error += loss.item() * data.num_graphs
        opt.step()
    return total_error / len(loader.dataset)


@torch.no_grad()
def validate(loader, model, output_dim):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    total_error = 0
    model.eval()
    for data in tqdm(loader):
        data = data.to(device)
        pred = model(data)
        error = model.loss_rmse(pred, data.y)
        total_error += error.item() * data.num_graphs

    return total_error / len(loader.dataset)


@torch.no_grad()
def test(loader, model, output_dim):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    total_error = 0
    model.eval()
    true_values = []
    predicted_values = []
    for data in tqdm(loader):
        data = data.to(device)
        pred = model(data)
        true_values.extend(data.y.tolist())
        predicted_values.extend(pred.tolist())
        error = model.loss_rmse(pred, data.y)
        total_error += error.item() * data.num_graphs

    return total_error / len(loader.dataset), true_values, predicted_values


def dataset_loading_and_splitting(
    config: {},
    chosen_dataset_option: Dataset,
):

    if chosen_dataset_option == Dataset.CuAu:
        dataset_CuAu = load_data(Dataset.CuAu.value, config)
        return split_dataset(
            dataset=dataset_CuAu,
            batch_size=config["batch_size"],
            perc_train=config["perc_train"],
        )
    elif chosen_dataset_option == Dataset.FePt:
        dataset_FePt = load_data(Dataset.FePt.value, config)
        return split_dataset(
            dataset=dataset_FePt,
            batch_size=config["batch_size"],
            perc_train=config["perc_train"],
        )
    elif chosen_dataset_option == Dataset.FeSi:
        dataset_FeSi = load_data(Dataset.FeSi.value, config)
        return split_dataset(
            dataset=dataset_FeSi,
            batch_size=config["batch_size"],
            perc_train=config["perc_train"],
        )
    else:
        dataset_CuAu = load_data(Dataset.CuAu.value, config)
        dataset_FePt = load_data(Dataset.FePt.value, config)
        dataset_FeSi = load_data(Dataset.FeSi.value, config)
        if chosen_dataset_option == Dataset.CuAu_FePt_SHUFFLE:
            dataset_CuAu.extend(dataset_FePt)
            dataset_combined = dataset_CuAu
            shuffle(dataset_combined)
            return split_dataset(
                dataset=dataset_combined,
                batch_size=config["batch_size"],
                perc_train=config["perc_train"],
            )
        elif chosen_dataset_option == Dataset.CuAu_TRAIN_FePt_TEST:

            return combine_and_split_datasets(
                dataset1=dataset_CuAu,
                dataset2=dataset_FePt,
                batch_size=config["batch_size"],
                perc_train=config["perc_train"],
            )
        elif chosen_dataset_option == Dataset.FePt_TRAIN_CuAu_TEST:
            return combine_and_split_datasets(
                dataset1=dataset_FePt,
                dataset2=dataset_CuAu,
                batch_size=config["batch_size"],
                perc_train=config["perc_train"],
            )
        elif chosen_dataset_option == Dataset.FePt_FeSi_SHUFFLE:
            dataset_FePt.extend(dataset_FeSi)
            dataset_combined = dataset_FePt
            shuffle(dataset_combined)
            return split_dataset(
                dataset=dataset_combined,
                batch_size=config["batch_size"],
                perc_train=config["perc_train"],
            )


def split_dataset(dataset: [], batch_size: int, perc_train: float):
    perc_val = (1 - perc_train) / 2
    data_size = len(dataset)
    train_loader = DataLoader(
        dataset[: int(data_size * perc_train)], batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        dataset[int(data_size * perc_train) : int(data_size * (perc_train + perc_val))],
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        dataset[int(data_size * (perc_train + perc_val)) :],
        batch_size=batch_size,
        shuffle=True,
    )

    return train_loader, val_loader, test_loader


def combine_and_split_datasets(
    dataset1: [], dataset2: [], batch_size: int, perc_train: float
):
    data_size = len(dataset1)
    train_loader = DataLoader(
        dataset1[: int(data_size * perc_train)], batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        dataset1[int(data_size * perc_train) :],
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(dataset2, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


def load_data(dataset_option, config):
    transform_raw_data_to_serialized()
    files_dir = (
        f"{os.environ['SERIALIZED_DATA_PATH']}/serialized_dataset/{dataset_option}.pkl"
    )

    # loading serialized data and recalculating neighbourhoods depending on the radius and max num of neighbours
    loader = SerializedDataLoader()
    dataset = loader.load_serialized_data(
        dataset_path=files_dir,
        config=config,
    )

    return dataset


def transform_raw_data_to_serialized():
    # Loading raw data if necessary
    raw_datasets = ["CuAu_32atoms", "FePt_32atoms", "FeSi_1024atoms"]
    if len(
        os.listdir(os.environ["SERIALIZED_DATA_PATH"] + "/serialized_dataset")
    ) < len(raw_datasets):
        for raw_dataset in raw_datasets:
            files_dir = (
                os.environ["SERIALIZED_DATA_PATH"]
                + "/dataset/"
                + raw_dataset
                + "/output_files/"
            )
            loader = RawDataLoader()
            loader.load_raw_data(dataset_path=files_dir)