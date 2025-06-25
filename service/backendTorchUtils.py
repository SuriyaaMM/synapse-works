from config import logging

import torch
from torch import nn
from torch.utils import data as tdu
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from backendTorchDataset import TorchCustomCSVDataset

def torch_layer_name_map(layer_name: str, debug: bool = True) -> type[nn.Module]:
    R"""maps layer_name to respective torch.nn.Module

        NOTE: converts the layer_name to lowercase string

    Args:
        layer_name:str, layer name like linear, dropout, conv2d
    """
    # map for iterating
    layer_name_map = {
        "linear" : nn.Linear,
        "conv2d": nn.Conv2d,
        "conv1d": nn.Conv1d,
        "maxpool2d": nn.MaxPool2d,
        "maxpool1d": nn.MaxPool1d,
        "avgpool2d": nn.AvgPool2d,
        "avgpool1d": nn.AvgPool1d,
        "batchnorm2d": nn.BatchNorm2d,
        "batchnorm1d": nn.BatchNorm1d,
        "flatten" : nn.Flatten,
        "dropout" : nn.Dropout,
        "elu" : nn.ELU,
        "relu" : nn.ReLU,
        "leakyrelu" : nn.LeakyReLU,
        "sigmoid": nn.Sigmoid,
        "logsigmoid": nn.LogSigmoid,
        "tanh" : nn.Tanh
    }
    # convert to lowercase
    layer_name = layer_name.lower()
     # check for key error
    if(debug):
        if layer_name.lower() not in layer_name_map.keys():
            logging.error(f"PARAM(layer_name)({layer_name}) is not a torch.nn.Module")
            raise KeyError(f"PARAM(layer_name)({layer_name}) is not a torch.nn.Module")
    # return layer type
    return layer_name_map[layer_name]


def torch_dataset_name_map(dataset_name: str, debug: bool = True) -> type[tdu.Dataset]:
    R"""maps dataset_name to valid torch inbuilt datasets

    Args:
        dataset_name:str, valid torch built-in dataset name
    """
    # map for iterating
    dataset_map = {
        "mnist" : MNIST,
        "cifar10" : CIFAR10,
        "cifar100" : CIFAR100,
        "custom_csv": TorchCustomCSVDataset
    }
    # convert to lowercase
    dataset_name = dataset_name.lower()
    if(debug):
        if dataset_name.lower() not in dataset_map.keys():
            logging.error(f"PARAM(dataset_name)({dataset_name}) is not a torch built-in dataset")
            raise KeyError(f"PARAM(dataset_name)({dataset_name}) is not a torch built-in dataset")
    # return dataset
    return dataset_map[dataset_name]

def torch_optimizer_name_map(optimizer_name: str, debug: bool = True) -> type[torch.optim.Optimizer]:
    R"""maps optimizer_name to valid torch.optim optimizers

    Args:
        optimizer_name:str, valid torch.optim optimizer
    """
    # map for iterating
    optimizer_map = {
        "adadelta" : torch.optim.Adadelta,
        "adafactor" : torch.optim.Adafactor,
        "adam" : torch.optim.Adam,
        "adamw" : torch.optim.AdamW,
        "sparseadam" : torch.optim.SparseAdam,
        "adamax" : torch.optim.Adamax,
        "asgd" : torch.optim.ASGD,
        "lbfgs" : torch.optim.LBFGS,
        "radam" : torch.optim.RAdam,
        "rmsprop" : torch.optim.RMSprop,
        "rprop" : torch.optim.Rprop,
        "sgd" : torch.optim.SGD
    }
    # convert to lowercase
    optimizer_name = optimizer_name.lower()
    if(debug):
        if optimizer_name.lower() not in optimizer_map.keys():
            logging.error(f"PARAM(optimizer_name)({optimizer_name}) is not a torch optimizer")
            raise KeyError(f"PARAM(optimizer_name)({optimizer_name}) is not a torch optimizer")
    # return optimizer
    return optimizer_map[optimizer_name]

def torch_loss_function_name_map(loss_function_name: str, debug: bool = True) -> type[nn.Module]:
    R"""maps loss_function_name to valid torch loss functions

    Args:
        loss_function_name:str, valid torch loss function
    """
    # map for iterating
    loss_function_map = {
        "ce" : nn.CrossEntropyLoss,
        "bce" : nn.BCEWithLogitsLoss
    }
    # convert to lowercase
    loss_function_name = loss_function_name.lower()
    if(debug):
        if loss_function_name.lower() not in loss_function_map.keys():
            logging.error(f"PARAM(loss_function_name)({loss_function_name}) is not a torch loss function")
            raise KeyError(f"PARAM(loss_function_name)({loss_function_name}) is not a torch loss function")
    # return loss function
    return loss_function_map[loss_function_name]