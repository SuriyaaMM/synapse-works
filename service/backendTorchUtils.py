from config import logging

import torch
from torch import nn
from torch.utils import data as tdu
from torchvision.datasets import MNIST, CIFAR10, CIFAR100

def torch_layer_name_map(layer_name: str, debug: bool = True) -> type[nn.Module]:
    R"""maps layer_name to respective torch.nn.Module

        NOTE: converts the layer_name to lowercase string

    Args:
        layer_name:str, layer name like linear, dropout, conv2d
    """
    # map for iterating
    layerNameMap = {
        "linear" : nn.Linear,
        "dropout" : nn.Dropout
    }
    # convert to lowercase
    layer_name = layer_name.lower()
     # check for key error
    if(debug):
        if layer_name.lower() not in layerNameMap.keys():
            logging.error(f"PARAM(layer_name)({layer_name}) is not a torch.nn.Module")
            raise KeyError(f"PARAM(layer_name)({layer_name}) is not a torch.nn.Module")
    # return layer type
    return layerNameMap[layer_name]


def torch_dataset_name_map(dataset_name: str, debug: bool = True) -> type[tdu.Dataset]:
    R"""maps dataset_name to valid torch inbuilt datasets

    Args:
        dataset_name:str, valid torch built-in dataset name
    """
    # map for iterating
    datasetMap = {
        "mnist" : MNIST,
        "cifar10" : CIFAR10,
        "cifar100" : CIFAR100
    }
    # convert to lowercase
    dataset_name = dataset_name.lower()
    if(debug):
        if dataset_name.lower() not in datasetMap.keys():
            logging.error(f"PARAM(dataset_name)({dataset_name}) is not a torch built-in dataset")
            raise KeyError(f"PARAM(dataset_name)({dataset_name}) is not a torch built-in dataset")
    # return dataset
    return datasetMap[dataset_name]

def torch_optimizer_name_map(optimizer_name: str, debug: bool = True) -> type[torch.optim.Optimizer]:
    R"""maps optimizer_name to valid torch.optim optimizers

    Args:
        optimizer_name:str, valid torch.optim optimizer
    """
    # map for iterating
    optimizerMap = {
        "adam" : torch.optim.Adam,
        "adamw" : torch.optim.AdamW,
        "sgd" : torch.optim.SGD
    }
    # convert to lowercase
    optimizer_name = optimizer_name.lower()
    if(debug):
        if optimizer_name.lower() not in optimizerMap.keys():
            logging.error(f"PARAM(optimizer_name)({optimizer_name}) is not a torch optimizer")
            raise KeyError(f"PARAM(optimizer_name)({optimizer_name}) is not a torch optimizer")
    # return optimizer
    return optimizerMap[optimizer_name]

def torch_loss_function_name_map(loss_function_name: str, debug: bool = True) -> type[nn.Module]:
    R"""maps loss_function_name to valid torch loss functions

    Args:
        loss_function_name:str, valid torch loss function
    """
    # map for iterating
    lossFunctionMap = {
        "ce" : nn.CrossEntropyLoss,
        "bce" : nn.BCEWithLogitsLoss
    }
    # convert to lowercase
    loss_function_name = loss_function_name.lower()
    if(debug):
        if loss_function_name.lower() not in lossFunctionMap.keys():
            logging.error(f"PARAM(loss_function_name)({loss_function_name}) is not a torch loss function")
            raise KeyError(f"PARAM(loss_function_name)({loss_function_name}) is not a torch loss function")
    # return loss function
    return lossFunctionMap[loss_function_name]