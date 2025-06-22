from typedefs import *
from typing import cast
from config import logging
import json

def parseFromLayerConfig(layer_config: TSLayerInput, debug: bool = True) -> LayerConfig:
    R"""Transforms the given layer_config dict into structured kwargs for passing to any other
        function that requires this

    Returns:
        LayerConfig object
    """
    logging.info(f"Received PARAM(layer_config):\n{json.dumps(layer_config, indent=4)}")

    layer_type = layer_config["type"]
    # put together layer config
    parsed_layer_config: LayerConfig = cast(LayerConfig, {
        "type": layer_type
    })

    # handle for linear layer
    if layer_type == "linear":
        layer_config = cast(TSLinearLayerInput, layer_config)
        # linear layer specific kwargs object
        kwargs = cast(LayerKwargs_T, {
            "in_features" : layer_config["in_features"], 
            "out_features": layer_config["out_features"],
        })
        # optional configurations
        if "bias" in layer_config.keys():
            kwargs["bias"] = layer_config["bias"] # type:ignore

    elif layer_type == "conv2d":
        layer_config = cast(TSConv2dLayerInput, layer_config)
        # conv2d layer specific kwargs object
        kwargs = cast(Conv2dLayerKwargs, {
            "in_channels": layer_config["in_channels"],
            "out_channels": layer_config["out_channels"],
            "kernel_size": layer_config["kernel_size"]
        })
        # optional configurations
        if "stride" in layer_config.keys():
            kwargs["stride"] = layer_config["stride"] # type:ignore

        if "padding" in layer_config.keys():
            kwargs["padding"] = layer_config["padding"] # type:ignore

        if "dilation" in layer_config.keys():
            kwargs["dilation"] = layer_config["dilation"] # type:ignore

        if "groups" in layer_config.keys():
            kwargs["groups"] = layer_config["groups"] # type:ignore

        if "bias" in layer_config.keys():
            kwargs["bias"] = layer_config["bias"] # type:ignore
            
        if "padding_mode" in layer_config.keys():
            kwargs["padding_mode"] = layer_config["padding_mode"] # type:ignore
    else:
        raise NotImplementedError(f"layer({layer_type}) is not implemented yet")
    
    # set layer specific kwargs
    parsed_layer_config["kwargs"] = kwargs
    logging.info(f"Parsed PARAM(layer_config):\n{json.dumps(parsed_layer_config, indent=4)}")

    return parsed_layer_config
    
def parseFromTrainConfig(train_config: TSTrainConfigInput) -> TrainConfig:
    R"""Transforms the given train_config dict into structuredd kwargs for passing to any other 
    function that requires this
    """
    logging.info(f"Received PARAM(train_config):\n{json.dumps(train_config, indent=4)}")

    # non optional settings
    optimizer: str = train_config["optimizer"]
    epochs: int = train_config["epochs"]
    loss_function: str = train_config["loss_function"]
    # put together train config
    parsed_train_config: TrainConfig = cast(TrainConfig, {
        "epochs" : epochs,
        "optimizer": optimizer,
        "loss_function": loss_function  
    })

    # optimizer dependent configuration
    optimizer_config: TSOptimizerConfigInput = train_config["optimizer_config"]

    # handle for adam optimizer
    if optimizer == "adam":
        optimizer_kwargs = cast(AdamOptimizerKwargs_T, {
            "lr" : optimizer_config["lr"]
        })
    else:
        raise NotImplementedError(f"{optimizer} is not implemented yet")
    
    # set optimizer-specific kwargs
    parsed_train_config["optimizer_kwargs"] = optimizer_kwargs
    logging.info(f"Parsed PARAM(train_config):\n{json.dumps(parsed_train_config, indent=4)}")
    
    return parsed_train_config
    
def parseFromDataset(dataset_config: TSDatasetInput) -> DatasetConfig:
    R"""Transforms the given dataset dict into structuredd kwargs for passing to any other 
    function that requires this
    """
    logging.info(f"Received PARAM(dataset_config):\n{json.dumps(dataset_config, indent=4)}")
    #TODO(mms) implement transforms support from str to torchvision.transforms.Compose mapping
    # non optional kwargs
    name: str = dataset_config["name"]
    root: str = dataset_config["root"]
    parsed_dataset_config: DatasetConfig = cast(DatasetConfig, {
        "name": name,
        "dataloader_config" : {"num_workers": 5, "pin_memory" : True}
    }) 

    # optional kwargs
    if "batch_size" in dataset_config.keys():
        parsed_dataset_config["dataloader_config"]["batch_size"] = dataset_config["batch_size"]
    if "split_length" in dataset_config.keys():
        parsed_dataset_config["split_length"] = dataset_config["split_length"] # type:ignore
    else:
        parsed_dataset_config["split_length"] = [0.8, 0.2]
    if "shuffle" in dataset_config.keys():
        parsed_dataset_config["dataloader_config"]["shuffle"] = dataset_config["shuffle"]

    # handle for mnist dataset
    if name == "mnist":
        kwargs = cast(MNISTDatasetConfig, {
            "root": root
        })
        # optional configurations
        if "train" in dataset_config.keys():
            kwargs["train"] = dataset_config["train"]
        if "download" in dataset_config.keys():
            kwargs["download"] = dataset_config["download"]

    elif name == "cifar10":
        kwargs = cast(CIFAR10DatasetConfig, {
            "root": root
        })
        # optional configurations
        if "train" in dataset_config.keys():
            kwargs["train"] = dataset_config["train"]
        if "download" in dataset_config.keys():
            kwargs["download"] = dataset_config["download"]
    else:
        raise NotImplementedError(f"{name} dataset is not implemented yet")
    
    # add dataset specific kwargs
    parsed_dataset_config["kwargs"] = kwargs
    if "transforms" in dataset_config.keys():
        parsed_dataset_config["kwargs"]["transform"] = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()])
    else:
        parsed_dataset_config["kwargs"]["transform"] = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()])
        
    logging.info(f"Parsed PARAM(dataset_config):\n{json.dumps(parsed_dataset_config, indent=4, default=custom_json_encoder)}")
        
    return parsed_dataset_config
