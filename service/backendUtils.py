from typedefs import *
from typing import cast
from config import logging

def parseFromLayerConfig(layer_config: dict, debug: bool = True) -> LayerConfig:
    R"""Transforms the given layer_config dict into structured kwargs for passing to any other
        function that requires this

    Returns:
        tuple[layer name, layer config object for this layer]
    """
    layerName = cast(str, layer_config.get("type"))

    if layerName == "linear":
        # create kwargs object for linear layer
        kwargs = cast(LayerKwargs_T, {
            "in_features" : layer_config["in_features"], 
            "out_features": layer_config["out_features"],
            # bias is optional, we default it to True
            "bias" : layer_config.get("bias", True)
        })
        # create the layerConfig object
        layerConfig = cast(LayerConfig, {
            "name": layerName,
            "kwargs" : kwargs
        })
        return layerConfig
    else:
        raise NotImplementedError(f"layer({layerName}) is not implemented yet")
    
def parseFromTrainConfig(train_config: dict, debug: bool = True) -> TrainConfig:
    R"""Transforms the given train_config dict into structuredd kwargs for passing to any other 
    function that requires this
    """
    if(debug):
        logging.info(f"received train config {train_config}")

    optimizerConfig: dict = cast(dict, train_config.get("optimizerConfig"))
    
    optimizer: str = train_config["optimizer"]
    batchSize: int = train_config["batch_size"]
    epochs: int = train_config["epochs"]
    lossFunction: str = train_config["loss_function"]
    optimizerKwargs = {}
    for key in optimizerConfig:
        optimizerKwargs[key] = optimizerConfig[key]

    trainConfig: TrainConfig = cast(TrainConfig, {
        "epochs" : epochs,
        "batch_size": batchSize,
        "optimizer": optimizer,
        "optimizer_kwargs": optimizerKwargs,
        "loss_function": lossFunction  
    })
    if(debug):
        logging.info(f"parsed train config {trainConfig}")
    return trainConfig
    

def parseFromDataset(dataset: dict, debug: bool = True) -> Dataset:
    R"""Transforms the given dataset dict into structuredd kwargs for passing to any other 
    function that requires this
    """

    if(debug):
        logging.info(f"received dataset object: {dataset}")

    name: str = dataset["name"]

    parsedDatasetObj: Dataset = cast(Dataset, {
        "name": name
    })

    for key in OptionalDatasetKeys:
        if key in dataset.keys():
            parsedDatasetObj[key] = dataset[key]

    if(name == "mnist"):
        kwargs = cast(DatasetKwargs_T, {
            "root": dataset["root"]
        })
        for key in OptionalDatasetKwargsKeys:
            if key in dataset.keys():
                kwargs[key] = dataset[key]
    else:
        raise NotImplementedError(f"{name} dataset is not implemented yet")
    
    parsedDatasetObj["kwargs"] = kwargs

    if debug:
        logging.info(f"parsed dataset object: {parsedDatasetObj}")  
        
    return parsedDatasetObj
