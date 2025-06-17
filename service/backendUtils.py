from typedefs import TrainConfig

def parseKwargsFromLayerConfig(layer_config: dict, debug: bool = True) -> tuple[str, dict]:
    R"""Transforms the given layer_config dict into structured kwargs for passing to any other
        function that requires this
    """
    layerType = layer_config.get("type")

    if layerType == "linear":
        # get LinearLayerConfig
        linearConfig = layer_config.get(layerType)
        # create the kwarg object
        kwargs = {
            "in_features" : layer_config.get("in_features"), # type:ignore
            "out_features": layer_config.get("out_features"), # type:ignore
            "bias" : layer_config.get("bias", True) # type:ignore
        }
        return layerType, kwargs
    else:
        raise NotImplementedError(f"layer({layerType}) is not implemented yet")

def parseKwargsFromTrainConfig(train_config: dict, debug: bool = True) -> TrainConfig:
    R"""Transforms the given train_config dict into structuredd kwargs for passing to any other 
    function that requires this
    """
    optimizerConfig = train_config.get("optimizerConfig")

    trainConfigObj: TrainConfig = {
        "epochs" :  train_config.get("epochs"), 
        "optimizerConfig" :  {
            "name": optimizerConfig.get("name"), # type:ignore
            "lr" : optimizerConfig.get("lr") # type:ignore
        },
        "loss_function": train_config.get("loss_function"),
        "batch_size" : train_config.get("batch_size")
    }

    return trainConfigObj

