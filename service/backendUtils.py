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
        "id" : layer_config["id"],
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
        optional_keys = ["bias"]
        for optional_key in optional_keys:
            if optional_key in layer_config.keys():
                kwargs[optional_key] = layer_config[optional_key]
            else:
                logging.warning(f"{optional_key} not found in {layer_config.__class__.__name__}")
    # handle for conv2d layer
    elif layer_type == "conv2d":
        layer_config = cast(TSConv2dLayerInput, layer_config)
        # conv2d layer specific kwargs object
        kwargs = cast(Conv2dLayerKwargs, {
            "in_channels": layer_config["in_channels"],
            "out_channels": layer_config["out_channels"],
            "kernel_size": layer_config["kernel_size"]
        })
        # optional configurations
        optional_keys = ["stride", "padding", "dilation", "groups", "bias", "padding_mode"]
        for optional_key in optional_keys:
            if optional_key in layer_config.keys():
                kwargs[optional_key] = layer_config[optional_key]
            else:
                logging.warning(f"{optional_key} not found in {layer_config.__class__.__name__}")
    # handle for convtranspose2d
    elif layer_type == "convtranspose2d":
        layer_config = cast(TSConvTranspose2dLayerInput, layer_config)
        # conv2d layer specific kwargs object
        kwargs = cast(Conv2dLayerKwargs, {
            "in_channels": layer_config["in_channels"],
            "out_channels": layer_config["out_channels"],
            "kernel_size": layer_config["kernel_size"]
        })
        # optional configurations
        optional_keys = ["stride", "padding", "dilation", "groups", "bias", "output_padding"]
        for optional_key in optional_keys:
            if optional_key in layer_config.keys():
                kwargs[optional_key] = layer_config[optional_key]
            else:
                logging.warning(f"{optional_key} not found in {layer_config.__class__.__name__}")
    # handle for conv1d layer
    elif layer_type == "conv1d":
        layer_config = cast(TSConv1dLayerInput, layer_config)
        # conv1d layer specific kwargs object
        kwargs = cast(Conv1dLayerKwargs, {
            "in_channels": layer_config["in_channels"],
            "out_channels": layer_config["out_channels"],
            "kernel_size": layer_config["kernel_size"]
        })
        # optional configurations
        optional_keys = ["stride", "padding", "dilation", "groups", "bias", "padding_mode"]
        for optional_key in optional_keys:
            if optional_key in layer_config.keys():
                kwargs[optional_key] = layer_config[optional_key]
            else:
                logging.warning(f"{optional_key} not found in {layer_config.__class__.__name__}")
    # handle for maxpool2d layer
    elif layer_type == "maxpool2d":
        layer_config = cast(TSMaxPool2dLayerInput, layer_config)
        # maxpool2d layer specific kwargs object
        kwargs = cast(MaxPool2dLayerKwargs, {
            "kernel_size": layer_config["kernel_size"]
        })
        # optional configurations
        optional_keys = ["stride", "padding", "dilation", "return_indices", "ceil_mode"]
        for optional_key in optional_keys:
            if optional_key in layer_config.keys():
                kwargs[optional_key] = layer_config[optional_key]
            else:
                logging.warning(f"{optional_key} not found in {layer_config.__class__.__name__}")
    # handle for maxpool1d layer
    elif layer_type == "maxpool1d":
        layer_config = cast(TSMaxPool1dLayerInput, layer_config)
        # maxpool1d layer specific kwargs object
        kwargs = cast(MaxPool1dLayerKwargs, {
            "kernel_size": layer_config["kernel_size"]
        })
        # optional configurations
        optional_keys = ["stride", "padding", "dilation", "return_indices", "ceil_mode"]
        for optional_key in optional_keys:
            if optional_key in layer_config.keys():
                kwargs[optional_key] = layer_config[optional_key]
            else:
                logging.warning(f"{optional_key} not found in {layer_config.__class__.__name__}")
    # handle for avgpool2d layer
    elif layer_type == "avgpool2d":
        layer_config = cast(TSAvgPool2dLayerInput, layer_config)
        # avgpool2d layer specific kwargs object
        kwargs = cast(AvgPool2dLayerKwargs, {
            "kernel_size": layer_config["kernel_size"]
        })
        # optional configurations
        optional_keys = ["stride", "padding", "count_include_pad", "divisor_override", "ceil_mode"]
        for optional_key in optional_keys:
            if optional_key in layer_config.keys():
                kwargs[optional_key] = layer_config[optional_key]
            else:
                logging.warning(f"{optional_key} not found in {layer_config.__class__.__name__}")
    # handle for avgpool1d layer
    elif layer_type == "avgpool1d":
        layer_config = cast(TSAvgPool1dLayerInput, layer_config)
        # avgpool1d layer specific kwargs object
        kwargs = cast(AvgPool1dLayerKwargs, {
            "kernel_size": layer_config["kernel_size"]
        })
        # optional configurations
        optional_keys = ["stride", "padding", "count_include_pad", "divisor_override", "ceil_mode"]
        for optional_key in optional_keys:
            if optional_key in layer_config.keys():
                kwargs[optional_key] = layer_config[optional_key]
            else:
                logging.warning(f"{optional_key} not found in {layer_config.__class__.__name__}")
    # handle for batchnorm2d layer
    elif layer_type == "batchnorm2d":
        layer_config = cast(TSBatchNorm2dLayerInput, layer_config)
        # batchnorm2d layer specific kwargs object
        kwargs = cast(BatchNorm2dLayerKwargs, {
            "num_features": layer_config["num_features"]
        })
        # optional configurations
        optional_keys = ["eps", "momentum", "affine", "track_running_status"]
        for optional_key in optional_keys:
            if optional_key in layer_config.keys():
                kwargs[optional_key] = layer_config[optional_key]
            else:
                logging.warning(f"{optional_key} not found in {layer_config.__class__.__name__}")
    # handle for batchnorm1d layer
    elif layer_type == "batchnorm1d":
        layer_config = cast(TSBatchNorm1dLayerInput, layer_config)
        # batchnorm1d layer specific kwargs object
        kwargs = cast(BatchNorm1dLayerKwargs, {
            "num_features": layer_config["num_features"]
        })
        # optional configurations
        optional_keys = ["eps", "momentum", "affine", "track_running_status"]
        for optional_key in optional_keys:
            if optional_key in layer_config.keys():
                kwargs[optional_key] = layer_config[optional_key]
            else:
                logging.warning(f"{optional_key} not found in {layer_config.__class__.__name__}")
    # handle for flatten layer
    elif layer_type == "flatten":
        layer_config = cast(TSFlattenLayerInput, layer_config)
        # batchnorm2d layer specific kwargs object
        kwargs = cast(FlattenLayerKwargs, {})
        # optional configurations
        optional_keys = ["start_dim", "end_dim"]
        for optional_key in optional_keys:
            if optional_key in layer_config.keys():
                kwargs[optional_key] = layer_config[optional_key]
            else:
                logging.warning(f"{optional_key} not found in {layer_config.__class__.__name__}")
    # handle for dropout layer
    elif layer_type == "dropout":
        layer_config = cast(TSDropoutLayerInput, layer_config)
        # batchnorm2d layer specific kwargs object
        kwargs = cast(DropoutLayerKwargs, {})
        # optional configurations
        optional_keys = ["p"]
        for optional_key in optional_keys:
            if optional_key in layer_config.keys():
                kwargs[optional_key] = layer_config[optional_key]
            else:
                logging.warning(f"{optional_key} not found in {layer_config.__class__.__name__}")
    # handle for dropout layer
    elif layer_type == "dropout2d":
        layer_config = cast(TSDropout2dLayerInput, layer_config)
        # batchnorm2d layer specific kwargs object
        kwargs = cast(Dropout2dLayerKwargs, {})
        # optional configurations
        optional_keys = ["p"]
        for optional_key in optional_keys:
            if optional_key in layer_config.keys():
                kwargs[optional_key] = layer_config[optional_key]
            else:
                logging.warning(f"{optional_key} not found in {layer_config.__class__.__name__}")
    # handle for elu layer
    elif layer_type == "elu":
        layer_config = cast(TSELULayerInput, layer_config)
        # batchnorm2d layer specific kwargs object
        kwargs = cast(ELULayerKwargs, {})
        # optional configurations
        optional_keys = ["alpha", "inplace"]
        for optional_key in optional_keys:
            if optional_key in layer_config.keys():
                kwargs[optional_key] = layer_config[optional_key]
            else:
                logging.warning(f"{optional_key} not found in {layer_config.__class__.__name__}")
    # handle for relu layer
    elif layer_type == "relu":
        layer_config = cast(TSReLULayerInput, layer_config)
        # relu layer specific kwargs object
        kwargs = cast(ReLULayerKwargs, {})
        # optional configurations
        optional_keys = ["inplace"]
        for optional_key in optional_keys:
            if optional_key in layer_config.keys():
                kwargs[optional_key] = layer_config[optional_key]
            else:
                logging.warning(f"{optional_key} not found in {layer_config.__class__.__name__}")
    # handle for leakyrelu
    elif layer_type == "leakyrelu":
        layer_config = cast(TSLeakyReLULayerInput, layer_config)
        # leakyrelu layer specific kwargs object
        kwargs = cast(LeakyReLULayerKwargs, {})
        # optional configurations
        optional_keys = ["negative_slope", "inplace"]
        for optional_key in optional_keys:
            if optional_key in layer_config.keys():
                kwargs[optional_key] = layer_config[optional_key]
            else:
                logging.warning(f"{optional_key} not found in {layer_config.__class__.__name__}")
    # handle for sigmoid layer
    elif layer_type == "sigmoid":
        layer_config = cast(TSSigmoidLayerInput, layer_config)
        # leakyrelu layer specific kwargs object
        kwargs = cast(SigmoidLayerKwargs, {})
        # optional configurations
        optional_keys = []
        for optional_key in optional_keys:
            if optional_key in layer_config.keys():
                kwargs[optional_key] = layer_config[optional_key]
            else:
                logging.warning(f"{optional_key} not found in {layer_config.__class__.__name__}")
    # handle for logsigmoid layer
    elif layer_type == "logsigmoid":
        layer_config = cast(TSLogSigmoidLayerInput, layer_config)
        # leakyrelu layer specific kwargs object
        kwargs = cast(LogSigmoidLayerKwargs, {})
        # optional configurations
        optional_keys = []
        for optional_key in optional_keys:
            if optional_key in layer_config.keys():
                kwargs[optional_key] = layer_config[optional_key]
            else:
                logging.warning(f"{optional_key} not found in {layer_config.__class__.__name__}")
    # handle for tanh layer
    elif layer_type == "tanh":
        layer_config = cast(TSTanhLayerInput, layer_config)
        # leakyrelu layer specific kwargs object
        kwargs = cast(TanhLayerKwargs, {})
        # optional configurations
        optional_keys = []
        for optional_key in optional_keys:
            if optional_key in layer_config.keys():
                kwargs[optional_key] = layer_config[optional_key]
            else:
                logging.info(f"{optional_key} not found in {layer_config.__class__.__name__}")

    # handle for cat layer
    elif layer_type == "cat":
        layer_config = cast(TSCatLayerInput, layer_config)
        # leakyrelu layer specific kwargs object
        kwargs = cast(CatLayerKwargs, {})
        # optional configurations
        optional_keys = ["dimension"]
        for optional_key in optional_keys:
            if optional_key in layer_config.keys():
                kwargs[optional_key] = layer_config[optional_key]
            else:
                logging.info(f"{optional_key} not found in {layer_config.__class__.__name__}")
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
    metrics: TSTrainMetricsInput = train_config["metrics"]
    # put together train config
    parsed_train_config: TrainConfig = cast(TrainConfig, {
        "epochs" : epochs,
        "optimizer": optimizer,
        "loss_function": loss_function,
        "metrics" : metrics  
    })

    # optimizer dependent configuration
    optimizer_config: TSOptimizerConfigInput = train_config["optimizer_config"]
    loss_function_config: TSLossConfigInput = train_config["loss_function_config"] # type:ignore

    # handle for adadelta optimizer
    if optimizer == "adadelta":
        optimizer_kwargs = cast(AdadeltaOptimizerKwargs_T, {
            "lr" : optimizer_config["lr"]
        })
        optional_keys = ["rho", "eps", "weight_decay"]
        for optional_key in optional_keys:
            if optional_key in optimizer_config.keys():
                optimizer_kwargs[optional_key] = optimizer_config[optional_key]
            else:
                logging.warning(f"{optional_key} is missing in {optimizer_config.__class__.__name__}, skipping")
    # handle for adafactor optimizer
    elif optimizer == "adafactor":
        optimizer_kwargs = cast(AdafactorOptimizerKwargs_T, {
            "lr" : optimizer_config["lr"]
        })
        optional_keys = ["beta2_decay", "eps", "d", "weight_decay"]
        for optional_key in optional_keys:
            if optional_key in optimizer_config.keys():
                optimizer_kwargs[optional_key] = optimizer_config[optional_key]
            else:
                logging.warning(f"{optional_key} is missing in {optimizer_config.__class__.__name__}, skipping")
    # handle for adam optimizer
    elif optimizer == "adam":
        optimizer_kwargs = cast(AdamOptimizerKwargs_T, {
            "lr" : optimizer_config["lr"]
        })
        optional_keys = ["betas", "eps", "weight_decay"]
        for optional_key in optional_keys:
            if optional_key in optimizer_config.keys():
                optimizer_kwargs[optional_key] = optimizer_config[optional_key]
            else:
                logging.warning(f"{optional_key} is missing in {optimizer_config.__class__.__name__}, skipping")
    # handle for adamw optimizer
    elif optimizer == "adamw":
        optimizer_kwargs = cast(AdamWOptimizerKwargs_T, {
            "lr" : optimizer_config["lr"]
        })
        optional_keys = ["betas", "eps", "weight_decay"]
        for optional_key in optional_keys:
            if optional_key in optimizer_config.keys():
                optimizer_kwargs[optional_key] = optimizer_config[optional_key]
            else:
                logging.warning(f"{optional_key} is missing in {optimizer_config.__class__.__name__}, skipping")
    elif optimizer == "sparseadam":
        optimizer_kwargs = cast(SparseAdamOptimizerKwargs_T, {
            "lr" : optimizer_config["lr"]
        })
        optional_keys = ["betas", "eps", "weight_decay"]
        for optional_key in optional_keys:
            if optional_key in optimizer_config.keys():
                optimizer_kwargs[optional_key] = optimizer_config[optional_key]
            else:
                logging.warning(f"{optional_key} is missing in {optimizer_config.__class__.__name__}, skipping")
    # handle for adamax optimizer
    elif optimizer == "adamax":
        optimizer_kwargs = cast(AdamaxOptimizerKwargs_T, {
            "lr" : optimizer_config["lr"]
        })
        optional_keys = ["betas", "eps", "weight_decay"]
        for optional_key in optional_keys:
            if optional_key in optimizer_config.keys():
                optimizer_kwargs[optional_key] = optimizer_config[optional_key]
            else:
                logging.warning(f"{optional_key} is missing in {optimizer_config.__class__.__name__}, skipping")
    # handle for asgd optimizer
    elif optimizer == "asgd":
        optimizer_kwargs = cast(ASGDOptimizerKwargs_T, {
            "lr" : optimizer_config["lr"]
        })
        optional_keys = ["lambd", "alpha", "t0", "weight_decay"]
        for optional_key in optional_keys:
            if optional_key in optimizer_config.keys():
                optimizer_kwargs[optional_key] = optimizer_config[optional_key]
            else:
                logging.warning(f"{optional_key} is missing in {optimizer_config.__class__.__name__}, skipping")
    # handle for lbfgs optimizer
    elif optimizer == "lbfgs":
        optimizer_kwargs = cast(LBFGSOptimizerKwargs_T, {
            "lr" : optimizer_config["lr"]
        })
        optional_keys = ["max_iter", "max_eval", "tolerance_grad", "tolerance_change"]
        for optional_key in optional_keys:
            if optional_key in optimizer_config.keys():
                optimizer_kwargs[optional_key] = optimizer_config[optional_key]
            else:
                logging.warning(f"{optional_key} is missing in {optimizer_config.__class__.__name__}, skipping")
    # handle for radam optimizer
    elif optimizer == "radam":
        optimizer_kwargs = cast(RAdamOptimizerKwargs_T, {
            "lr" : optimizer_config["lr"]
        })
        optional_keys = ["betas", "eps", "weight_decay"]
        for optional_key in optional_keys:
            if optional_key in optimizer_config.keys():
                optimizer_kwargs[optional_key] = optimizer_config[optional_key]
            else:
                logging.warning(f"{optional_key} is missing in {optimizer_config.__class__.__name__}, skipping")
    # handle for rmsprop optimizer
    elif optimizer == "rmsprop":
        optimizer_kwargs = cast(RMSpropOptimizerKwargs_T, {
            "lr" : optimizer_config["lr"]
        })
        optional_keys = ["alpha", "eps", "weight_decay", "momentum"]
        for optional_key in optional_keys:
            if optional_key in optimizer_config.keys():
                optimizer_kwargs[optional_key] = optimizer_config[optional_key]
            else:
                logging.warning(f"{optional_key} is missing in {optimizer_config.__class__.__name__}, skipping")
    # handle for rprop optimizer
    elif optimizer == "rprop":
        optimizer_kwargs = cast(RpropOptimizerKwargs_T, {
            "lr" : optimizer_config["lr"]
        })
        optional_keys = ["etas", "step_sizes"]
        for optional_key in optional_keys:
            if optional_key in optimizer_config.keys():
                optimizer_kwargs[optional_key] = optimizer_config[optional_key]
            else:
                logging.warning(f"{optional_key} is missing in {optimizer_config.__class__.__name__}, skipping")
    # handle for sgd optimizer
    elif optimizer == "sgd":
        optimizer_kwargs = cast(SGDOptimizerKwargs_T, {
            "lr" : optimizer_config["lr"]
        })
        optional_keys = ["momentum", "dampening", "weight_decay", "nesterov"]
        for optional_key in optional_keys:
            if optional_key in optimizer_config.keys():
                optimizer_kwargs[optional_key] = optimizer_config[optional_key]
            else:
                logging.warning(f"{optional_key} is missing in {optimizer_config.__class__.__name__}, skipping")
    else:
        raise NotImplementedError(f"{optimizer} is not implemented yet")
    
    # handle loss function configurations
    if loss_function == "ce":
        loss_function_kwargs = {}
        optional_keys = ["reduction", "ignore_index", "label_smoothing"]
        for optional_key in optional_keys:
            if optional_key in loss_function_config.keys():
                loss_function_kwargs[optional_key] = loss_function_config[optional_key]
            else:
                logging.warning(f"{optional_key} is missing in {loss_function_config.__class__.__name__}, skipping")
        
    # set optimizer-specific kwargs
    parsed_train_config["optimizer_kwargs"] = optimizer_kwargs
    parsed_train_config["loss_function_kwargs"] = loss_function_kwargs # type:ignore
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
        "dataloader_config" : {"num_workers": 8, "pin_memory" : True}
    }) 

    # optional kwargs
    if "batch_size" in dataset_config.keys():
        parsed_dataset_config["dataloader_config"]["batch_size"] = dataset_config["batch_size"] # type:ignore
    if "split_length" in dataset_config.keys():
        parsed_dataset_config["split_length"] = dataset_config["split_length"] # type:ignore
    else:
        parsed_dataset_config["split_length"] = [0.8, 0.2]
    if "shuffle" in dataset_config.keys():
        parsed_dataset_config["dataloader_config"]["shuffle"] = dataset_config["shuffle"] # type:ignore

    # handle for mnist dataset
    if name == "mnist":
        kwargs = cast(MNISTDatasetConfig, {
            "root": root,
            # TODO(mms)
            "transform" : torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()])
        })
        # optional configurations
        optional_keys = ["download", "train"]
        for optional_key in optional_keys:
            if optional_key in dataset_config.keys():
                kwargs[optional_key] = dataset_config[optional_key]
            else:
                logging.warning(f"{optional_key} is missing in {dataset_config.__class__.__name__}, skipping")
    # handle for cifar10 dataset
    elif name == "cifar10":
        kwargs = cast(CIFAR10DatasetConfig, {
            "root": root,
            # TODO(mms)
            "transform" : torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()])
        })
        optional_keys = ["download", "train"]
        for optional_key in optional_keys:
            if optional_key in dataset_config.keys():
                kwargs[optional_key] = dataset_config[optional_key]
            else:
                logging.warning(f"{optional_key} is missing in {dataset_config.__class__.__name__}, skipping")
    # handle for celeba dataset
    elif name == "celeba":
        kwargs = cast(CelebADatasetConfig, {
            "root": root,
            # TODO(mms)
            "transform" : torchvision.transforms.Compose([
            torchvision.transforms.PILToTensor()]), 
            "target_transform" : torchvision.transforms.Compose([
                torchvision.transforms.PILToTensor()])
        })
        optional_keys = ["download", "target_type"]
        for optional_key in optional_keys:
            if optional_key in dataset_config.keys():
                kwargs[optional_key] = dataset_config[optional_key]
            else:
                logging.warning(f"{optional_key} is missing in {dataset_config.__class__.__name__}, skipping")
    # handle for vocsegmentation dataset
    elif name == "vocsegmentation":
        kwargs = cast(VOCSegmentationDatasetConfig, {
            "root": root,
            # TODO(mms)
            "transform" : torchvision.transforms.Compose([
            torchvision.transforms.Resize((64, 64)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0,0,0], std=[1,1,1])]), 
            "target_transform" : torchvision.transforms.Compose([
                torchvision.transforms.Resize((128, 128), interpolation=torchvision.transforms.InterpolationMode.NEAREST),
                torchvision.transforms.PILToTensor(),
                torchvision.transforms.Lambda(lambda t: t.squeeze(0).long())])
        })
        optional_keys = ["download", "image_set", "year"]
        for optional_key in optional_keys:
            if optional_key in dataset_config.keys():
                kwargs[optional_key] = dataset_config[optional_key]
            else:
                logging.warning(f"{optional_key} is missing in {dataset_config.__class__.__name__}, skipping")
    # handle for custom_csv
    elif name == "custom_csv":
        kwargs = cast(CustomCSVDatasetConfig, {
            "root": root,
            "feature_columns" : dataset_config["feature_columns"], # type:ignore
            "label_columns" : dataset_config["label_columns"], # type:ignore
            "is_regression_task" : dataset_config["is_regression_task"] # type:ignore
        })
        optional_keys = []
        for optional_key in optional_keys:
            if optional_key in dataset_config.keys():
                kwargs[optional_key] = dataset_config[optional_key]
            else:
                logging.warning(f"{optional_key} is missing in {dataset_config.__class__.__name__}, skipping")
    # handle for image_folder
    elif name == "image_folder":
        kwargs = cast(ImageFolderDatasetConfig, {
            "root": root,
            # TODO(mms)
            "transform" : torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()])
        })
        optional_keys = ["allow_empty"]
        for optional_key in optional_keys:
            if optional_key in dataset_config.keys():
                kwargs[optional_key] = dataset_config[optional_key]
            else:
                logging.warning(f"{optional_key} is missing in {dataset_config.__class__.__name__}, skipping")
    else:
        raise NotImplementedError(f"{name} dataset is not implemented yet")
    
    # add dataset specific kwargs
    parsed_dataset_config["kwargs"] = kwargs
    logging.info(f"Parsed PARAM(dataset_config):\n{json.dumps(parsed_dataset_config, indent=4, default=custom_json_encoder)}")
        
    return parsed_dataset_config