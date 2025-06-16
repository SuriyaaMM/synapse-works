from config import logging
from typing import Union, TypedDict

from abc import ABC, abstractmethod

from torch import nn

class _LinearLayerConfig(TypedDict):
    in_features: int
    out_features:int
    bias: bool

class _DropoutLayerConfig(TypedDict):
    P: float

LayerKwargs = Union[_LinearLayerConfig, _DropoutLayerConfig]

def _torch_layer_name_map(layer_name: str, debug: bool = True) -> type[nn.Module]:
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

def transformLayerConfig(layer_config: dict, debug: bool = True) -> tuple[str, dict]:
    R"""Transforms the given layer_config dict into structured kwargs for passing to any other
        function that required this
    """

    """
        Schema for reference
            
            # LayerConfig input for collective layers
            input LayerConfig {
                type: String! # Linear, Conv2D
                linear: LinearLayerConfig # optional LinearLayerConfig
            }

            # LinearLayerConfig input 
            input LinearLayerConfig {
                inputDim: Int!      
                outputDim: Int!
                name: String
            }
    """
    layerType = layer_config.get("type")

    if layerType == "Linear":
        # get LinearLayerConfig
        linearConfig = layer_config.get(layerType)
        # create the kwarg object
        kwargs = {
            "in_features" : linearConfig["inputDim"], # type:ignore
            "out_features": linearConfig["outputDim"], # type:ignore
            "bias" : linearConfig["bias"] # type:ignore
        }

        return layerType, kwargs
    else:
        raise NotImplementedError(f"layer({layerType}) is not implemented yet")

class AbstractManager(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def appendLayer(self, layer_name: str, kwargs: LayerKwargs):
        pass

    @abstractmethod
    def deleteLayer(self, layer_name: str, kwargs: LayerKwargs):
        pass


class _TorchManager(AbstractManager):

    def __init__(self):
        super().__init__()
        self.layers = []        

    def appendLayer(self, layer_name: str, kwargs: LayerKwargs, debug: bool = True):
        R"""appends the layer to the model's layers

        Args:
            layer_name:str, layer name like linear, dropout, conv2d
        """
        if(debug):
            try:
                layer = _torch_layer_name_map(layer_name)(**kwargs)
            # ----- exceptions
            except TypeError as e:
                logging.error(f"type error {e}")
        else:
            layer = _torch_layer_name_map(layer_name)(**kwargs)

        # append to existing layers
        self.layers.append(layer)

    def deleteLayer(self, layer_name: str, kwargs: LayerKwargs, debug: bool = True):
        R"""deletes the layer_name from model's layers

        Args:
            layer_name:str, layer_name
        """
        if(debug):
            try:
                layer = _torch_layer_name_map(layer_name)(**kwargs)
            # ----- exceptions
            except TypeError as e:
                logging.error(f"type error {e}")
        else:
            layer = _torch_layer_name_map(layer_name)(**kwargs)

        # search in layers & remove it
        if layer in self.layers:
            self.layers.remove(layer)
            logging.info(f"removed layer {layer}")

class ModelManager(object):

    def __init__(self, model_id: str, model_name: str, backend: str = "torch"):
        R"""ModelManager handles model creation, deletion & modification

        Args:
            model_id: str, graphql object modelId
            model_name: str, graphql object modelName
            backend: str, graphql object backend
        """
        self.debug = True
        self.id = model_id
        self.mame = model_name
        self.backend = backend

        if backend == "torch":
            self._InternalManager = _TorchManager()
        else:
            raise NotImplementedError(f"{backend} is not implemented yet")

    def appendLayer(self, layer_name: str, kwargs:LayerKwargs):
        R"""Appends the layer to model
        
        Args:
            layer_name: str, graphql object layerName
        """
        self._InternalManager.appendLayer(layer_name=layer_name, kwargs=kwargs, debug=self.debug)
    
    def deleteLayer(self, layer_name: str, kwargs: LayerKwargs):
        R"""Deletes the layer in model

        Args:
            layer_name: str, graphql object layerName
        """
        self._InternalManager.deleteLayer(layer_name=layer_name, kwargs=kwargs, debug=self.debug)


