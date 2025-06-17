from typing import TypedDict, Union, Optional, Literal

""" ------------------------------------ graphql schema -----------------------------------
input LinearLayerConfig {
    in_features: Int!      
    out_features: Int!
    bias: Boolean
    name: String
}

input LayerConfig {
    type: String! # Linear, Conv2D
    linear: LinearLayerConfig # optional LinearLayerConfig
}
"""
class LinearLayerKwargs(TypedDict):
    in_features: int
    out_features:int
    bias: Optional[bool]

class DropoutLayerKwargs(TypedDict):
    p: float

LayerKwargs_T = Union[LinearLayerKwargs, DropoutLayerKwargs]
class LayerConfig(TypedDict):
    name: str
    kwargs: LayerKwargs_T

""" ------------------------------------ Train Configurations ----------------------------
    ------------------------------------ graphql schema -----------------------------------
    type OptimizerConfig {
        lr: Float!
    }
    type TrainConfig {
        epochs: Int!
        batch_size: Int!
        optimizer: String!
        optimizerConfig: OptimizerConfig!
        loss_function: String!
    }
"""
class OptimizerKwargs_T(TypedDict):
    lr: float

class TrainConfig(TypedDict):
    epochs: int
    batch_size: int
    optimizer: str
    optimizer_kwargs: OptimizerKwargs_T
    loss_function: str

""" ------------------------------------ Dataset Configurations ----------------------------
"""
# refer: https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html
class _MNISTDatasetConfig(TypedDict):
    root: str
    train: bool
    download: bool

DatasetKwargs_T = Union[_MNISTDatasetConfig]

class DatasetConfig(TypedDict):
    name: str
    split_length: list
    shuffle: bool 
    kwargs: DatasetKwargs_T


