from typing import TypedDict, Union

class _LinearLayerConfig(TypedDict):
    in_features: int
    out_features:int
    bias: bool

class _DropoutLayerConfig(TypedDict):
    p: float

# refer: https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html
class _MNISTDatasetConfig(TypedDict):
    root: str
    train: bool
    download: bool

LayerKwargs = Union[_LinearLayerConfig, _DropoutLayerConfig]
DatasetKwargs = Union[_MNISTDatasetConfig]

class OptimizerConfig(TypedDict):
    name: str
    lr: float

class TrainConfig(TypedDict):
    epochs: int
    batch_size: int
    optimizerConfig: OptimizerConfig
    loss_function: str

class DatasetConfig(TypedDict):
    name: str
    split_length: list
    shuffle: bool 
    kwargs: DatasetKwargs
