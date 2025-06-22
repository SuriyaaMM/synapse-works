from typing import TypedDict, Union, Optional, NotRequired, Tuple

""" ------------------------------------ Layer Config ----------------------------------- """
class LinearLayerKwargs(TypedDict):
    in_features: int
    out_features:int
    bias: NotRequired[bool]

class Conv2dLayerKwargs(TypedDict):
    in_channels: int
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: NotRequired[Tuple[int, ...]]
    padding: NotRequired[Tuple[int, ...]]
    dilation: NotRequired[Tuple[int, ...]]
    groups: NotRequired[Tuple[int, ...]]
    bias: NotRequired[bool]
    padding_mode: NotRequired[str]

class DropoutLayerKwargs(TypedDict):
    p: float

LayerKwargs_T = Union[LinearLayerKwargs, Conv2dLayerKwargs, DropoutLayerKwargs]
class LayerConfig(TypedDict):
    type: str
    kwargs: LayerKwargs_T

class TSLinearLayerInput(TypedDict):
    id: str
    type: str
    name: str
    in_features: int
    out_features: int
    bias: NotRequired[bool]

class TSConv2dLayerInput(TypedDict):
    id: str
    type: str
    name: NotRequired[str]
    in_channels: int
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: NotRequired[Tuple[int, ...]]
    padding: NotRequired[Tuple[int, ...]]
    dilation: NotRequired[Tuple[int, ...]]
    groups: NotRequired[Tuple[int, ...]]
    bias: NotRequired[bool]
    padding_mode: NotRequired[str]

class TSDropoutLayerInput(TypedDict):
    id: str
    type: str
    name: str
    p: float

TSLayerInput = Union[TSLinearLayerInput, TSConv2dLayerInput, TSDropoutLayerInput]

""" ------------------------------------ Train Configurations ---------------------------- """
class AdamOptimizerKwargs_T(TypedDict):
    lr: float

class SGDOptimizerKwargs_T(TypedDict):
    momentum: float

OptimizerKwargs_T = Union[AdamOptimizerKwargs_T | SGDOptimizerKwargs_T]
class TrainConfig(TypedDict):
    epochs: int
    optimizer: str
    optimizer_kwargs: OptimizerKwargs_T
    loss_function: str

class TSOptimizerConfigInput(TypedDict):
    lr: float
class TSTrainConfigInput(TypedDict):
    epochs: int
    optimizer: str
    optimizer_config: TSOptimizerConfigInput
    loss_function: str

""" ------------------------------------ Dataset Configurations ---------------------------- """
# refer: https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html
class MNISTDatasetConfig(TypedDict):
    root: str
    train: Optional[bool]
    download: Optional[bool]

class CIFAR10DatasetConfig(TypedDict):
    root: str
    train: Optional[bool]
    download: Optional[bool]

DatasetKwargs_T = Union[MNISTDatasetConfig, CIFAR10DatasetConfig]

class DataLoaderConfig(TypedDict):
    batch_size: Optional[int]
    shuffle: Optional[bool] 

class DatasetConfig(TypedDict):
    name: str
    dataloader_config: DataLoaderConfig
    split_length: list[float | int]
    kwargs: DatasetKwargs_T

class TSMNISTDatasetInput(TypedDict):
    name: str
    batch_size: Optional[int]
    split_length: Optional[list[float]]
    shuffle: Optional[bool]
    root: str
    train: Optional[bool]
    download: Optional[bool]

class TSCIFAR10DatasetInput(TypedDict):
    name: str
    batch_size: Optional[int]
    split_length: Optional[list[float]]
    shuffle: Optional[bool]
    root: str
    train: Optional[bool]
    download: Optional[bool]

TSDatasetInput = Union[TSMNISTDatasetInput, TSCIFAR10DatasetInput]