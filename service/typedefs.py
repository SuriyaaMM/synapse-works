from typing import TypedDict, Union, NotRequired, NotRequired, Tuple, Literal, Dict
from dataclasses import dataclass
# used in abstractManager.py
import torch
import torchvision

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

class ConvTranspose2dLayerKwargs(TypedDict):
    in_channels: int
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: NotRequired[Tuple[int, ...]]
    padding: NotRequired[Tuple[int, ...]]
    dilation: NotRequired[Tuple[int, ...]]
    groups: NotRequired[Tuple[int, ...]]
    bias: NotRequired[bool]
    output_padding: NotRequired[Tuple[int, ...]]

class Conv1dLayerKwargs(TypedDict):
    in_channels: int
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: NotRequired[Tuple[int, ...]]
    padding: NotRequired[Tuple[int, ...]]
    dilation: NotRequired[Tuple[int, ...]]
    groups: NotRequired[Tuple[int, ...]]
    bias: NotRequired[bool]
    padding_mode: NotRequired[str]

class MaxPool2dLayerKwargs(TypedDict):
    kernel_size: Tuple[int, ...]
    stride: NotRequired[Tuple[int, ...]]
    padding: NotRequired[Tuple[int, ...]]
    dilation: NotRequired[Tuple[int, ...]]
    return_indices: NotRequired[bool]
    ceil_mode: NotRequired[bool]

class MaxPool1dLayerKwargs(TypedDict):
    kernel_size: Tuple[int, ...]
    stride: NotRequired[Tuple[int, ...]]
    padding: NotRequired[Tuple[int, ...]]
    dilation: NotRequired[Tuple[int, ...]]
    return_indices: NotRequired[bool]
    ceil_mode: NotRequired[bool]

class AvgPool2dLayerKwargs(TypedDict):
    kernel_size: Tuple[int, ...]
    stride: NotRequired[Tuple[int, ...]]
    padding: NotRequired[Tuple[int, ...]]
    count_include_pad: NotRequired[bool]
    divisor_override: NotRequired[int]
    ceil_mode: NotRequired[bool]

class AvgPool1dLayerKwargs(TypedDict):
    kernel_size: Tuple[int, ...]
    stride: NotRequired[Tuple[int, ...]]
    padding: NotRequired[Tuple[int, ...]]
    count_include_pad: NotRequired[bool]
    divisor_override: NotRequired[int]
    ceil_mode: NotRequired[bool]

class BatchNorm2dLayerKwargs(TypedDict):
    num_features: int
    eps: NotRequired[float]
    momentum: NotRequired[float]
    affine: NotRequired[bool]
    track_running_status: NotRequired[bool]

class BatchNorm1dLayerKwargs(TypedDict):
    num_features: int
    eps: NotRequired[float]
    momentum: NotRequired[float]
    affine: NotRequired[bool]
    track_running_status: NotRequired[bool]

class FlattenLayerKwargs(TypedDict):
    start_dim: NotRequired[int]
    end_dim: NotRequired[int]

class DropoutLayerKwargs(TypedDict):
    p: NotRequired[float]

class Dropout2dLayerKwargs(TypedDict):
    p: NotRequired[float]

class ELULayerKwargs(TypedDict):
    alpha: NotRequired[float]
    inplace: NotRequired[bool]

class ReLULayerKwargs(TypedDict):
    inplace: NotRequired[bool]

class LeakyReLULayerKwargs(TypedDict):
    negative_slope: NotRequired[float]
    inplace: NotRequired[bool]

class SigmoidLayerKwargs(TypedDict):
    pass

class LogSigmoidLayerKwargs(TypedDict):
    pass

class TanhLayerKwargs(TypedDict):
    pass

class CatLayerKwargs(TypedDict):
    dimension: NotRequired[int]

LayerKwargs_T = Union[LinearLayerKwargs, 
                      Conv2dLayerKwargs,
                      ConvTranspose2dLayerKwargs,
                      Conv2dLayerKwargs,
                      MaxPool2dLayerKwargs,
                      MaxPool1dLayerKwargs,
                      AvgPool2dLayerKwargs,
                      AvgPool1dLayerKwargs,
                      BatchNorm2dLayerKwargs,
                      BatchNorm1dLayerKwargs,
                      FlattenLayerKwargs,
                      DropoutLayerKwargs,
                      Dropout2dLayerKwargs,
                      ELULayerKwargs,
                      ReLULayerKwargs,
                      LeakyReLULayerKwargs,
                      SigmoidLayerKwargs,
                      LogSigmoidLayerKwargs,
                      TanhLayerKwargs,
                      CatLayerKwargs]
class LayerConfig(TypedDict):
    id: str
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

class TSConvTranspose2dLayerInput(TypedDict):
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
    output_padding: NotRequired[Tuple[int, ...]]

class TSConv1dLayerInput(TypedDict):
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

class TSMaxPool2dLayerInput(TypedDict):
    id: str
    type: str
    name: NotRequired[str]
    kernel_size: Tuple[int, ...]
    stride: NotRequired[Tuple[int, ...]]
    padding: NotRequired[Tuple[int, ...]]
    dilation: NotRequired[Tuple[int, ...]]
    return_indices: NotRequired[bool]
    ceil_mode: NotRequired[bool]

class TSMaxPool1dLayerInput(TypedDict):
    id: str
    type: str
    name: NotRequired[str]
    kernel_size: Tuple[int, ...]
    stride: NotRequired[Tuple[int, ...]]
    padding: NotRequired[Tuple[int, ...]]
    dilation: NotRequired[Tuple[int, ...]]
    return_indices: NotRequired[bool]
    ceil_mode: NotRequired[bool]

class TSAvgPool2dLayerInput(TypedDict):
    id: str
    type: str
    name: NotRequired[str]
    kernel_size: Tuple[int, ...]
    stride: NotRequired[Tuple[int, ...]]
    padding: NotRequired[Tuple[int, ...]]
    count_include_pad: NotRequired[bool]
    divisor_override: NotRequired[int]
    ceil_mode: NotRequired[bool]

class TSAvgPool1dLayerInput(TypedDict):
    id: str
    type: str
    name: NotRequired[str]
    kernel_size: Tuple[int, ...]
    stride: NotRequired[Tuple[int, ...]]
    padding: NotRequired[Tuple[int, ...]]
    count_include_pad: NotRequired[bool]
    divisor_override: NotRequired[int]
    ceil_mode: NotRequired[bool]

class TSBatchNorm2dLayerInput(TypedDict):
    id: str
    type: str
    name: NotRequired[str]
    num_features: int
    eps: NotRequired[float]
    momentum: NotRequired[float]
    affine: NotRequired[bool]
    track_running_status: NotRequired[bool]

class TSBatchNorm1dLayerInput(TypedDict):
    id: str
    type: str
    name: NotRequired[str]
    num_features: int
    eps: NotRequired[float]
    momentum: NotRequired[float]
    affine: NotRequired[bool]
    track_running_status: NotRequired[bool]

class TSFlattenLayerInput(TypedDict):
    id: str
    type: str
    name: NotRequired[str]
    start_dim: NotRequired[int]
    end_dim: NotRequired[int]

class TSDropoutLayerInput(TypedDict):
    id: str
    type: str
    name: NotRequired[str]
    p: NotRequired[float]

class TSDropout2dLayerInput(TypedDict):
    id: str
    type: str
    name: NotRequired[str]
    p: NotRequired[float]

class TSELULayerInput(TypedDict):
    id: str
    type: str
    name: NotRequired[str]
    alpha: NotRequired[float]
    inplace: NotRequired[bool]

class TSReLULayerInput(TypedDict):
    id: str
    type: str
    name: NotRequired[str]
    inplace: NotRequired[bool]

class TSLeakyReLULayerInput(TypedDict):
    id: str
    type: str
    name: NotRequired[str]
    negative_slope: NotRequired[float]
    inplace: NotRequired[bool]

class TSSigmoidLayerInput(TypedDict):
    id: str
    type: str
    name: NotRequired[str]

class TSLogSigmoidLayerInput(TypedDict):
    id: str
    type: str
    name: NotRequired[str]

class TSTanhLayerInput(TypedDict):
    id: str
    type: str
    name: NotRequired[str]

class TSCatLayerInput(TypedDict):
    id: str
    type: str
    name: NotRequired[str]
    dimension: NotRequired[int]

TSLayerInput = Union[TSLinearLayerInput, 
                     TSConv2dLayerInput,
                     TSConvTranspose2dLayerInput,
                     TSConv1dLayerInput,
                     TSMaxPool2dLayerInput,
                     TSMaxPool1dLayerInput,
                     TSAvgPool2dLayerInput,
                     TSAvgPool1dLayerInput,
                     TSBatchNorm2dLayerInput,
                     TSBatchNorm1dLayerInput,
                     TSFlattenLayerInput,
                     TSDropout2dLayerInput, 
                     TSDropoutLayerInput,
                     TSELULayerInput,
                     TSReLULayerInput,
                     TSLeakyReLULayerInput,
                     TSSigmoidLayerInput,
                     TSLogSigmoidLayerInput,
                     TSTanhLayerInput,
                     TSCatLayerInput]

""" ------------------------------------ Train Configurations ---------------------------- """
class AdadeltaOptimizerKwargs_T(TypedDict):
    lr: float
    rho: NotRequired[float]
    eps: NotRequired[float]
    weight_decay: NotRequired[float]

class AdafactorOptimizerKwargs_T(TypedDict):
    lr: float
    beta2_decay: NotRequired[float]
    eps: NotRequired[float]
    d: NotRequired[float]
    weight_decay: NotRequired[float]

# TODO(mms) adagrad
class AdamOptimizerKwargs_T(TypedDict):
    lr: float
    betas: NotRequired[Tuple[float, float]]
    eps: NotRequired[float]
    weight_decay: NotRequired[float]

class AdamWOptimizerKwargs_T(TypedDict):
    lr: float
    betas: NotRequired[Tuple[float, float]]
    eps: NotRequired[float]
    weight_decay: NotRequired[float]

class SparseAdamOptimizerKwargs_T(TypedDict):
    lr: float
    betas: NotRequired[Tuple[float, float]]
    eps: NotRequired[float]
    weight_decay: NotRequired[float]

class AdamaxOptimizerKwargs_T(TypedDict):
    lr: float
    betas: NotRequired[Tuple[float, float]]
    eps: NotRequired[float]
    weight_decay: NotRequired[float]

class ASGDOptimizerKwargs_T(TypedDict):
    lr: float
    lambd: NotRequired[float]
    alpha: NotRequired[float]
    t0: NotRequired[float]
    weight_decay: NotRequired[float]

class LBFGSOptimizerKwargs_T(TypedDict):
    lr: float
    max_iter: NotRequired[int]
    max_eval: NotRequired[int]
    tolerance_grad: NotRequired[int]
    tolerance_change: NotRequired[int]

class RAdamOptimizerKwargs_T(TypedDict):
    lr: float
    betas: NotRequired[Tuple[float, float]]
    eps: NotRequired[float]
    weight_decay: NotRequired[float]

class RMSpropOptimizerKwargs_T(TypedDict):
    lr: float
    alpha: NotRequired[float]
    eps: NotRequired[float]
    weight_decay: NotRequired[float]
    momentum: NotRequired[float]

class RpropOptimizerKwargs_T(TypedDict):
    lr: float
    etas: NotRequired[Tuple[float, float]]
    step_sizes: NotRequired[Tuple[float, float]]

class SGDOptimizerKwargs_T(TypedDict):
    lr: float
    momentum: NotRequired[float]
    dampening: NotRequired[float]
    weight_decay: NotRequired[float]
    nesterov: NotRequired[bool]

OptimizerKwargs_T = Union[AdadeltaOptimizerKwargs_T,
                          AdafactorOptimizerKwargs_T,
                          AdamOptimizerKwargs_T,
                          AdamWOptimizerKwargs_T,
                          SparseAdamOptimizerKwargs_T,
                          AdamaxOptimizerKwargs_T,
                          ASGDOptimizerKwargs_T,
                          LBFGSOptimizerKwargs_T,
                          RAdamOptimizerKwargs_T,
                          RMSpropOptimizerKwargs_T,
                          RpropOptimizerKwargs_T,
                          SGDOptimizerKwargs_T]
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
    train: NotRequired[bool]
    download: NotRequired[bool]
    transform: torchvision.transforms.Compose

class CIFAR10DatasetConfig(TypedDict):
    root: str
    train: NotRequired[bool]
    download: NotRequired[bool]
    transform: torchvision.transforms.Compose

class CelebADatasetConfig(TypedDict):
    root: str
    target_type: NotRequired[list[str]]
    download: NotRequired[bool]
    transform: torchvision.transforms.Compose
    target_transform: torchvision.transforms.Compose

class VOCSegmentationDatasetConfig(TypedDict):
    name: str
    batch_size: NotRequired[int]
    split_length: NotRequired[list[float]]
    shuffle: NotRequired[bool]
    root: str
    image_set: NotRequired[str]
    year: NotRequired[str]
    download: NotRequired[bool]
    transform: torchvision.transforms.Compose
    target_transform: torchvision.transforms.Compose

class CustomCSVDatasetConfig(TypedDict):
    root: str
    feature_columns: list[str]
    label_columns: list[str]
    is_regression_task: bool

class ImageFolderDatasetConfig(TypedDict):
    root: str
    transform: torchvision.transforms.Compose
    allow_empty: NotRequired[bool]


DatasetKwargs_T = Union[MNISTDatasetConfig, 
                        CIFAR10DatasetConfig,
                        CelebADatasetConfig,
                        VOCSegmentationDatasetConfig,
                        CustomCSVDatasetConfig,
                        ImageFolderDatasetConfig]

class DataLoaderConfig(TypedDict):
    batch_size: NotRequired[int]
    shuffle: NotRequired[bool]
    num_workers: int
    pin_memory: bool

class DatasetConfig(TypedDict):
    name: str
    dataloader_config: DataLoaderConfig
    split_length: list[float | int]
    kwargs: DatasetKwargs_T

class TSMNISTDatasetInput(TypedDict):
    name: str
    batch_size: NotRequired[int]
    split_length: NotRequired[list[float]]
    shuffle: NotRequired[bool]
    transforms: list[str]
    root: str
    train: NotRequired[bool]
    download: NotRequired[bool]

class TSCIFAR10DatasetInput(TypedDict):
    name: str
    batch_size: NotRequired[int]
    split_length: NotRequired[list[float]]
    shuffle: NotRequired[bool]
    transforms: list[str]
    root: str
    train: NotRequired[bool]
    download: NotRequired[bool]

class TSCelebADatasetInput(TypedDict):
    name: str
    batch_size: NotRequired[int]
    split_length: NotRequired[list[float]]
    shuffle: NotRequired[bool]
    root: str
    target_type: NotRequired[list[str]]
    download: NotRequired[bool]
    transform: torchvision.transforms.Compose
    target_transform: torchvision.transforms.Compose

class TSVOCSegmentationDatasetInput(TypedDict):
    name: str
    batch_size: NotRequired[int]
    split_length: NotRequired[list[float]]
    shuffle: NotRequired[bool]
    root: str
    image_set: NotRequired[str]
    year: NotRequired[str]
    download: NotRequired[bool]
    transform: torchvision.transforms.Compose
    target_transform: torchvision.transforms.Compose

class TSCustomCSVDatasetInput(TypedDict):
    name: str
    batch_size: NotRequired[int]
    split_length: NotRequired[list[float]]
    shuffle: NotRequired[bool]
    root: str
    feature_columns: list[str]
    label_columns: list[str]
    is_regression_task: bool

class TSImageFolderDatasetInput(TypedDict):
    name: str
    batch_size: NotRequired[int]
    split_length: NotRequired[list[float]]
    shuffle: NotRequired[bool]
    root: str
    transform: list[str]
    allow_empty: NotRequired[bool]
    

TSDatasetInput = Union[TSMNISTDatasetInput, 
                       TSCIFAR10DatasetInput,
                       TSCelebADatasetInput,
                       TSVOCSegmentationDatasetInput,
                       TSCustomCSVDatasetInput,
                       TSImageFolderDatasetInput]


class TSTrainArgsInput(TypedDict):
    export_to: Literal["TorchTensor", "ONNX"]


# ----------------------------------- Module Graph --------------------------------
@dataclass
class ModuleAdjacencyList:
    source_id: str
    target_ids: list[str]
@dataclass
class ModuleGraph:
    layers: list[LayerConfig]
    edges: list[ModuleAdjacencyList]
    sorted_ids: list[str]

@dataclass
class ModuleGraphInput:
    layers: list[TSLayerInput]
    edges: list[ModuleAdjacencyList]
    sorted_ids: list[str]

def custom_json_encoder(obj):
    if isinstance(obj, (torchvision.transforms.Compose)):
        return f"{obj.__class__.__name__} is non serializeable"
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")