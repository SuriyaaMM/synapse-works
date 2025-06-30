import { Module } from "vm";

// ------------------------------- Layers & Configuration ----------------------------------
export interface LayerConfig {
    id: string;
    type: string;
    name?: string;
    in_dimension?: number[];
    out_dimension?: number[];
};

export interface LinearLayerConfig extends LayerConfig {
    in_features: number;
    out_features: number;
    bias?: boolean;
};

export interface Conv2dLayerConfig extends LayerConfig {
    in_channels: number;
    out_channels: number;
    kernel_size: number[];
    stride?: number[];
    padding?: number[];
    dilation?: number[];
    groups?: number[];
    bias?: boolean;
    padding_mode?: string;
};

export interface ConvTranspose2dLayerConfig extends LayerConfig {
    in_channels: number;
    out_channels: number;
    kernel_size: number[];
    stride?: number[];
    padding?: number[];
    dilation?: number[];
    groups?: number[];
    bias?: boolean;
    output_padding?: number[];
};

export interface Conv1dLayerConfig extends LayerConfig {
    in_channels: number;
    out_channels: number;
    kernel_size: number[];
    stride?: number[];
    padding?: number[];
    dilation?: number[];
    groups?: number[];
    bias?: boolean;
    padding_mode?: string;
};

export interface MaxPool2dLayerConfig extends LayerConfig {
    kernel_size: number[];
    stride?: number[];
    padding?: number[];
    dilation?: number[];
    return_indices?: boolean;
    ceil_mode?: boolean;
}

export interface MaxPool1dLayerConfig extends LayerConfig {
    kernel_size: number[];
    stride?: number[];
    padding?: number[];
    dilation?: number[];
    return_indices?: boolean;
    ceil_mode?: boolean;
}

export interface AvgPool2dLayerConfig extends LayerConfig {
    kernel_size: number[];
    stride?: number[];
    padding?: number[];
    count_include_pad?: boolean;
    divisor_override?: number;
    ceil_mode?: boolean;
}

export interface AvgPool1dLayerConfig extends LayerConfig {
    kernel_size: number[];
    stride?: number[];
    padding?: number[];
    count_include_pad?: boolean;
    divisor_override?: number;
    ceil_mode?: boolean;
}

export interface BatchNorm2dLayerConfig extends LayerConfig {
    num_features: number;
    eps?: number;
    momentum?: number;
    affine?: boolean;
    track_running_status?:boolean;
}

export interface BatchNorm1dLayerConfig extends LayerConfig {
    num_features: number;
    eps?: number;
    momentum?: number;
    affine?: boolean;
    track_running_status: boolean;
}

export interface FlattenLayerConfig extends LayerConfig {
    start_dim?: number;
    end_dim?: number;
}

export interface DropoutLayerConfig extends LayerConfig {
    p?: number;
}

export interface Dropout2dLayerConfig extends LayerConfig {
    p?: number;
}

export interface ELULayerConfig extends LayerConfig {
    alpha?: number;
    inplace?: boolean;
}

export interface ReLULayerConfig extends LayerConfig {
    inplace?: boolean;
}

export interface LeakyReLULayerConfig extends LayerConfig {
    negative_slope?: number;
    inplace?: boolean;
}


export interface SigmoidLayerConfig extends LayerConfig {}
export interface LogSigmoidLayerConfig extends LayerConfig {}
export interface TanhLayerConfig extends LayerConfig {}

export interface CatLayerConfig extends LayerConfig {
    dimension?: number;
}

export type LinearLayerConfigInput = {
    name?: string;
    in_features: number;
    out_features: number;
    bias?: boolean;
};

export type Conv2dLayerConfigInput = {
    name?: string;
    in_channels: number;
    out_channels: number;
    kernel_size: number[];
    stride?: number[];
    padding?: number[];
    dilation?: number[];
    groups?: number[];
    bias?: boolean;
    padding_mode?: string;
}

export type ConvTranspose2dLayerConfigInput = {
    name?: string;
    in_channels: number;
    out_channels: number;
    kernel_size: number[];
    stride?: number[];
    padding?: number[];
    dilation?: number[];
    groups?: number[];
    bias?: boolean;
    output_padding?: number[];
};

export type Conv1dLayerConfigInput  = {
    name?: string;
    in_channels: number;
    out_channels: number;
    kernel_size: number[];
    stride?: number[];
    padding?: number[];
    dilation?: number[];
    groups?: number[];
    bias?: boolean;
    padding_mode?: string;
};

export type MaxPool2dLayerConfigInput = {
    name?: string;
    kernel_size: number[];
    stride?: number[];
    padding?: number[];
    dilation?: number[];
    return_indices?: boolean;
    ceil_mode?: boolean;
}

export type MaxPool1dLayerConfigInput = {
    name?: string;
    kernel_size: number[];
    stride?: number[]
    padding?: number[];
    dilation?: number[];
    return_indices?: boolean;
    ceil_mode?: boolean;
}

export type AvgPool2dLayerConfigInput =  {
    name? : string;
    kernel_size: number[];
    stride?: number[]
    padding?: number[];
    count_include_pad?: boolean;
    divisor_override?: number;
    ceil_mode?: boolean;
}

export type AvgPool1dLayerConfigInput =  {
    name? : string;
    kernel_size: number[];
    stride?: number[]
    padding?: number[];
    count_include_pad?: boolean;
    divisor_override?: number;
    ceil_mode?: boolean;
}

export type BatchNorm2dLayerConfigInput = {
    name?: string;
    num_features: number;
    eps?: number;
    momentum?: number;
    affine?: boolean;
    track_running_status?: boolean;
}

export type BatchNorm1dLayerConfigInput = {
    name?: string;
    num_features: number;
    eps?: number;
    momentum?: number;
    affine?: boolean;
    track_running_status?: boolean;
}

export type FlattenLayerConfigInput = {
    name?: string;
    start_dim?: number;
    end_dim?: number;
}

export type DropoutLayerConfigInput = {
    name?: string;
    p?: number;
}

export type Dropout2dLayerConfigInput = {
    name?: string;
    p?: number;
}

export type ELULayerConfigInput = {
    name?: string;
    alpha?: number;
    inplace?: boolean;
}

export type ReLULayerConfigInput = {
    name?: string;
    inplace?: boolean;
}

export type LeakyReLULayerConfigInput = {
    name?: string;
    negative_slope?: number;
    inplace?: boolean;
}

export type SigmoidLayerConfigInput = {
    name?: string;
}

export type LogSigmoidLayerConfigInput = {
    name?: string;
}

export type TanhLayerConfigInput = {
    name?: string;
}

export type CatLayerConfigInput = {
    name?: string;
    dimension?: number;
}

export type LayerConfigInput = {
    type: string; 
    linear?: LinearLayerConfigInput;
    conv2d?: Conv2dLayerConfigInput;
    convtranspose2d?: ConvTranspose2dLayerConfigInput;
    conv1d?: Conv1dLayerConfigInput;
    maxpool2d?: MaxPool2dLayerConfigInput;
    maxpool1d?: MaxPool1dLayerConfigInput;
    avgpool2d?: AvgPool2dLayerConfigInput;
    avgpool1d?: AvgPool1dLayerConfigInput;
    batchnorm2d?: BatchNorm2dLayerConfigInput;
    batchnorm1d?: BatchNorm1dLayerConfigInput;
    flatten?: FlattenLayerConfigInput;
    dropout?: DropoutLayerConfigInput;
    dropout2d?: Dropout2dLayerConfigInput;
    elu?: ELULayerConfigInput;
    relu?: ReLULayerConfigInput;
    leakyrelu?: LeakyReLULayerConfigInput;
    sigmoid?: SigmoidLayerConfigInput;
    logsigmoid?: LogSigmoidLayerConfigInput;
    tanh?: TanhLayerConfigInput;
    cat?: CatLayerConfigInput;
};

// ------------------------------- Training Configuration ----------------------------------
export type OptimizerConfig = {
    lr: number;
    eps?: number;
    weight_decay?: number;
    betas?: number[];
    rho?: number;
    beta2_decay?: number;
    d?: number;
    lambd?: number;
    alpha?: number;
    t0?: number;
    momentum?: number;
    dampening?: number;
    nesterov?: boolean;
    etas?: number[]
    step_sizes?: number[];
    max_iter?: number;
    max_eval?: number;
    tolerance_grad?: number;
    tolerance_change?: number;
    history_size?: number;
};  

export type TrainConfig = {
    epochs: number;
    optimizer: string;
    optimizer_config: OptimizerConfig;
    loss_function: string;
};

export type OptimizerConfigInput = {
    lr: number;
    eps?: number;
    weight_decay?: number;
    betas?: number[];
    rho?: number;
    beta2_decay?: number;
    d?: number;
    lambd?: number;
    alpha?: number;
    t0?: number;
    momentum?: number;
    dampening?: number;
    nesterov?: boolean;
    etas?: number[]
    step_sizes?: number[];
    max_iter?: number;
    max_eval?: number;
    tolerance_grad?: number;
    tolerance_change?: number;
    history_size?: number;
};

export type TrainConfigInput = {
    epochs: number;
    optimizer: string;
    optimizer_config: OptimizerConfigInput;
    loss_function: string;
};

export type TrainStatus = {
    epoch: number;
    loss: number;
    accuracy: number;
    started: boolean;
    completed: boolean;
    timestamp?: string;
}

// ------------------------------- Dataset Configuration ----------------------------------
export interface DatasetConfig {
    name: string;
    batch_size?: number;
    split_length?: number[];
    shuffle?: boolean;
};

export interface MNISTDatasetConfig extends DatasetConfig {
    root: string;
    train?: boolean;
    download?: boolean;
    transform?: string[];
};

export interface CIFAR10DatasetConfig extends DatasetConfig {
    root: string;
    train?: boolean;
    download?: boolean;
    transform?: string[];
};

export interface CustomCSVDatasetConfig extends DatasetConfig {
    root: string;
    feature_columns: string[];
    label_columns: string[];
    is_regression_task: boolean;
}

export interface ImageFolderDatasetConfig extends DatasetConfig {
    root: string;
    transform?: string[];
    allow_empty?: boolean;
}

export type MNISTDatasetConfigInput = {
    root: string;
    train?: boolean;
    download?: boolean;
    transform?: string[];
};

export type CIFAR10DatasetConfigInput = {
    root: string;
    train?: boolean;
    download?: boolean;
    transform?: string[];
};

export type CustomCSVDatasetConfigInput = {
    root: string;
    feature_columns: string[];
    label_columns: string[];
    is_regression_task?: boolean;
}

export type ImageFolderDatasetConfigInput = {
    root: string;
    transform?: string[];
    allow_empty?: boolean;
}

export type DatasetConfigInput  = {
    name: string;
    batch_size?: number;
    split_length?: number[];
    shuffle?: boolean;
    mnist?: MNISTDatasetConfigInput
    cifar10?: CIFAR10DatasetConfigInput
    image_folder?: ImageFolderDatasetConfigInput
    custom_csv?: CustomCSVDatasetConfigInput
};

// ------------------------------ Module Graphs ------------------------------
export type GraphLayerDimensionResult = {
    out_dimension: number[];
    message?: string;
    required_in_dimension?: number[];
};

export type ModuleGraphDimensionStatus = {
    status?: GraphLayerDimensionResult[];
}

export type GraphLayerDimensionHandler = (
    layer_config: LayerConfig,
    in_dimension: number[]
    
) => GraphLayerDimensionResult;

export type ModuleAdjacencyList =  {
    source_id: string;
    target_ids: string[];
}

export type ModuleGraph = {
    layers: LayerConfig[];
    edges: ModuleAdjacencyList[];
    sorted: string[];
}

export type ModuleAdjacencyListInput =  {
    source_id: string;
    target_ids: string[];
}

export type ModuleGraphInput = {
    layers: LayerConfigInput[];
    edges: ModuleAdjacencyListInput[];
}

// ------------------------------- Model ----------------------------------
export type Model  = {
    id: string;
    name: string;
    module_graph?: ModuleGraph;
    layers_config: LayerConfig[];
    train_config: TrainConfig;
    dataset_config: DatasetConfig;
};

export type ModelDimensionResolveStatusStruct = {
    layer_id: string;
    message?: string;
    in_dimension: number[];
    out_dimension: number[];
    required_in_dimension?: number[];
}

export type ModelDimensionResolveStatus = {
    status?: ModelDimensionResolveStatusStruct[];
}

// createModel function args
export type CreateModelArgs = {
    name: string;
};

// appendLayer function args
export type AppendLayerArgs = {
    model_id: string;
    layer_config: LayerConfigInput;
};

// deleteLayer function args
export type DeleteLayerArgs = {
    model_id: string;
    layer_id: string;
};

// modifyLayer function args
export type ModifyLayerArgs = {
    model_id: string;
    layer_id: string;
    layer_config: LayerConfigInput;
};

// appendToModuleGraph function args
export type AppendToModuleGraphArgs = {
    layer_config: LayerConfigInput
}

// connectInModuleGraph function args
export type ConnectInModuleGraphArgs = {
    source_layer_id: string;
    target_layer_id: string;
}

// disconnectInModuleGraph function args
export type DisconnectInModuleGraphArgs = {
    source_layer_id: string;
    target_layer_id: string;
}

// deleteInModuleGraph function args
export type DeleteInModuleGraphArgs = {
    layer_id: string
}

// buildModuleGraph function args
export type BuildModuleGraphArgs = {
    module_graph: ModuleGraphInput;
}

// setTrainConfig function args
export type SetTrainConfigArgs = {
    train_config: TrainConfigInput;
};

// setDataset function args
export type SetDatasetArgs = {
    dataset_config: DatasetConfigInput;
};

enum ExportType {
    TorchTensor,
    ONNX
}

export type GraphQLTrainArgs = {
    export_to: ExportType
}

// train function args
export type TrainArgs = {
    args: GraphQLTrainArgs;
};

