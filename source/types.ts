// ------------------------------- Layers & Configuration ----------------------------------
export interface LayerConfig {
    id: string;
    type: string;
    name?: string;
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


export type LayerConfigInput = {
    type: string; 
    linear?: LinearLayerConfigInput;
    conv2d?: Conv2dLayerConfigInput;
    conv1d?: Conv1dLayerConfigInput;
    maxpool2d?: MaxPool2dLayerConfigInput;
    maxpool1d?: MaxPool1dLayerConfigInput;
    avgpool2d?: AvgPool2dLayerConfigInput;
    avgpool1d?: AvgPool1dLayerConfigInput;
    batchnorm2d?: BatchNorm2dLayerConfigInput;
    batchnorm1d?: BatchNorm1dLayerConfigInput;
    flatten?: FlattenLayerConfigInput;
    dropout?: DropoutLayerConfigInput;
    elu?: ELULayerConfigInput;
    relu?: ReLULayerConfigInput;
    leakyrelu?: LeakyReLULayerConfigInput;
    sigmoid?: SigmoidLayerConfigInput;
    logsigmoid?: LogSigmoidLayerConfigInput;
    tanh?: TanhLayerConfigInput;
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

export type DatasetConfigInput  = {
    name: string;
    batch_size?: number;
    split_length?: number[];
    shuffle?: boolean;
    mnist?: MNISTDatasetConfigInput
    cifar10?: CIFAR10DatasetConfigInput
};

// ------------------------------- Model ----------------------------------
export type Model  = {
    id: string;
    name: string;
    layers_config: LayerConfig[];
    train_config: TrainConfig;
    dataset_config: DatasetConfig;
};

export type ModelDimensionResolveStatusStruct = {
    layer_id: string;
    message: string;
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

// setTrainConfig function args
export type SetTrainConfigArgs = {
    model_id: string;
    train_config: TrainConfigInput;
};

// setDataset function args
export type SetDatasetArgs = {
    model_id: string;
    dataset_config: DatasetConfigInput;
};

// train function args
export type TrainArgs = {
    model_id: string;
};

