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
    bias?: boolean
    padding_mode?: string
};

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
    bias?: boolean
    padding_mode?: string
}

export type LayerConfigInput = {
    type: string; 
    linear?: LinearLayerConfigInput;
    conv2d?: Conv2dLayerConfigInput;
};

// ------------------------------- Training Configuration ----------------------------------
export type OptimizerConfig = {
    lr: number;
};  

export type TrainConfig = {
    epochs: number;
    optimizer: string;
    optimizer_config: OptimizerConfig
    loss_function: string;
};

export type OptimizerConfigInput = {
    lr: number;
};

export type TrainConfigInput = {
    epochs: number;
    optimizer: string;
    optimizer_config: OptimizerConfigInput
    loss_function: string;
};

export type TrainStatus = {
    epoch: number;
    loss: number;
    accuracy: number;
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
};

export interface CIFAR10DatasetConfig extends DatasetConfig {
    root: string;
    train?: boolean;
    download?: boolean;
};

export type MNISTDatasetConfigInput = {
    root: string;
    train?: boolean;
    download?: boolean;
};

export type CIFAR10DatasetConfigInput = {
    root: string;
    train?: boolean;
    download?: boolean;
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

// createModel function args
export type CreateModelArgs = {
    name: string;
};

// appendLayer function args
export type AppendLayerArgs = {
    model_id: string;
    layer_config: LayerConfigInput;
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

