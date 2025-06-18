// ------------------------------- Layers & Configuration ----------------------------------
export interface LayerConfig {
    id: string;
    type: string;
    name?: string;
}

export interface LinearLayerConfig extends LayerConfig {
    in_features: number;
    out_features: number;
    bias?: boolean;
}

export type LinearLayerConfigInput = {
    name?: string;
    in_features: number;
    out_features: number;
    bias?: boolean;
};

export type LayerConfigInput = {
    type: string; 
    linear?: LinearLayerConfigInput;
};

// ------------------------------- Training Configuration ----------------------------------
type OptimizerConfig = {
    lr: number;
}  
type TrainConfig = {
    epochs: number;
    optimizer: string;
    optimizer_config: OptimizerConfig
    loss_function: string;
}
type OptimizerConfigInput = {
    lr: number;
}  
type TrainConfigInput = {
    epochs: number;
    optimizer: string;
    optimizer_config: OptimizerConfigInput
    loss_function: string;
}

// ------------------------------- Dataset Configuration ----------------------------------
export interface DatasetConfig {
    name: string;
    batch_size?: number;
    split_length?: number[];
    shuffle?: boolean;
}

export interface MNISTDatasetConfig extends DatasetConfig {
    root: string;
    train?: boolean;
    download?: boolean;
}

export type MNISTDatasetConfigInput = {
    root: string;
    train?: boolean;
    download?: boolean;
}

export type DatasetConfigInput  = {
    name: string;
    batch_size?: number;
    split_length?: number[];
    shuffle?: boolean;
    mnist?: MNISTDatasetConfigInput
}

// ------------------------------- Model ----------------------------------
export type Model  = {
    id: string;
    name: string;
    layers_config: LayerConfig[];
    train_config: TrainConfig;
    dataset_config: DatasetConfig;
}

// createModel function args
export type CreateModelArgs = {
    name: string;
}

// appendLayer function args
export type AppendLayerArgs = {
    model_id: string;
    layer_config: LayerConfigInput;
};

// setTrainConfig function args
export type SetTrainConfigArgs = {
    model_id: string;
    train_config: TrainConfigInput;
}

// setDataset function args
export type SetDatasetArgs = {
    model_id: string;
    dataset_config: DatasetConfigInput;
}

// train function args
export type TrainArgs = {
    model_id: string;
}

