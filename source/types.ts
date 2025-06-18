// ------------------------------- Layers & Configuration ----------------------------------
export interface Layer {
    id: string;
    type: string;
    name?: string;
}

export interface LinearLayer extends Layer {
    type: 'linear';
    name?: string;
    in_features: number;
    out_features: number;
    bias?: boolean;
}

export type LinearLayerInput = {
    name?: string;
    in_features: number;
    out_features: number;
    bias?: boolean;
};

export type LayerInput = {
    type: string; 
    linear?: LinearLayerInput;
};

// ------------------------------- Training Configuration ----------------------------------
type OptimizerConfig = {
    lr: number;
}  
type TrainConfig = {
    epochs: number;
    batch_size: number;
    optimizer: string;
    optimizerConfig: OptimizerConfig
    loss_function: string;
}

// ------------------------------- Dataset Configuration ----------------------------------
export interface Dataset {
    name: string;
    split_length?: number[];
    shuffle?: boolean;
}

export interface MNISTDataset extends Dataset {
    root: string;
    train?: boolean;
    download?: boolean;
}

export type MNISTDatasetInput = {
    root: string;
    train?: boolean;
    download?: boolean;
}

export type DatasetInput  = {
    name: string;
    split_length?: number[];
    shuffle?: boolean;
    mnist?: MNISTDatasetInput
}

// ------------------------------- Model ----------------------------------
export type Model  = {
  id: string;
  name: string;
  layers: Layer[];
  trainConfig: TrainConfig;
  dataset: Dataset;
}

// createModel function args
export type CreateModelArgs = {
    name: string;
}

// appendLayer function args
export type AppendLayerArgs = {
    modelId: string;
    layerInput: LayerInput;
};

// setTrainConfig function args
export type SetTrainConfigArgs = {
    modelId: string;
    trainConfig: TrainConfig;
}

// setDataset function args
export type SetDatasetArgs = {
    modelId: string;
    datasetInput: DatasetInput;
}

// train function args
export type TrainArgs = {
    modelId: string;
}

