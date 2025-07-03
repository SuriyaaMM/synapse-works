import { LayerConfigInput } from "./layerTypes.js";
import { ModuleGraphInput } from "./graphTypes.js";
import { TrainConfigInput } from "./trainTypes.js";
import { DatasetConfigInput } from "./datasetTypes.js";

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

export enum ExportType {
    TorchTensor = 'TorchTensor',
    ONNX = 'ONNX',
}

export type GraphQLTrainArgs = {
    export_to: ExportType
}

// train function args
export type TrainArgs = {
    args: GraphQLTrainArgs;
};

