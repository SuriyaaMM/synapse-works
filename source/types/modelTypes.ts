import { ModuleGraph } from "./graphTypes.js";
import { LayerConfig } from "./layerTypes.js";
import { TrainConfig } from "./trainTypes.js";
import { DatasetConfig } from "./datasetTypes.js";

export type Model  = {
    id: string;
    name: string;
    module_graph?: ModuleGraph;
    layers_config?: LayerConfig[];
    train_config?: TrainConfig;
    dataset_config?: DatasetConfig;
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