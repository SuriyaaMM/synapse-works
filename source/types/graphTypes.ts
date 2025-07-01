import { LayerConfig, LayerConfigInput } from "./layerTypes.js";


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

export type ModuleGraphValidateDimensionStatusStruct = {
    layer_id: string;
    in_dimension: number[];
    out_dimension: number[];
    message?: string;
    required_in_dimension?: number[];
}

export type ModuleGraphValidateDimensionStatus = {
    status: ModuleGraphValidateDimensionStatusStruct[];
}

export type ModuleGraphLayerDimensionResult = {
    out_dimension: number[];
    // optional, returns these only on occurence of error
    message?: string;
    required_in_dimension?: number[];
};

export type GraphLayerDimensionHandler = (
    layer_config: LayerConfig,
    in_dimension: number[]
    
) => ModuleGraphLayerDimensionResult;
