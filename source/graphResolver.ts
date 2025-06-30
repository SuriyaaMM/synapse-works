import { layerDimensionHandler, layerHandler } from "./layerResolver.js";
import { enqueueMessage } from "./redisClient.js";
import { setModuleGraph } from "./resolvers.js";
import { 
    Model,
    LayerConfig,
    ModuleGraph,
    BuildModuleGraphArgs,
    AppendToModuleGraphArgs,
    ConnectInModuleGraphArgs,
    ModuleAdjacencyList,
    DeleteInModuleGraphArgs,
    DisconnectInModuleGraphArgs,
    ModelDimensionResolveStatus,
    GraphLayerDimensionResult,
    ModuleGraphDimensionStatus,
} from "./types.js";

// initialize an empty module graph
const module_graph: ModuleGraph = { layers: [], edges: [], sorted: []}

export async function appendToModuleGraphResolver(args: AppendToModuleGraphArgs){
    
    // get the handler for this layer config
    const handler = layerHandler[args.layer_config.type];
    // validate handler
    if(!handler) throw new Error(`[synapse]: Layer ${args.layer_config.type} is invalid`)
    // get the new layer config
    const new_layer_config = handler(args.layer_config);
    // add to graph    
    module_graph.layers.push(new_layer_config);

    return module_graph;

    // overall O(1) at worst case
}

export async function connectInModuleGraphResolver(args: ConnectInModuleGraphArgs){
    // check for whether these indices exist as layer first
    const source_layer_index = module_graph.layers.findIndex(layer => layer.id === args.source_layer_id);
    const target_layer_index = module_graph.layers.findIndex(layer => layer.id === args.target_layer_id);
    // throw error if not
    if(source_layer_index === -1 || target_layer_index === -1){
        throw new Error(`[synapse]: 
            either of these (${args.source_layer_id}, ${args.target_layer_id}) doesn't exist in the graph already`);
    }
    // check whether already this source_id has an element in adj list
    const source_layer_index_in_adj_list = module_graph.edges.findIndex(
        module_adj_list_element => module_adj_list_element.source_id === args.source_layer_id);
    
    if(source_layer_index_in_adj_list === -1){
        // new element for appending
        const new_adj_list_element: ModuleAdjacencyList = {source_id: args.source_layer_id, target_ids: [args.target_layer_id]};
        module_graph.edges.push(new_adj_list_element);
    }
    else{
        const target_layer_index_in_adj_list = module_graph.edges[source_layer_index_in_adj_list].target_ids.findIndex(
                id => id === args.target_layer_id);
        // if this target_id is not already in the list, add it
        if(target_layer_index_in_adj_list === -1){
            
            module_graph.edges[source_layer_index_in_adj_list].target_ids.push(args.target_layer_id);
        }
        else {
            console.warn(`${args.target_layer_id} is in the list for connections from ${args.source_layer_id}`);
        }
        
    }

    return module_graph;

    // overall O(3n) at worst case
}

export async function disconnectInModuleGraphResolver(args: DisconnectInModuleGraphArgs){
    // check for whether these indices exist as layer first
    const source_layer_index = module_graph.layers.findIndex(layer => layer.id === args.source_layer_id);
    const target_layer_index = module_graph.layers.findIndex(layer => layer.id === args.target_layer_id);
    // throw error if not
    if(source_layer_index === -1 || target_layer_index === -1){
        throw new Error(`[synapse]: 
            either of these (${args.source_layer_id}, ${args.target_layer_id}) doesn't exist in the graph already`);
    }
    // check whether already this source_id has an element in adj list
    const source_layer_index_in_adj_list = module_graph.edges.findIndex(
        module_adj_list_element => module_adj_list_element.source_id === args.source_layer_id);
    if(source_layer_index_in_adj_list === -1){
        console.warn(`[synapse]: ${args.source_layer_id} is not an connection in the graph`);
    }
    else {
        const target_layer_index_in_adj_list = module_graph.edges[source_layer_index_in_adj_list].target_ids.findIndex(
            id => id === args.target_layer_id);
        
        // if this is not there then just warn.
        if(target_layer_index_in_adj_list === -1){
            console.warn(`[synapse]: ${args.target_layer_id} is not in connection of ${args.source_layer_id}`)
        }
        else{
            // remove that edge
            module_graph.edges[source_layer_index_in_adj_list].target_ids.splice(target_layer_index_in_adj_list, 1);
            // remove any adjacency list of length 0
            if (module_graph.edges[source_layer_index_in_adj_list].target_ids.length === 0) {
                module_graph.edges.splice(source_layer_index_in_adj_list, 1);
            }
        }

    }

    return module_graph;
}

export async function deleteInModuleGraphResolver(args: DeleteInModuleGraphArgs){
    // check for whether this layer exists first
    const source_layer_index = module_graph.layers.findIndex(layer => layer.id === args.layer_id);
    // if this layer doesn't exist, put a warning 
    if(source_layer_index === -1){
        console.warn(`[synapse]: ${args.layer_id} doesn't exist in the graph`);
    }
    else {
        // remove the layers
        module_graph.layers = module_graph.layers.filter(layer => layer.id !== args.layer_id);
        // remove the edges containing this layer
        module_graph.edges = module_graph.edges.filter(edge => edge.source_id !== args.layer_id);
        // remove target id's containg this layer
        for (let i = 0; i < module_graph.edges.length; i++) {
            module_graph.edges[i].target_ids = module_graph.edges[i].target_ids.filter(target_id => target_id !== args.layer_id);
        }
        // remove any adjacency list of length 0 (cleanup)
        module_graph.edges = module_graph.edges.filter(edge => edge.target_ids.length > 0);
    }

    return module_graph;

    // overall O(n) at worst case
}

export async function buildModuleGraphResolver(model: Model){
    // handle model doesn't exist 
    if(!model){
        throw new Error(`[synapse][graphql]: Model doesn't exist yet, create it first`);
    }

    // construct layer hashmap for O(1) lookup
    // hashmap of <layer_id:layer_config>
    const layer_map: Map<string, LayerConfig> = new Map<string, LayerConfig>();
    // required for topological sort
    const indegree_map: Map<string, number> = new Map<string, number>();
    // construct layer_map
    for(const layer of module_graph.layers){
        layer_map.set(layer.id, layer);
        // initialize indegree map for this element
        indegree_map.set(layer.id, 0);
    }
    // calculate indegree
    for(const edge of module_graph.edges){
        for(const neighbour of edge.target_ids){
            indegree_map.set(neighbour, (indegree_map.get(neighbour) || 0) + 1);
        }
    }

    // collect notes with zero degrees
    let q: string[] = [];

    for (const [layer_id, degree] of indegree_map.entries()) {
        if (degree === 0) {
            q.push(layer_id);
        }
    }

    let sorted: string[] = [];

    while (q.length > 0) {
        const current = q.shift()!;
        sorted.push(current);
        
        for (const edge of module_graph.edges) {
            if (edge.source_id === current) {
                for (const neighbour of edge.target_ids) {
                    indegree_map.set(neighbour, indegree_map.get(neighbour)! - 1);
                    if (indegree_map.get(neighbour) === 0) {
                        q.push(neighbour);
                    }
                }
            }
        }
    }

    // cycle detected
    if (sorted.length !== module_graph.layers.length) {
        throw new Error(`[synapse][graphql]: Cycle detected in the model graph`);
    }

    model.module_graph = {
        layers: Array.from(layer_map.values()),
        edges: module_graph.edges,
        sorted: sorted
    }

    // push message to redis
    console.log(`[synapse][graphql]: Appending to LAYER_MODIFIED redis message Queue`);
    const message = {
        event_type: "CONSTRUCT_MODULE_GRAPH",
        module_graph: model.module_graph,
        timestamp: new Date().toISOString()
    };
    await enqueueMessage(message);

    return model;
}

export async function validateModuleGraphResolver(model: Model, initial_in_dimension: number[]): Promise<ModelDimensionResolveStatus> {
    console.log("[synapse]: Validating graph began!")
    const resolve_status: ModelDimensionResolveStatus = { status: [] };

    if (!model.module_graph) {
        throw new Error(`[synapse]: Module Graph must be provided for validation!`);
    }

    const layer_id_to_config_map = new Map<string, LayerConfig>();
    for (const layer of model.module_graph.layers) {
        layer_id_to_config_map.set(layer.id, layer);
    }

    const resolved_output_dimensions = new Map<string, number[]>();

    // iterate through topologically sorted order
    for (let i = 0; i < model.module_graph.sorted.length; i++) {

        const layer_id = model.module_graph.sorted[i];
        const layer_config = layer_id_to_config_map.get(layer_id);

        let current_in_dimension: number[];

        // for the first layer in the topological sort, use the initial input dimension
        if (i === 0) {
            current_in_dimension = initial_in_dimension;
        } else {
            // for subsequent layers, find its predecessor in the sorted list
            // and use its resolved output dimension as the input.
            const previous_layer_id = model.module_graph.sorted[i - 1];
            const prev_out_dim = resolved_output_dimensions.get(previous_layer_id);

            console.log(`[synapse]: Processing previous layer_id: ${previous_layer_id}`)

            if (!prev_out_dim) {
                // This should ideally not happen if the topological sort is correct
                // and all previous layers were successfully processed.
                resolve_status.status?.push({
                    layer_id: layer_id,
                    message: `Could not determine input dimension for layer ${layer_id}: predecessor ${previous_layer_id} output not resolved.`,
                    in_dimension: [],
                    out_dimension: []
                });
                return resolve_status;
            }
            current_in_dimension = prev_out_dim;
        }

        // get the appropriate dimension handler for the layer type
        const handler = layerDimensionHandler[layer_config.type];
        if (!handler) {
            resolve_status.status?.push({
                layer_id: layer_id,
                message: `No dimension handler found for layer type: ${layer_config.type}`,
                in_dimension: current_in_dimension,
                out_dimension: []
            });
            return resolve_status; // Stop on critical error
        }

        // calculate the output dimension for the current layer
        const result = handler(layer_config, current_in_dimension);

        // add the status for the current layer
        resolve_status.status?.push({
            layer_id: layer_id,
            message: result.message,
            in_dimension: current_in_dimension,
            out_dimension: result.out_dimension,
            required_in_dimension: result.required_in_dimension
        });

        // if there's an error message, stop validation and return the current status
        if (result.message) {
            return resolve_status;
        }

        // store the resolved output dimension for this layer
        resolved_output_dimensions.set(layer_id, result.out_dimension);
    }

    return resolve_status;
}