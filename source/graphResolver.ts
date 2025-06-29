import { layerHandler } from "./layerResolver.js";
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
    DisconnectInModuleGraphArgs
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