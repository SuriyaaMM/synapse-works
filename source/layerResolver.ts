import { v4 as uuidv4 } from "uuid";
import { enqueueMessage } from "./redisClient.js";

import {
    LayerConfig,
    LayerConfigInput,
    LinearLayerConfig, 
    Model,
    AppendLayerArgs
} from "./types.js"

type LayerHandlerMap = {
    [K in LayerConfigInput['type']]: (config: LayerConfigInput) => LayerConfig;
};

// handles runtime validation of layer types
const layerHandler: LayerHandlerMap = {
    "linear": (layer_config: LayerConfigInput) => {
        // destructure layer Input
        const { linear }  =  layer_config;
        // handle missing linear layer config
        if(!linear) throw new Error(`[synapse][graphql]: Linear layer config is missing`);

        // initialize new layer & its uuid
        let new_layer_config: LinearLayerConfig;
        const layer_id = uuidv4();

        new_layer_config = {
            id: layer_id,
            type: "linear",
            name: linear.name || `linear_${layer_id.substring(0, 4)}`,
            in_features: linear.in_features,
            out_features: linear.out_features,
            bias: linear.bias
        };

        return new_layer_config;
    }
}

export async function appendLayerResolver (models: Model[], args: AppendLayerArgs) {
    // find the model 
    const model = models.find(m => m.id === args.model_id);

    // handle model doesn't exist 
    if(!model){
        throw new Error(`[synapse][graphql]: Model with ID ${args.model_id} not found`)
    }
            
    // get the right layer object
    const handler = layerHandler[args.layer_config.type];
    // handle invalid layer
    if(!handler) throw new Error(`[synapse][graphql]: layer ${args.layer_config.type} is invalid`);
    // get the parsed layer 
    const newLayer = handler(args.layer_config);
    // push layer to model
    model.layers_config.push(newLayer)
    console.log(`[synapse][graphql]: Appended ${args.layer_config.type} layer (ID: ${newLayer.id}) to model ${model.name} (Model ID: ${model.id})`);
            
    // push message to redis
    console.log(`[synapse][graphql]: Appending to redis message Queue`)
    const message = {
        event_type: "LAYER_ADDED",
        model_id: model.id,
        layer_config: model.layers_config.at(-1),
        timestamp: new Date().toISOString()
    };
    await enqueueMessage(message);

    // return model
    return model;
}
