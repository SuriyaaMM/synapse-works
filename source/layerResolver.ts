import { v4 as uuidv4 } from "uuid";
import { enqueueMessage } from "./redisClient.js";

import {
    Layer,
    LayerInput,
    LinearLayer, 
    Model,
    AppendLayerArgs
} from "./types.js"

type LayerHandlerMap = {
    [K in LayerInput['type']]: (config: LayerInput) => Layer;
};

// handles runtime validation of layer types
const layerHandler: LayerHandlerMap = {
    "linear": (layerInput: LayerInput) => {
        // destructure layer Input
        const { linear }  =  layerInput;
        // handle missing linear layer config
        if(!linear) throw new Error(`[synapse][graphql]: Linear layer config is missing`);

        // initialize new layer & its uuid
        let newLayer: LinearLayer;
        const layerId = uuidv4();

        newLayer = {
            id: layerId,
            type: "linear",
            name: linear.name || `linear_${layerId.substring(0, 4)}`,
            in_features: linear.in_features,
            out_features: linear.out_features,
            bias: linear.bias
        };

        return newLayer;
    }
}

export async function appendLayerResolver (models: Model[], args: AppendLayerArgs) {
    // find the model 
    const model = models.find(m => m.id === args.modelId);

    // handle model doesn't exist 
    if(!model){
        throw new Error(`[synapse][graphql]: Model with ID ${args.modelId} not found`)
    }
            
    // get the right layer object
    const handler = layerHandler[args.layerInput.type];
    // handle invalid layer
    if(!handler) throw new Error(`[synapse][graphql]: layer ${args.layerInput.type} is invalid`);
    // get the parsed layer 
    const newLayer = handler(args.layerInput);
    // push layer to model
    model.layers.push(newLayer)
    console.log(`[synapse][graphql]: Appended ${args.layerInput.type} layer (ID: ${newLayer.id}) to model ${model.name} (Model ID: ${model.id})`);
            
    // push message to redis
    console.log(`[synapse][graphql]: Appending to redis message Queue`)
    const message = {
        eventType: "LAYER_ADDED",
        modelId: model.id,
        layerData: model.layers.at(-1),
        timestamp: new Date().toISOString()
    };
    await enqueueMessage(message);

    // return model
    return model;
}
