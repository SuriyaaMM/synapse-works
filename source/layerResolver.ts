import { v4 as uuidv4 } from "uuid";
import { enqueueMessage } from "./redisClient.js";

import {
    LayerConfig,
    LayerConfigInput,
    LinearLayerConfig, 
    Model,
    AppendLayerArgs,
    Conv2dLayerConfig
} from "./types.js"

type LayerHandlerMap = {
    [K in LayerConfigInput['type']]: (config: LayerConfigInput) => LayerConfig;
};

// handles runtime validation of layer types
const layerHandler: LayerHandlerMap = {
    "linear": (layer_config: LayerConfigInput) => {
        // destructure layer input
        const { linear }  =  layer_config;
        // handle missing linear layer config
        if(!linear) throw new Error(`[synapse][layerHandler]: Linear layer config is missing`);

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
    },

    "conv2d": (layer_config: LayerConfigInput) => {
        // destructure layer input
        const { conv2d } = layer_config;

        if(!conv2d) throw new Error('[synapse][layerHandler]: Conv2d layer config is missing');
        // initialize new layer & its uuid
        let new_layer_config: Conv2dLayerConfig;
        const layer_id = uuidv4();

        new_layer_config = {
            id: layer_id,
            type: "conv2d",
            name: conv2d.name || `conv2d_${layer_id.substring(0, 4)}`,
            in_channels: conv2d.in_channels,
            out_channels: conv2d.out_channels,
            kernel_size: conv2d.kernel_size,
            stride: conv2d.stride,
            padding: conv2d.padding,
            dilation: conv2d.dilation,
            groups: conv2d.groups,
            bias: conv2d.bias,
            padding_mode: conv2d.padding_mode
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
    const new_layer = handler(args.layer_config);
    // push layer to model
    model.layers_config.push(new_layer)
    console.log(`[synapse][graphql]: Appended ${args.layer_config.type} layer (ID: ${new_layer.id}) to model ${model.name} (Model ID: ${model.id})`);
            
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
