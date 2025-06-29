import { v4 as uuidv4 } from "uuid";
import { enqueueMessage } from "./redisClient.js";

import {
    LayerConfig,
    LayerConfigInput,
    LinearLayerConfig, 
    Model,
    AppendLayerArgs,
    Conv2dLayerConfig,
    ConvTranspose2dLayerConfig,
    Conv1dLayerConfig,
    MaxPool2dLayerConfig,
    MaxPool1dLayerConfig,
    AvgPool2dLayerConfig,
    AvgPool1dLayerConfig,
    BatchNorm2dLayerConfig,
    BatchNorm1dLayerConfig,
    FlattenLayerConfig,
    Dropout2dLayerConfig,
    DropoutLayerConfig,
    ELULayerConfig,
    ReLULayerConfig,
    LeakyReLULayerConfig,
    SigmoidLayerConfig,
    LogSigmoidLayerConfig,
    TanhLayerConfig,
    CatLayerConfig,
    DeleteLayerArgs,
    ModifyLayerArgs,
} from "./types.js"

type LayerHandlerMap = {
    [K in LayerConfigInput['type']]: (config: LayerConfigInput) => LayerConfig;
};

// handles runtime validation of layer types
export const layerHandler: LayerHandlerMap = {
    // ---------- linear ----------
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
            bias: linear.bias,
        };

        return new_layer_config;
    },
    // ---------- conv2d ----------
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
    },
    // ---------- convtranspose2d ----------
    "convtranspose2d" : (layer_config: LayerConfigInput) => {
        // destructure layer input
        const { convtranspose2d } = layer_config;

        if(!convtranspose2d) throw new Error('[synapse][layerHandler]: ConvTranspose2d layer config is missing');
        else console.log(`[synapse][graphql]: parsing ${JSON.stringify(layer_config)}`)
        // initialize new layer & its uuid
        let new_layer_config: ConvTranspose2dLayerConfig;
        const layer_id = uuidv4();

        new_layer_config = {
            id: layer_id,
            type: "convtranspose2d",
            name: convtranspose2d.name || `convtranspose2d_${layer_id.substring(0, 4)}`,
            in_channels: convtranspose2d.in_channels,
            out_channels: convtranspose2d.out_channels,
            kernel_size: convtranspose2d.kernel_size,
            stride: convtranspose2d.stride,
            padding: convtranspose2d.padding,
            dilation: convtranspose2d.dilation,
            groups: convtranspose2d.groups,
            bias: convtranspose2d.bias,
            output_padding: convtranspose2d.output_padding
        };

        return new_layer_config;
    },
    // ---------- conv1d ----------
    "conv1d": (layer_config: LayerConfigInput) => {
        // destructure layer input
        const { conv1d } = layer_config;

        if(!conv1d) throw new Error('[synapse][layerHandler]: Conv1d layer config is missing');
        // initialize new layer & its uuid
        let new_layer_config: Conv1dLayerConfig;
        const layer_id = uuidv4();

        new_layer_config = {
            id: layer_id,
            type: "conv1d",
            name: conv1d.name || `conv1d_${layer_id.substring(0, 4)}`,
            in_channels: conv1d.in_channels,
            out_channels: conv1d.out_channels,
            kernel_size: conv1d.kernel_size,
            stride: conv1d.stride,
            padding: conv1d.padding,
            dilation: conv1d.dilation,
            groups: conv1d.groups,
            bias: conv1d.bias,
            padding_mode: conv1d.padding_mode
        };

        return new_layer_config;
    },
    // ---------- maxpool2d ----------
    "maxpool2d": (layer_config: LayerConfigInput) => {
        // destructure layer input
        const { maxpool2d } = layer_config;

        if(!maxpool2d) throw new Error('[synapse][layerHandler]: MaxPool2d layer config is missing');
        // initialize new layer & its uuid
        let new_layer_config: MaxPool2dLayerConfig;
        const layer_id = uuidv4();

        new_layer_config = {
            id: layer_id,
            type: "maxpool2d",
            name: maxpool2d.name || `maxpool2d_${layer_id.substring(0, 4)}`,
            kernel_size: maxpool2d.kernel_size,
            stride: maxpool2d.stride,
            padding: maxpool2d.padding,
            dilation: maxpool2d.dilation,
            return_indices: maxpool2d.return_indices,
            ceil_mode: maxpool2d.ceil_mode
        };

        return new_layer_config;
    },
    // ---------- maxpool1d ----------
    "maxpool1d": (layer_config: LayerConfigInput) => {
        // destructure layer input
        const { maxpool1d } = layer_config;

        if(!maxpool1d) throw new Error('[synapse][layerHandler]: MaxPool1d layer config is missing');
        // initialize new layer & its uuid
        let new_layer_config: MaxPool1dLayerConfig;
        const layer_id = uuidv4();

        new_layer_config = {
            id: layer_id,
            type: "maxpool1d",
            name: maxpool1d.name || `maxpool1d_${layer_id.substring(0, 4)}`,
            kernel_size: maxpool1d.kernel_size,
            stride: maxpool1d.stride,
            padding: maxpool1d.padding,
            dilation: maxpool1d.dilation,
            return_indices: maxpool1d.return_indices,
            ceil_mode: maxpool1d.ceil_mode
        };

        return new_layer_config;
    },
    // ---------- avgpool2d ----------
    "avgpool2d": (layer_config: LayerConfigInput) => {
        // destructure layer input
        const { avgpool2d } = layer_config;

        if(!avgpool2d) throw new Error('[synapse][layerHandler]: AvgPool2d layer config is missing');
        // initialize new layer & its uuid
        let new_layer_config: AvgPool2dLayerConfig;
        const layer_id = uuidv4();

        new_layer_config = {
            id: layer_id,
            type: "avgpool2d",
            name: avgpool2d.name || `avgpool2d_${layer_id.substring(0, 4)}`,
            kernel_size: avgpool2d.kernel_size,
            stride: avgpool2d.stride,
            padding: avgpool2d.padding,
            count_include_pad: avgpool2d.count_include_pad,
            divisor_override: avgpool2d.divisor_override,
            ceil_mode: avgpool2d.ceil_mode
        };

        return new_layer_config;
    },
    // ---------- avgpool1d ----------
    "avgpool1d": (layer_config: LayerConfigInput) => {
        // destructure layer input
        const { avgpool1d } = layer_config;

        if(!avgpool1d) throw new Error('[synapse][layerHandler]: AvgPool1d layer config is missing');
        // initialize new layer & its uuid
        let new_layer_config: AvgPool1dLayerConfig;
        const layer_id = uuidv4();

        new_layer_config = {
            id: layer_id,
            type: "avgpool1d",
            name: avgpool1d.name || `avgpool1d_${layer_id.substring(0, 4)}`,
            kernel_size: avgpool1d.kernel_size,
            stride: avgpool1d.stride,
            padding: avgpool1d.padding,
            count_include_pad: avgpool1d.count_include_pad,
            divisor_override: avgpool1d.divisor_override,
            ceil_mode: avgpool1d.ceil_mode
        };

        return new_layer_config;
    },
    // ---------- batchnorm2d ----------
    "batchnorm2d": (layer_config: LayerConfigInput) => {
        // destructure layer input
        const { batchnorm2d } = layer_config;

        if(!batchnorm2d) throw new Error('[synapse][layerHandler]: BatchNorm2d layer config is missing');
        // initialize new layer & its uuid
        let new_layer_config: BatchNorm2dLayerConfig;
        const layer_id = uuidv4();

        new_layer_config = {
            id: layer_id,
            type: "batchnorm2d",
            name: batchnorm2d.name || `batchnorm2d_${layer_id.substring(0, 4)}`,
            num_features: batchnorm2d.num_features,
            eps: batchnorm2d.eps,
            momentum: batchnorm2d.momentum,
            affine: batchnorm2d.affine,
            track_running_status: batchnorm2d.track_running_status
        };

        return new_layer_config;
    },
    // ---------- batchnorm1d ----------
    "batchnorm1d": (layer_config: LayerConfigInput) => {
        // destructure layer input
        const { batchnorm1d } = layer_config;

        if(!batchnorm1d) throw new Error('[synapse][layerHandler]: BatchNorm1d layer config is missing');
        // initialize new layer & its uuid
        let new_layer_config: BatchNorm1dLayerConfig;
        const layer_id = uuidv4();

        new_layer_config = {
            id: layer_id,
            type: "batchnorm1d",
            name: batchnorm1d.name || `batchnorm1d_${layer_id.substring(0, 4)}`,
            num_features: batchnorm1d.num_features,
            eps: batchnorm1d.eps,
            momentum: batchnorm1d.momentum,
            affine: batchnorm1d.affine,
            track_running_status: batchnorm1d.track_running_status
        };

        return new_layer_config;
    },
    // ---------- flatten ----------
    "flatten": (layer_config: LayerConfigInput) => {
        // destructure layer input
        const { flatten } = layer_config;

        if(!flatten) throw new Error('[synapse][layerHandler]: Flatten layer config is missing');
        // initialize new layer & its uuid
        let new_layer_config: FlattenLayerConfig;
        const layer_id = uuidv4();

        new_layer_config = {
            id: layer_id,
            type: "flatten",
            name: flatten.name || `flatten_${layer_id.substring(0, 4)}`,
            start_dim: flatten.start_dim,
            end_dim: flatten.end_dim,
        };

        return new_layer_config;
    },
    // ---------- dropout ----------
    "dropout": (layer_config: LayerConfigInput) => {
        // destructure layer input
        const { dropout } = layer_config;

        if(!dropout) throw new Error('[synapse][layerHandler]: Dropout layer config is missing');
        // initialize new layer & its uuid
        let new_layer_config: DropoutLayerConfig;
        const layer_id = uuidv4();

        new_layer_config = {
            id: layer_id,
            type: "dropout",
            name: dropout.name || `dropout_${layer_id.substring(0, 4)}`,
            p: dropout.p
        };

        return new_layer_config;
    },
    // ---------- dropout ----------
    "dropout2d": (layer_config: LayerConfigInput) => {
        // destructure layer input
        const { dropout2d } = layer_config;

        if(!dropout2d) throw new Error('[synapse][layerHandler]: Dropout2d layer config is missing');
        // initialize new layer & its uuid
        let new_layer_config: Dropout2dLayerConfig;
        const layer_id = uuidv4();

        new_layer_config = {
            id: layer_id,
            type: "dropout2d",
            name: dropout2d.name || `dropout_${layer_id.substring(0, 4)}`,
            p: dropout2d.p
        };

        return new_layer_config;
    },
    // ---------- elu ----------
    "elu": (layer_config: LayerConfigInput) => {
        // destructure layer input
        const { elu } = layer_config;

        if(!elu) throw new Error('[synapse][layerHandler]: ELU layer config is missing');
        // initialize new layer & its uuid
        let new_layer_config: ELULayerConfig;
        const layer_id = uuidv4();

        new_layer_config = {
            id: layer_id,
            type: "dropout",
            name: elu.name || `dropout_${layer_id.substring(0, 4)}`,
            alpha: elu.alpha,
            inplace: elu.inplace
        };

        return new_layer_config;
    },
    // ---------- relu ----------
    "relu": (layer_config: LayerConfigInput) => {
        // destructure layer input
        const { relu } = layer_config;

        if(!relu) throw new Error('[synapse][layerHandler]: ReLU layer config is missing');
        // initialize new layer & its uuid
        let new_layer_config: ReLULayerConfig;
        const layer_id = uuidv4();

        new_layer_config = {
            id: layer_id,
            type: "relu",
            name: relu.name || `relu_${layer_id.substring(0, 4)}`,
            inplace: relu.inplace
        };

        return new_layer_config;
    },
    // ---------- leakurelu ----------
    "leakyrelu": (layer_config: LayerConfigInput) => {
        // destructure layer input
        const { leakyrelu } = layer_config;

        if(!leakyrelu) throw new Error('[synapse][layerHandler]: LeakyReLU layer config is missing');
        // initialize new layer & its uuid
        let new_layer_config: LeakyReLULayerConfig;
        const layer_id = uuidv4();

        new_layer_config = {
            id: layer_id,
            type: "leakyrelu",
            name: leakyrelu.name || `leakyrelu_${layer_id.substring(0, 4)}`,
            negative_slope: leakyrelu.negative_slope,
            inplace: leakyrelu.inplace
        };

        return new_layer_config;
    },
    // ---------- sigmoid ----------
    "sigmoid": (layer_config: LayerConfigInput) => {
        // destructure layer input
        const { sigmoid } = layer_config;

        if(!sigmoid) throw new Error('[synapse][layerHandler]: Sigmoid layer config is missing');
        // initialize new layer & its uuid
        let new_layer_config: SigmoidLayerConfig;
        const layer_id = uuidv4();

        new_layer_config = {
            id: layer_id,
            type: "sigmoid",
            name: sigmoid.name || `sigmoid_${layer_id.substring(0, 4)}`,
        };

        return new_layer_config;
    },
    // ---------- logsigmoid ----------
    "logsigmoid": (layer_config: LayerConfigInput) => {
        // destructure layer input
        const { logsigmoid } = layer_config;

        if(!logsigmoid) throw new Error('[synapse][layerHandler]: LogSigmoid layer config is missing');
        // initialize new layer & its uuid
        let new_layer_config: LogSigmoidLayerConfig;
        const layer_id = uuidv4();

        new_layer_config = {
            id: layer_id,
            type: "logsigmoid",
            name: logsigmoid.name || `logsigmoid_${layer_id.substring(0, 4)}`,
        };

        return new_layer_config;
    },
    // ---------- tanh ----------
    "tanh": (layer_config: LayerConfigInput) => {
        // destructure layer input
        const { tanh } = layer_config;

        if(!tanh) throw new Error('[synapse][layerHandler]: Tanh layer config is missing');
        // initialize new layer & its uuid
        let new_layer_config: TanhLayerConfig;
        const layer_id = uuidv4();

        new_layer_config = {
            id: layer_id,
            type: "tanh",
            name: tanh.name || `tanh_${layer_id.substring(0, 4)}`,
        };

        return new_layer_config;
    },
    // ---------- cat ----------
    "cat": (layer_config: LayerConfigInput) => {
        // destructure layer input
        const { cat } = layer_config;

        if(!cat) throw new Error('[synapse]: Cat layer config is missing');
        // initialize new layer & its uuid
        let new_layer_config: CatLayerConfig;
        const layer_id = uuidv4();

        new_layer_config = {
            id: layer_id,
            type: "cat",
            name: cat.name || `cat_${layer_id.substring(0, 4)}`,
            dimension: cat.dimension
        };

        return new_layer_config;
    },
}

export async function appendLayerResolver(model: Model, args: AppendLayerArgs) {

    // handle model doesn't exist 
    if(!model){
        throw new Error(`[synapse][graphql]: Model doesn't exist yet, create it first`)
    }
            
    // get the right layer object
    const handler = layerHandler[args.layer_config.type];
    // handle invalid layer
    if(!handler) throw new Error(`[synapse][graphql]: layer ${args.layer_config.type} is invalid`);
    // get the parsed layer 
    const new_layer = handler(args.layer_config);
    // push layer to model
    model.layers_config.push(new_layer);
    console.log(`[synapse][graphql]: Appended ${args.layer_config.type} layer (ID: ${new_layer.id}) to model ${model.name} (Model ID: ${model.id})`);
            
    // push message to redis
    console.log(`[synapse][graphql]: Appending to redis message Queue`);
    const message = {
        event_type: "LAYER_ADDED",
        model_id: model.id,
        layer_config: new_layer,
        timestamp: new Date().toISOString()
    };
    await enqueueMessage(message);

    // return model
    return model;
}

export async function deleteLayerResolver(model: Model, args: DeleteLayerArgs) {

    // handle model doesn't exist 
    if(!model){
        throw new Error(`[synapse][graphql]: Model doesn't exist yet, create it first`);
    }

    model.layers_config = model.layers_config.filter(layer_config => layer_config.id !== args.layer_id);
            
    // push message to redis
    console.log(`[synapse][graphql]: Appending to redis message Queue`);
    const message = {
        event_type: "LAYER_DELETED",
        model_id: args.model_id,
        layer_id: args.layer_id,
        timestamp: new Date().toISOString()
    };
    await enqueueMessage(message);

    // return model
    return model;
}

export async function modifyLayerResolver(model: Model, args: ModifyLayerArgs) {

    // handle model doesn't exist 
    if(!model){
        throw new Error(`[synapse][graphql]: Model doesn't exist yet, create it first`);
    }
    
    // get the right layer object
    const handler = layerHandler[args.layer_config.type];
    // handle invalid layer
    if(!handler) throw new Error(`[synapse][graphql]: layer ${args.layer_config.type} is invalid`);
    // get the parsed layer 
    const new_layer = handler(args.layer_config);
    new_layer.id = args.layer_id;

    const index_of_layer_to_change = model.layers_config.findIndex(layer => layer.id === args.layer_id);

    if(index_of_layer_to_change === -1) throw new Error(`[synapse][graphql]: Layer with ID ${args.layer_id} not found`);
    
    model.layers_config[index_of_layer_to_change] = new_layer;

    // push message to redis
    console.log(`[synapse][graphql]: Appending to LAYER_MODIFIED redis message Queue`);
    const message = {
        event_type: "LAYER_MODIFIED",
        model_id: args.model_id,
        layer_id: args.layer_id,
        layer_config: new_layer,
        timestamp: new Date().toISOString()
    };
    await enqueueMessage(message);

    // return model
    return model;
}

