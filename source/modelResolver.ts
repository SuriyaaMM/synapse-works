import { v4 as uuidv4 } from "uuid";
import { enqueueMessage } from "./redisClient.js";
import { 
    Model,
    CreateModelArgs, 
    ModelDimensionResolveStatus,
    LinearLayerConfig,
    Conv2dLayerConfig,
    Conv1dLayerConfig,
    MaxPool2dLayerConfig,
    MaxPool1dLayerConfig,
    AvgPool2dLayerConfig,
    AvgPool1dLayerConfig,
    FlattenLayerConfig,
    ModelDimensionResolveStatusStruct
} from "./types";
import { setModel } from "./resolvers.js";

export async function createModelResolver(args: CreateModelArgs){
    // create a new model
    const new_model: Model = {
        id: uuidv4(), // generate unique uuid
        name: args.name, 
        layers_config: [], // initialize model with no layers to begin with
        train_config: {
            epochs: 50,  
            optimizer: "adam",
            optimizer_config: {
                lr: 3e-4
            }, 
            loss_function: "ce"
            }, // default train config
            dataset_config: {
                name: "mnist",
                split_length: [0.8, 0.2]
            }
    }

    setModel(new_model);
    console.log(`[synapse][graphql]: Created model: ${new_model.name} (ID: ${new_model.id})`);
    console.log(`[synapse][graphql]: Appending to redis message Queue`)
    // push message to redis
    const message = {
        event_type: "MODEL_CREATED",
        model_id: new_model.id,
        name: new_model.name,
        timestamp: new Date().toISOString()
    };
    
    await enqueueMessage(message);
    return new_model;
}

function convOutputDim(
    input: number,
    padding: number,
    dilation: number,
    kernel: number,
    stride: number
): number {
    return Math.floor((input + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1);
}

export async function validateModelResolver(
    model: Model, 
    in_dimension: number[]): Promise<ModelDimensionResolveStatus> {

    let return_object: ModelDimensionResolveStatus = { status: [] };

    if (!model) throw new Error('[synapse][graphql]: model does not exist');

    for (const layer_config of model.layers_config) {
        // handle for different layers
        if (layer_config.type === "linear") {
            const linear_layer_config = layer_config as LinearLayerConfig;
            const required_in_dimension = [linear_layer_config.in_features];
            const out_dimension = [linear_layer_config.out_features];

            let message: ModelDimensionResolveStatusStruct = {
                layer_id: linear_layer_config.id,
                in_dimension: in_dimension,
                out_dimension: out_dimension
            }
            // check if dimensions match
            if (JSON.stringify(required_in_dimension) !== JSON.stringify(in_dimension)) {
                message.message = `invalid configuration for linear layer, expected ${required_in_dimension} but received ${in_dimension}`;
                message.required_in_dimension = required_in_dimension;
                return_object.status.push(message);
                return return_object;
            }
            // push the message
            return_object.status.push(message);
            // for next layer
            in_dimension = out_dimension;
        } else if (layer_config.type === "flatten") {
            const flatten_layer_config = layer_config as FlattenLayerConfig;
            const out_dimension: number[] = [];
            let end = flatten_layer_config.end_dim ?? in_dimension.length;
            if(end == -1){
                end = in_dimension.length;
            }
            let begin = flatten_layer_config.start_dim ?? 0;
            // this is there to ensure that torch requires batch size and our's doesn't have
            // a batch size in calculation
            if(begin == 1){
                begin = 0;
            }
            // accumulator
            let flattened_dim = 1;

            for (let i = 0; i < end; i++) {
                if (i < begin) {
                    out_dimension.push(in_dimension[i]);
                } else {
                    flattened_dim *= in_dimension[i];
                }
            }
            out_dimension.push(flattened_dim);

            let message: ModelDimensionResolveStatusStruct = {
                layer_id: flatten_layer_config.id,
                in_dimension: in_dimension,
                out_dimension: out_dimension
            }

            return_object.status.push(message);
            in_dimension = out_dimension;
           
        } else if (layer_config.type === "conv2d") {
            const cfg = layer_config as Conv2dLayerConfig;

            let message: ModelDimensionResolveStatusStruct = {
                layer_id: cfg.id,
                in_dimension: in_dimension,
                out_dimension: []
            }

            if (in_dimension.length !== 3) {
                message.message = `conv2d requires 3d tensor, but received ${in_dimension}`;
                message.required_in_dimension = [0, 0, 0];
                return_object.status.push(message);
                return return_object;
            }

            if (in_dimension[0] !== cfg.in_channels) {
                message.message = `invalid configuration for conv2d, expected input_channels: ${cfg.in_channels}, received: ${in_dimension[0]}`
                message.required_in_dimension = [cfg.in_channels, 0, 0];
                return_object.status.push(message);
                return return_object;
            }

            const padding = cfg.padding ?? [0, 0];
            const dilation = cfg.dilation ?? [1, 1];
            const stride = cfg.stride ?? [1, 1];
            const kernel = cfg.kernel_size;

            const h_out = convOutputDim(in_dimension[1], padding[0], dilation[0], kernel[0], stride[0]);
            const w_out = convOutputDim(in_dimension[2], padding[1], dilation[1], kernel[1], stride[1]);
            const out_dimension = [cfg.out_channels, h_out, w_out];

            message.out_dimension = out_dimension;
            return_object.status.push(message);
            in_dimension = out_dimension;

        } else if (layer_config.type === "conv1d") {
            const cfg = layer_config as Conv1dLayerConfig;

            let message: ModelDimensionResolveStatusStruct = {
                layer_id: cfg.id,
                in_dimension: in_dimension,
                out_dimension: []
            }

            if (in_dimension.length !== 2) {
                message.message = `conv1d requires 2d tensor, but received ${in_dimension}`;
                message.required_in_dimension = [0, 0, 0];
                return_object.status.push(message);
                return return_object;
            }

            if (in_dimension[0] !== cfg.in_channels) {
                message.message = `invalid configuration for conv1d, expected input_channels: ${cfg.in_channels}, received: ${in_dimension[0]}`
                message.required_in_dimension = [cfg.in_channels, 0, 0];
                return_object.status.push(message);
                return return_object;
            }

            const padding = cfg.padding ?? [0, 0];
            const dilation = cfg.dilation ?? [1, 1];
            const stride = cfg.stride ?? [1, 1];
            const kernel = cfg.kernel_size;

            const l_out = convOutputDim(in_dimension[1], padding[0], dilation[0], kernel[0], stride[0]);
            const out_dimension = [cfg.out_channels, l_out];

            message.out_dimension = out_dimension;
            return_object.status.push(message);
            in_dimension = out_dimension;
        } else if (layer_config.type === "maxpool2d" || layer_config.type === "avgpool2d") {
            const cfg = layer_config as MaxPool2dLayerConfig | AvgPool2dLayerConfig;
            let message: ModelDimensionResolveStatusStruct = {
                layer_id: cfg.id,
                in_dimension: in_dimension,
                out_dimension: []
            }

            if (in_dimension.length !== 3) {
                message.message = `${layer_config.type} requires 3d tensor, but received ${in_dimension}`
                message.required_in_dimension = [0, 0, 0]
                return_object.status.push(message);
                return return_object;
            }

            const padding = cfg.padding ?? [0, 0];
            // only maxpool has dilation
            const dilation = (cfg as any).dilation ?? [1, 1];
            const stride = cfg.stride ?? [1, 1];
            const kernel = cfg.kernel_size;

            const h_out = convOutputDim(in_dimension[1], padding[0], dilation[0], kernel[0], stride[0]);
            const w_out = convOutputDim(in_dimension[2], padding[1], dilation[1], kernel[1], stride[1]);
            const out_dimension = [in_dimension[0], h_out, w_out];

            message.out_dimension = out_dimension;
            return_object.status.push(message);
            in_dimension = out_dimension;
        } else if (layer_config.type === "maxpool1d" || layer_config.type === "avgpool1d") {
            const cfg = layer_config as MaxPool1dLayerConfig | AvgPool1dLayerConfig;

            let message: ModelDimensionResolveStatusStruct = {
                layer_id: cfg.id,
                in_dimension: in_dimension,
                out_dimension: []
            }

            if (in_dimension.length !== 2) {
                message.message = `${layer_config.type} requires 2d tensor, but received ${in_dimension}`
                message.required_in_dimension = [0, 0]
                return_object.status.push(message);
                return return_object;
            }

            const padding = cfg.padding ?? [0, 0];
            // only maxpool has dilation
            const dilation = (cfg as any).dilation ?? [1, 1];
            const stride = cfg.stride ?? [1, 1];
            const kernel = cfg.kernel_size;

            const l_out = convOutputDim(in_dimension[1], padding[0], dilation[0], kernel[0], stride[0]);
            const out_dimension = [in_dimension[0], l_out];

            message.out_dimension = out_dimension;
            return_object.status.push(message);
            in_dimension = out_dimension;
        } else if (layer_config.type === "batchnorm2d") {

            let message: ModelDimensionResolveStatusStruct = {
                layer_id: layer_config.id,
                in_dimension: in_dimension,
                out_dimension: []
            }

            if (in_dimension.length !== 3) {
                message.message = `batchnorm2d requires 3d tensor, but received ${in_dimension}`
                message.required_in_dimension = [0, 0, 0]
                return_object.status.push(message);
                return return_object;
            }

            message.out_dimension = message.in_dimension;
            return_object.status.push(message);

        } else if (layer_config.type === "batchnorm1d") {
            let message: ModelDimensionResolveStatusStruct = {
                layer_id: layer_config.id,
                in_dimension: in_dimension,
                out_dimension: []
            }

            if (in_dimension.length !== 2) {
                message.message = `batchnorm1d requires 2d tensor, but received ${in_dimension}`
                message.required_in_dimension = [0, 0]
                return_object.status.push(message);
                return return_object;
            }

            message.out_dimension = message.in_dimension;
            return_object.status.push(message);
        }
    }

    return return_object;
}
