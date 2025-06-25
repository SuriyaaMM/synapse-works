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
    FlattenLayerConfig
} from "./types";

export async function createModelResolver(models: Model[], args: CreateModelArgs){
    // create a new model
    const newModel: Model = {
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

    models.push(newModel);
    console.log(`[synapse][graphql]: Created model: ${newModel.name} (ID: ${newModel.id})`);
    console.log(`[synapse][graphql]: Appending to redis message Queue`)
    // push message to redis
    const message = {
        event_type: "MODEL_CREATED",
        model_id: newModel.id,
        name: newModel.name,
        timestamp: new Date().toISOString()
    };
    
    await enqueueMessage(message);
    return newModel;
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
    in_dimension: number[]
): Promise<ModelDimensionResolveStatus> {
    let return_object: ModelDimensionResolveStatus = { status: [] };

    if (!model) throw new Error('[synapse][graphql]: model does not exist');

    for (const layer_config of model.layers_config) {
        // handle for different layers
        if (layer_config.type === "linear") {
            const linear_layer_config = layer_config as LinearLayerConfig;
            const required_in_dimension = [linear_layer_config.in_features];
            const out_dimension = [linear_layer_config.out_features];

            if (JSON.stringify(required_in_dimension) !== JSON.stringify(in_dimension)) {
                return_object.status.push({
                    layer_id: layer_config.id,
                    message: `invalid configuration for linear layer, expected ${required_in_dimension} but received ${in_dimension}`
                });
            }

            in_dimension = out_dimension;
        } else if (layer_config.type === "flatten") {
            console.log(`[synapse][graphql]: Received for Flatten: ${in_dimension}`)
            const flatten_layer_config = layer_config as FlattenLayerConfig;
            const out_dimension: number[] = [];
            let end = flatten_layer_config.end_dim ?? in_dimension.length;
            if(end == -1){
                end = in_dimension.length;
            }
            let begin = flatten_layer_config.start_dim ?? 0;
            if(begin == 1){
                begin = 0;
            }
            let flattened_dim = 1;

            for (let i = 0; i < end; i++) {
                if (i < begin) {
                    out_dimension.push(in_dimension[i]);
                } else {
                    flattened_dim *= in_dimension[i];
                }
            }

            out_dimension.push(flattened_dim);
            in_dimension = out_dimension;
            console.log(`[synapse][graphql]: Flattened Output: ${in_dimension}`)
        } else if (layer_config.type === "conv2d") {
            const cfg = layer_config as Conv2dLayerConfig;

            if (in_dimension.length !== 3) {
                return_object.status.push({
                    layer_id: layer_config.id,
                    message: `conv2d requires 3d tensor, but received ${in_dimension}`
                });
                continue;
            }

            if (in_dimension[0] !== cfg.in_channels) {
                return_object.status.push({
                    layer_id: layer_config.id,
                    message: `invalid configuration for conv2d, expected input_channels: ${cfg.in_channels}, received: ${in_dimension[0]}`
                });
            }

            const padding = cfg.padding ?? [0, 0];
            const dilation = cfg.dilation ?? [1, 1];
            const stride = cfg.stride ?? [1, 1];
            const kernel = cfg.kernel_size;

            const h_out = convOutputDim(in_dimension[1], padding[0], dilation[0], kernel[0], stride[0]);
            const w_out = convOutputDim(in_dimension[2], padding[1], dilation[1], kernel[1], stride[1]);
            const out_dimension = [cfg.out_channels, h_out, w_out];

            in_dimension = out_dimension;
        } else if (layer_config.type === "conv1d") {
            const cfg = layer_config as Conv1dLayerConfig;

            if (in_dimension.length !== 2) {
                return_object.status.push({
                    layer_id: layer_config.id,
                    message: `conv1d requires 2d tensor, but received ${in_dimension}`
                });
                continue;
            }

            if (in_dimension[0] !== cfg.in_channels) {
                return_object.status.push({
                    layer_id: layer_config.id,
                    message: `invalid configuration for conv1d, expected input_channels: ${cfg.in_channels}, received: ${in_dimension[0]}`
                });
            }

            const padding = cfg.padding ?? [0, 0];
            const dilation = cfg.dilation ?? [1, 1];
            const stride = cfg.stride ?? [1, 1];
            const kernel = cfg.kernel_size;

            const l_out = convOutputDim(in_dimension[1], padding[0], dilation[0], kernel[0], stride[0]);
            const out_dimension = [cfg.out_channels, l_out];

            in_dimension = out_dimension;
        } else if (layer_config.type === "maxpool2d" || layer_config.type === "avgpool2d") {
            const cfg = layer_config as MaxPool2dLayerConfig | AvgPool2dLayerConfig;

            if (in_dimension.length !== 3) {
                return_object.status.push({
                    layer_id: layer_config.id,
                    message: `${layer_config.type} requires 3d tensor, but received ${in_dimension}`
                });
                continue;
            }

            const padding = cfg.padding ?? [0, 0];
            // only maxpool has dilation
            const dilation = (cfg as any).dilation ?? [1, 1];
            const stride = cfg.stride ?? [1, 1];
            const kernel = cfg.kernel_size;

            const h_out = convOutputDim(in_dimension[1], padding[0], dilation[0], kernel[0], stride[0]);
            const w_out = convOutputDim(in_dimension[2], padding[1], dilation[1], kernel[1], stride[1]);

            in_dimension = [in_dimension[0], h_out, w_out];
        } else if (layer_config.type === "maxpool1d" || layer_config.type === "avgpool1d") {
            const cfg = layer_config as MaxPool1dLayerConfig | AvgPool1dLayerConfig;

            if (in_dimension.length !== 2) {
                return_object.status.push({
                    layer_id: layer_config.id,
                    message: `${layer_config.type} requires 2d tensor, but received ${in_dimension}`
                });
                continue;
            }

            const padding = cfg.padding ?? [0, 0];
            // only maxpool has dilation
            const dilation = (cfg as any).dilation ?? [1, 1];
            const stride = cfg.stride ?? [1, 1];
            const kernel = cfg.kernel_size;

            const l_out = convOutputDim(in_dimension[1], padding[0], dilation[0], kernel[0], stride[0]);
            in_dimension = [in_dimension[0], l_out];
        } else if (layer_config.type === "batchnorm2d") {
            if (in_dimension.length !== 3) {
                return_object.status.push({
                    layer_id: layer_config.id,
                    message: `batchnorm2d requires 3d tensor, but received ${in_dimension}`
                });
            }
        } else if (layer_config.type === "batchnorm1d") {
            if (in_dimension.length !== 2) {
                return_object.status.push({
                    layer_id: layer_config.id,
                    message: `batchnorm1d requires 2d tensor, but received ${in_dimension}`
                });
            }
        } else {
            return_object.status.push({
                layer_id: layer_config.id,
                message: `Unknown layer type: ${layer_config.type}`
            });
        }
    }

    return return_object;
}
