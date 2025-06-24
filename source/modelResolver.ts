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
    AvgPool1dLayerConfig
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

export async function validateModelResolver(model: Model, in_dimension: number[]): Promise<ModelDimensionResolveStatus> {
    let return_object: ModelDimensionResolveStatus = {status: []};

    if(!model) throw new Error('[synapse][graphql]: model does not exist');

    for(const layer_config of model.layers_config){
        // linear dimension resolver
        if(layer_config.type == "linear"){
            // we know it is a linear layer
            const linear_layer_config = layer_config as LinearLayerConfig;
            const required_in_dimension = [linear_layer_config.in_features];
            const out_dimension = [linear_layer_config.out_features];
            // if in_dimension length itself greater than 1, then it is not valid
            if(JSON.stringify(required_in_dimension) !== JSON.stringify(in_dimension)){
                return_object.status.push({layer_id: layer_config.id, message: `invalid configuration for linear layer, expected ${required_in_dimension} but received ${in_dimension}`});
            }
            // regardless of whether it was correct dimension or not, set this so that
            // subsequent dimensions also can be validated
            in_dimension = out_dimension;
        }
        // conv2d dimension resolver
        else if(layer_config.type == "conv2d"){
            // we know it is a conv2d layer
            const conv2d_layer_config = layer_config as Conv2dLayerConfig;
            if(in_dimension.length != 3){
                return_object.status.push({layer_id: layer_config.id, message: `conv2d requires 3d tensor, but received ${in_dimension}`});
            }
            else{
                // for now just check whether channel's are matching
                if(in_dimension[0] !== conv2d_layer_config.in_channels){
                    return_object.status.push({layer_id: layer_config.id, message: `invalid configuration for conv2d, expected input_channels: ${conv2d_layer_config.in_channels}, received: ${in_dimension[0]}`});
                }
                
                const padding = conv2d_layer_config.padding || [0, 0];
                const dilation = conv2d_layer_config.dilation || [1, 1];
                const stride = conv2d_layer_config.stride || [1, 1];
                const kernel_size = conv2d_layer_config.kernel_size;
                const intermediate_0 = (2 * padding[0] - dilation[0] * (kernel_size[0] - 1));
                const intermediate_1 = (2 * padding[1] - dilation[1] * (kernel_size[1] - 1));
                
                const c_in = in_dimension[0];
                const h_in = in_dimension[1];
                const w_in = in_dimension[2];
                
                const c_out = conv2d_layer_config.out_channels;
                const h_out = ((h_in * intermediate_0)/stride[0]) + 1;
                const w_out = ((w_in * intermediate_1)/stride[0]) + 1;

                const out_dimension = [c_out, h_out, w_out];
                in_dimension = out_dimension;
            }
        }
        // conv1d dimension resolver
        else if(layer_config.type == "conv1d"){
            // we know it is a conv1d layer
            const conv1d_layer_config = layer_config as Conv1dLayerConfig;
            if(in_dimension.length != 2){
                return_object.status.push({layer_id: layer_config.id, message: `conv1d requires 2d tensor, but received ${in_dimension}`});
            }
            else{
                // for now just check whether channel's are matching
                if(in_dimension[0] !== conv1d_layer_config.in_channels){
                    return_object.status.push({layer_id: layer_config.id, message: `invalid configuration for conv1d, expected input_channels: ${conv1d_layer_config.in_channels}, received: ${in_dimension[0]}`});
                }
                
                const padding = conv1d_layer_config.padding || [0, 0];
                const dilation = conv1d_layer_config.dilation || [1, 1];
                const stride = conv1d_layer_config.stride || [1, 1];
                const kernel_size = conv1d_layer_config.kernel_size;
                const intermediate = (2 * padding[0] - dilation[0] * (kernel_size[0] - 1));
                
                const c_in = in_dimension[0];
                const l_in = in_dimension[1];
                
                const c_out = conv1d_layer_config.out_channels;
                const l_out = ((l_in * intermediate)/stride[0]) + 1;

                const out_dimension = [c_out, l_out];
                in_dimension = out_dimension;
            }
        }
        // maxpool2d dimension resolver
        else if(layer_config.type == "maxpool2d"){
            // we know it is a maxpool2d layer
            const maxpool2d_layer_config = layer_config as MaxPool2dLayerConfig;
            if(in_dimension.length != 3){
                return_object.status.push({layer_id: layer_config.id, message: `maxpool2d requires 3d tensor, but received ${in_dimension}`});
            }
            else{
                
                const padding = maxpool2d_layer_config.padding || [0, 0];
                const dilation = maxpool2d_layer_config.dilation || [1, 1];
                const stride = maxpool2d_layer_config.stride || [1, 1];
                const kernel_size = maxpool2d_layer_config.kernel_size;
                const intermediate_0 = (2 * padding[0] - dilation[0] * (kernel_size[0] - 1));
                const intermediate_1 = (2 * padding[1] - dilation[1] * (kernel_size[1] - 1));
                
                const c = in_dimension[0];
                const h_in = in_dimension[1];
                const w_in = in_dimension[2];
                
                const h_out = ((h_in * intermediate_0)/stride[0]) + 1;
                const w_out = ((w_in * intermediate_1)/stride[0]) + 1;

                const out_dimension = [c, h_out, w_out];
                in_dimension = out_dimension;
            }
        }
        // maxpool1d dimension resolver
        else if(layer_config.type == "maxpool1d"){
            // we know it is a maxpool1d layer
            const maxpool1d_layer_config = layer_config as MaxPool1dLayerConfig;
            if(in_dimension.length != 2){
                return_object.status.push({layer_id: layer_config.id, message: `maxpool1d requires 2d tensor, but received ${in_dimension}`});
            }
            else{
                
                const padding = maxpool1d_layer_config.padding || [0, 0];
                const dilation = maxpool1d_layer_config.dilation || [1, 1];
                const stride = maxpool1d_layer_config.stride || [1, 1];
                const kernel_size = maxpool1d_layer_config.kernel_size;
                const intermediate = (2 * padding[0] - dilation[0] * (kernel_size[0] - 1));
                
                const c = in_dimension[0];
                const l_in = in_dimension[1];
                
                
                const l_out = ((l_in * intermediate)/stride[0]) + 1;

                const out_dimension = [c, l_out];
                in_dimension = out_dimension;
            }
        }
        // avgpool2d dimension resolver
        else if(layer_config.type == "avgpool2d"){
            // we know it is a avgpool2d layer
            const avgpool2d_layer_config = layer_config as AvgPool2dLayerConfig;
            if(in_dimension.length != 3){
                return_object.status.push({layer_id: layer_config.id, message: `avgpool2d requires 3d tensor, but received ${in_dimension}`});
            }
            else{
                
                const padding = avgpool2d_layer_config.padding || [0, 0];
                const stride = avgpool2d_layer_config.stride || [1, 1];
                const kernel_size = avgpool2d_layer_config.kernel_size;
                const intermediate_0 = (2 * padding[0] - (kernel_size[0] - 1));
                const intermediate_1 = (2 * padding[1] - (kernel_size[1] - 1));
                
                const c = in_dimension[0];
                const h_in = in_dimension[1];
                const w_in = in_dimension[2];
                
                const h_out = ((h_in * intermediate_0)/stride[0]) + 1;
                const w_out = ((w_in * intermediate_1)/stride[0]) + 1;

                const out_dimension = [c, h_out, w_out];
                in_dimension = out_dimension;
            }
        }
        // avgpool1d dimension resolver
        else if(layer_config.type == "avgpool1d"){
            // we know it is a avgpool1d layer
            const avgpool1d_layer_config = layer_config as AvgPool1dLayerConfig;
            if(in_dimension.length != 2){
                return_object.status.push({layer_id: layer_config.id, message: `avgpool1d requires 3d tensor, but received ${in_dimension}`});
            }
            else{
                
                const padding = avgpool1d_layer_config.padding || [0, 0];
                const stride = avgpool1d_layer_config.stride || [1, 1];
                const kernel_size = avgpool1d_layer_config.kernel_size;
                const intermediate = (2 * padding[0] - (kernel_size[0] - 1));
                
                const c = in_dimension[0];
                const l_in = in_dimension[1];
                
                const l_out = ((l_in * intermediate)/stride[0]) + 1;

                const out_dimension = [c, l_out];
                in_dimension = out_dimension;
            }
        }
        // batchnorm2d dimension resolver
        else if(layer_config.type == "batchnorm2d"){
            if(in_dimension.length != 3){
                return_object.status.push({layer_id: layer_config.id, message: `batchnorm2d requires 3d tensor, but received ${in_dimension}`});
            }
        }
        // batchnorm1d dimension resolver
        else if(layer_config.type == "batchnorm1d"){
            if(in_dimension.length != 2){
                return_object.status.push({layer_id: layer_config.id, message: `batchnorm1d requires 2d tensor, but received ${in_dimension}`});
            }
        }
    }

    return return_object;
}