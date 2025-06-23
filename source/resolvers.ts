import {
    LayerConfig,
    Model,
    AppendLayerArgs,
    SetDatasetArgs,
    SetTrainConfigArgs,
    CreateModelArgs,
    DatasetConfig,
    TrainArgs
} from "./types.js"

import { appendLayerResolver } from "./layerResolver.js";
import { createModelResolver } from './modelResolver.js';
import { setTrainConfigResolver, trainResolver } from './trainResolvers.js';
import { setDatasetResolver } from './datasetResolver.js';
import { dequeueMessage } from "./redisClient.js";
import { spawn } from "child_process";
import { loadResolver, saveResolver } from "./save.js";

const models: Model[] = [];

export const resolvers = {
    // graphql interface inferring for Layer
    LayerConfig: {
        // for inferring underlying concrete type
        __resolveType(layer_config: LayerConfig, _: unknown){
            
            if(layer_config.type === "linear"){
                return 'LinearLayerConfig';
            }
            else if(layer_config.type == "conv2d"){
                return 'Conv2dLayerConfig';
            }
            else if(layer_config.type == "conv1d"){
                return 'Conv1dLayerConfig';
            }
            else if(layer_config.type == "maxpool2d"){
                return 'MaxPool2dLayerConfig';
            }
            else if(layer_config.type == "maxpool1d"){
                return 'MaxPool1dLayerConfig';
            }
            else if(layer_config.type == "avgpool2d"){
                return 'AvgPool2dLayerConfig';
            }
            else if(layer_config.type == "avgpool1d"){
                return 'AvgPool1dLayerConfig';
            }
            else if(layer_config.type == "batchnorm2d"){
                return 'BatchNorm2dLayerConfig';
            }
            else if(layer_config.type == "batchnorm1d"){
                return 'BatchNorm1dLayerConfig';
            }
            else if(layer_config.type == "flatten"){
                return 'FlattenLayerConfig';
            }
            else if(layer_config.type == "dropout"){
                return 'DropoutLayerConfig';
            }
            else if(layer_config.type == "elu"){
                return 'ELULayerConfig';
            }
            else if(layer_config.type == "relu"){
                return 'ReLULayerConfig';
            }
            else if(layer_config.type == "leakyrelu"){
                return 'LeakyReLULayerConfig';
            }
            else if(layer_config.type == "sigmoid"){
                return 'SigmoidLayerConfig';
            }
            else if(layer_config.type == "logsigmoid"){
                return 'LogSigmoidLayerConfig';
            }
            else if(layer_config.type == "tanh"){
                return 'TanhLayerConfig';
            }
            return null;
        }
    },
    // graphql interface inferring for Dataset
    DatasetConfig: {
        // for inferring underlying concrete type
        __resolveType(dataset_config: DatasetConfig, _: unknown){

            if(dataset_config.name === "mnist"){
                // must match the one in schema
                return 'MNISTDatasetConfig';
            }
            else if(dataset_config.name === "cifar10"){
                return 'CIFAR10DatasetConfig';
            }
            return null;
        }
    },
    // graphql queries
    Query: {
        // getModel query
        // return the model based on id
        getModel: (_: unknown, {id}: {id:string}) =>{
            return models.find(m => m.id === id);
        },
        // getModels query
        // return the models list
        getModels: () => models,
        // getTrainStatus query
        // return the status (pop from redis queue)
        getTrainingStatus: () => dequeueMessage()
    },
    // graphql mutations
    Mutation: {
        // createModel mutation
        createModel: async (_: unknown, args: CreateModelArgs) => {
            return await createModelResolver(models, args);
        },
        // appendLayer mutation
        appendLayer: async (_: unknown, args : AppendLayerArgs) => {
            return await appendLayerResolver(models, args);
        },
        // setTrainConfig mutation
        setTrainConfig: async (_:unknown, args: SetTrainConfigArgs) => {
            return await setTrainConfigResolver(models, args);
        },
        // setDataset mutation
        setDataset: async (_:unknown, args: SetDatasetArgs) => {
            return await setDatasetResolver(models, args);
        },
        // train mutation
        train: async (_:unknown, args: TrainArgs) => {
            return await trainResolver(models, args);
        },
        // save mutation
        save: async(_:unknown) => {
            return await saveResolver(models);
        },
        // load mutation
        load: async(_:unknown) => {
            return await loadResolver();
        },
        // startTensorboard mutation
        startTensorboard: async(_:unknown) => {
            const tb = spawn('tensorboard', ['--logdir', './tbsummary']);

            tb.stdout.on('data', (data) => {
            console.log(`[TensorBoard]: ${data}`);
            });

            tb.stderr.on('data', (data) => {
            console.error(`[TensorBoard Error]: ${data}`);
            });

            tb.on('close', (code) => {
            console.log(`[TensorBoard exited with code ${code}]`);
            });

            return `http://localhost:6006`;
        }
    }
}