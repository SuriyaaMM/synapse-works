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

const models: Model[] = [];

export const resolvers = {
    // graphql interface inferring for Layer
    LayerConfig: {
        // for inferring underlying concrete type
        __resolveType(layer_config: LayerConfig, _: unknown){
            
            if(layer_config.type === "linear"){
                // must match the one in schema
                return 'LinearLayerConfig';
            }
            else if(layer_config.type == "conv2d"){
                return 'Conv2dLayerConfig';
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
        }
    }
}