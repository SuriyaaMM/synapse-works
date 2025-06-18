import {
    Layer,
    Model,
    AppendLayerArgs,
    SetDatasetArgs,
    SetTrainConfigArgs,
    CreateModelArgs,
    Dataset
} from "./types.js"

import { appendLayerResolver } from "./layerResolver.js";
import { createModelResolver } from './modelResolver.js';
import { setTrainConfigResolver } from './trainResolvers.js';
import { setDatasetResolver } from './datasetResolver.js';

const models: Model[] = [];

export const resolvers = {
    // graphql interface inferring for Layer
    Layer: {
        // for inferring underlying concrete type
        __resolveType(layer: Layer, _: unknown){
            
            if(layer.type === "linear"){
                // must match the one in schema
                return 'LinearLayer';
            }
            return null;
        }
    },
    // graphql interface inferring for Dataset
    Dataset: {
        // for inferring underlying concrete type
        __resolveType(dataset: Dataset, _: unknown){

            if(dataset.name === "mnist"){
                // must match the one in schema
                return 'MNISTDataset';
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
        }
    }
}