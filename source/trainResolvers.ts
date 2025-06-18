import { enqueueMessage } from "./redisClient.js";
import { Model, SetTrainConfigArgs, TrainArgs } from "./types";

export async function setTrainConfigResolver(models: Model[], args: SetTrainConfigArgs){
    // find the model 
    const model = models.find(m => m.id === args.modelId);
    // handle model doesn't exist 
    if(!model){
        throw new Error(`[synapse][graphql]: Model with ID ${args.modelId} not found`)
    }

    model.trainConfig = args.trainConfig

    console.log(`[synapse][graphql]: Appending to redis message Queue`)
    // push message to redis
    const message = {
        eventType: "SET_TRAIN_CONFIG",
        modelId: model.id,
        trainConfig: args.trainConfig,
        timestamp: new Date().toISOString()
    };
    await enqueueMessage(message);
    
    return model;
}

export async function trainResolver(models: Model[], args: TrainArgs){
    // find the model 
    const model = models.find(m => m.id === args.modelId);
    // handle model doesn't exist 
    if(!model){
        throw new Error(`[synapse][graphql]: Model with ID ${args.modelId} not found`)
    }

    console.log(`[synapse][graphql]: Appending to redis message Queue`)
    // push message to redis
    const message = {
        eventType: "TRAIN_MODEL",
        modelId: model.id,
        timestamp: new Date().toISOString()
    };
    await enqueueMessage(message);
    
    return model;
}