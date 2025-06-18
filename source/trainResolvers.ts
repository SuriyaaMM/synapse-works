import { enqueueMessage } from "./redisClient.js";
import { Model, SetTrainConfigArgs, TrainArgs } from "./types";

export async function setTrainConfigResolver(models: Model[], args: SetTrainConfigArgs){
    // find the model 
    const model = models.find(m => m.id === args.model_id);
    // handle model doesn't exist 
    if(!model){
        throw new Error(`[synapse][graphql]: Model with ID ${args.model_id} not found`)
    }

    model.train_config = args.train_config

    console.log(`[synapse][graphql]: Appending to redis message Queue`)
    // push message to redis
    const message = {
        event_type: "SET_TRAIN_CONFIG",
        model_id: model.id,
        train_config: args.train_config,
        timestamp: new Date().toISOString()
    };
    await enqueueMessage(message);
    
    return model;
}

export async function trainResolver(models: Model[], args: TrainArgs){
    // find the model 
    const model = models.find(m => m.id === args.model_id);
    // handle model doesn't exist 
    if(!model){
        throw new Error(`[synapse][graphql]: Model with ID ${args.model_id} not found`)
    }

    console.log(`[synapse][graphql]: Appending to redis message Queue`)
    // push message to redis
    const message = {
        event_type: "TRAIN_MODEL",
        model_id: model.id,
        timestamp: new Date().toISOString()
    };
    await enqueueMessage(message);
    
    return model;
}