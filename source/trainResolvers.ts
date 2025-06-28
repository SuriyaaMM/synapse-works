import { enqueueMessage } from "./redisClient.js";
import { Model, SetTrainConfigArgs, TrainArgs } from "./types";

export async function setTrainConfigResolver(model: Model, args: SetTrainConfigArgs){
    // handle model doesn't exist 
    if(!model){
        throw new Error(`[synapse][graphql]: Model doesn't exist yet, create it first`)
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

export async function trainResolver(model: Model, args: TrainArgs){
    // handle model doesn't exist 
    if(!model){
        throw new Error(`[synapse][graphql]: Model doesn't exist yet, create it first`)
    }

    console.log(`[synapse][graphql]: Appending to redis message Queue`)
    // push message to redis
    const message = {
        event_type: "TRAIN_MODEL",
        model_id: model.id,
        args: args.args,
        timestamp: new Date().toISOString()
    };
    await enqueueMessage(message);
    
    return model;
}