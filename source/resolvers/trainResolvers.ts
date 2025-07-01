import { enqueueMessage } from "../redisClient.js";
import { Model } from "../types/modelTypes.js";
import { SetTrainConfigArgs, TrainArgs } from "../types/argTypes.js";

export async function setTrainConfigResolver(model: Model, args: SetTrainConfigArgs){
    // handle model doesn't exist 
    if(!model){
        throw new Error(`[synapse]: Model doesn't exist yet, create it first`)
    }

    model.train_config = args.train_config

    console.log(`[synapse]: Appending SET_TRAIN_CONFIG Event to redis Queue`)
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
        throw new Error(`[synapse]: Model doesn't exist yet, create it first`)
    }

    console.log(`[synapse]: Appending TRAIN_MODEL Event to redis Queue`)
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