import { v4 as uuidv4 } from "uuid";
import { enqueueMessage } from "./redisClient.js";
import { 
    Model,
    CreateModelArgs 
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