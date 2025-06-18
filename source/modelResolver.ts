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
        layers: [], // initialize model with no layers to begin with
        trainConfig : {
            epochs: 50, 
            batch_size: 64, 
            optimizer: "adam",
            optimizerConfig: {
                lr: 3e-4
            }, 
            loss_function: "ce"
            }, // default train config
            dataset: {
                name: "mnist",
                split_length: [0.8, 0.2]
            }
    }

    models.push(newModel);
    console.log(`[synapse][graphql]: Created model: ${newModel.name} (ID: ${newModel.id})`);
    console.log(`[synapse][graphql]: Appending to redis message Queue`)
    // push message to redis
    const message = {
        eventType: "MODEL_CREATED",
        modelId: newModel.id,
        name: newModel.name,
        timestamp: new Date().toISOString()
    };
    
    await enqueueMessage(message);
    return newModel;
}