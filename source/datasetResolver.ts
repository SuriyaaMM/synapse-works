import { enqueueMessage } from "./redisClient.js";
import { DatasetInput, MNISTDataset, Model, SetDatasetArgs } from "./types";
import { Dataset,  } from "./types";

type DatasetHandlerMap = {
    [K in Dataset['name']]: (config: DatasetInput) => Dataset;
};

// handle runtime validation of datasets
export const datasetHandlers: DatasetHandlerMap = {
    "mnist": (dataset: DatasetInput) => {
        // destructure the dataset
        const {split_length, shuffle, mnist} = dataset;
        // if mnistConfig is not found, report error
        if(!mnist) throw new Error("[synapse][graphql]: mnist config is missing");
        // create MNISTDataset object & return it
        const newDataset: MNISTDataset = {
            name: "mnist",
            split_length: split_length,
            shuffle: shuffle,
            root: mnist.root,
            train: mnist.train,
            download: mnist.download
        };
        return newDataset;
    }
}

export async function setDatasetResolver(models: Model[], args: SetDatasetArgs){
    // find the model 
    const model = models.find(m => m.id === args.modelId);
    // handle model doesn't exist 
    if(!model){
        throw new Error(`[synapse][graphql]: Model with ID ${args.modelId} not found`)
    }
    // get the corresponding dataset
    const handler = datasetHandlers[args.datasetInput.name];
    // handle invalid dataset
    if(!handler) throw new Error(`[synapse][graphql]: dataset with name ${args.datasetInput.name} doesn't exist`);
    // create new dataset object
    const newDataset = handler(args.datasetInput)
    // set newDataset in the model
    model.dataset = newDataset;
    console.log(`[symapse][graphql]: set new dataset configuration ${JSON.stringify(newDataset)} to model (Id = ${model.id})`);
    
    // push message to redis
    console.log(`[synapse][graphql]: Appending to redis message Queue`)
    const message = {
        eventType: "SET_DATSET",
        modelId: model.id,
        dataset: newDataset,
        timestamp: new Date().toISOString()
    };
    await enqueueMessage(message);

    return model;
}