import { enqueueMessage } from "./redisClient.js";
import { CIFAR10DatasetConfig, CustomCSVDatasetConfig, DatasetConfigInput, MNISTDatasetConfig, Model, SetDatasetArgs } from "./types";
import { DatasetConfig  } from "./types";

type DatasetHandlerMap = {
    [K in DatasetConfig['name']]: (config: DatasetConfigInput) => DatasetConfig;
};

// handle runtime validation of datasets
export const datasetHandlers: DatasetHandlerMap = {
    "mnist": (dataset: DatasetConfigInput) => {
        // destructure the dataset
        const {split_length, shuffle, batch_size,mnist} = dataset;
        // if mnistConfig is not found, report error
        if(!mnist) throw new Error("[synapse][graphql]: mnist config is missing");
        // create MNISTDataset object & return it
        const new_dataset: MNISTDatasetConfig = {
            name: "mnist",
            split_length: split_length,
            shuffle: shuffle,
            batch_size: batch_size,
            root: mnist.root,
            train: mnist.train,
            download: mnist.download,
            transform: mnist.transform
        };
        return new_dataset;
    },
    "cifar10": (dataset: DatasetConfigInput) => {
        // destructure the dataset
        const {split_length, shuffle, batch_size, cifar10} = dataset;
        // if cifar10 is not found, report error
        if(!cifar10) throw new Error("[synapse][graphql]: cifar10 config is missing");
        // create CIFAR10Dataset object & return it
        const new_dataset: CIFAR10DatasetConfig = {
            name: "cifar10",
            split_length: split_length,
            shuffle: shuffle,
            batch_size: batch_size,
            root: cifar10.root,
            train: cifar10.train,
            download: cifar10.download,
            transform: cifar10.transform
        };
        return new_dataset;
    },
    "custom_csv": (dataset: DatasetConfigInput) => {
        // destructure the dataset
        const {split_length, shuffle, batch_size, custom_csv} = dataset;
        // if custom_csv is not found, report error
        if(!custom_csv) throw new Error("[synapse][graphql]: custom csv config is missing");
        // create CustomCSVDatasetConfig
        const new_dataset: CustomCSVDatasetConfig = {
            name: "custom_csv",
            split_length: split_length,
            shuffle: shuffle,
            batch_size: batch_size,
            path_to_csv: custom_csv.path_to_csv,
            feature_columns: custom_csv.feature_columns,
            label_columns: custom_csv.label_columns,
            transform: custom_csv.transform
        }

        return new_dataset;
    }
}

export async function setDatasetResolver(models: Model[], args: SetDatasetArgs){
    // find the model 
    const model = models.find(m => m.id === args.model_id);
    // handle model doesn't exist 
    if(!model){
        throw new Error(`[synapse][graphql]: Model with ID ${args.model_id} not found`)
    }
    // get the corresponding dataset
    const handler = datasetHandlers[args.dataset_config.name];
    // handle invalid dataset
    if(!handler) throw new Error(`[synapse][graphql]: dataset with name ${args.dataset_config.name} doesn't exist`);
    // create new dataset object
    const new_dataset_config = handler(args.dataset_config)
    // set newDataset in the model
    model.dataset_config = new_dataset_config;
    console.log(`[symapse][graphql]: set new dataset configuration ${JSON.stringify(new_dataset_config)} to model (Id = ${model.id})`);
    
    // push message to redis
    console.log(`[synapse][graphql]: Appending to redis message Queue`)
    const message = {
        event_type: "SET_DATSET",
        model_id: model.id,
        dataset_config: new_dataset_config,
        timestamp: new Date().toISOString()
    };
    await enqueueMessage(message);

    return model;
}