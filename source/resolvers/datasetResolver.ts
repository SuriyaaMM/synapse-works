import { enqueueMessage } from "../redisClient.js";
import { 
    DatasetConfig,
    DatasetConfigInput, 
    MNISTDatasetConfig, 
    CIFAR10DatasetConfig,
    CustomCSVDatasetConfig,
    ImageFolderDatasetConfig } from "../types/datasetTypes.js";

import { Model } from "../types/modelTypes.js";
import { SetDatasetArgs } from "../types/argTypes.js";

type DatasetHandlerMap = {
    [K in DatasetConfig['name']]: (config: DatasetConfigInput) => DatasetConfig;
};

// handle runtime validation of datasets
export const datasetHandlers: DatasetHandlerMap = {
    "mnist": (dataset: DatasetConfigInput) => {
        // destructure the dataset
        const {split_length, shuffle, batch_size,mnist} = dataset;
        // if mnistConfig is not found, report error
        if(!mnist) throw new Error("[synapse]: mnist config is missing");
        // create MNISTDataset object & return it
        const newDataset: MNISTDatasetConfig = {
            name: "mnist",
            split_length: split_length,
            shuffle: shuffle,
            batch_size: batch_size,
            root: mnist.root,
            train: mnist.train,
            download: mnist.download,
            transform: mnist.transform
        };
        return newDataset;
    },
    "cifar10": (dataset: DatasetConfigInput) => {
        // destructure the dataset
        const {split_length, shuffle, batch_size, cifar10} = dataset;
        // if cifar10Config is not found, report error
        if(!cifar10) throw new Error("[synapse]: cifar10 config is missing");
        // create CIFAR10Dataset object & return it
        const newDataset: CIFAR10DatasetConfig = {
            name: "cifar10",
            split_length: split_length,
            shuffle: shuffle,
            batch_size: batch_size,
            root: cifar10.root,
            train: cifar10.train,
            download: cifar10.download,
            transform: cifar10.transform
        };
        return newDataset;
    },
    "custom_csv": (dataset: DatasetConfigInput) => {
        // destructure the dataset
        const {split_length, shuffle, batch_size, custom_csv} = dataset;
        // if customCSVDatasetConfig is not found, report error
        if(!custom_csv) throw new Error("[synapse]: custom_csv config is missing");
        // create CustomCSVDataset object & return it
        const newDataset: CustomCSVDatasetConfig = {
            name: "custom_csv",
            split_length: split_length,
            shuffle: shuffle,
            batch_size: batch_size,
            root: custom_csv.root,
            feature_columns: custom_csv.feature_columns,
            label_columns: custom_csv.label_columns,
            is_regression_task: custom_csv.is_regression_task
        };
        return newDataset;
    },
    "image_folder": (dataset: DatasetConfigInput) => {
        // destructure the dataset
        const {split_length, shuffle, batch_size, image_folder} = dataset;
        // if image_folder Config is not found, report error
        if(!image_folder) throw new Error("[synapse]: custom_csv config is missing");
        // create ImageFolderDataset object & return it
        const newDataset: ImageFolderDatasetConfig = {
            name: "image_folder",
            split_length: split_length,
            shuffle: shuffle,
            batch_size: batch_size,
            root: image_folder.root,
            transform: image_folder.transform,
            allow_empty: image_folder.allow_empty
        };
        return newDataset;
    }
}

export async function setDatasetResolver(model: Model, args: SetDatasetArgs){
    // handle model doesn't exist 
    if(!model){
        throw new Error(`[synapse]: Model doesn't exist yet, create it first`)
    }
    // get the corresponding dataset
    const handler = datasetHandlers[args.dataset_config.name];
    // handle invalid dataset
    if(!handler) throw new Error(`[synapse]: Dataset with name ${args.dataset_config.name} doesn't exist`);
    // create new dataset object
    const new_dataset_config = handler(args.dataset_config)
    // set newDataset in the model
    model.dataset_config = new_dataset_config;
    console.log(`[symapse]: Set new dataset configuration ${JSON.stringify(new_dataset_config)} to model (Id = ${model.id})`);
    
    // push message to redis
    console.log(`[synapse][graphql]: Appending SET_DATASET Event to redis message Queue`)
    const message = {
        event_type: "SET_DATSET",
        model_id: model.id,
        dataset_config: new_dataset_config,
        timestamp: new Date().toISOString()
    };
    await enqueueMessage(message);

    return model;
}