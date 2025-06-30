import { createModelResolver, validateModelResolver } from './modelResolver.js';
import { setTrainConfigResolver, trainResolver } from './trainResolvers.js';
import { setDatasetResolver } from './datasetResolver.js';
import { dequeueMessage } from "./redisClient.js";
import { spawn, ChildProcess } from "child_process";
import { loadModelResolver, saveResolver } from "./saveResolver.js";
import {
    LayerConfig,
    Model,
    AppendLayerArgs,
    SetDatasetArgs,
    SetTrainConfigArgs,
    CreateModelArgs,
    DatasetConfig,
    TrainArgs,
    DeleteLayerArgs,
    ModifyLayerArgs,
    ModuleGraph,
    AppendToModuleGraphArgs,
    DeleteInModuleGraphArgs,
    ConnectInModuleGraphArgs,
    DisconnectInModuleGraphArgs,
    BuildModuleGraphArgs
} from "./types.js"
import { 
    appendLayerResolver, 
    deleteLayerResolver, 
    modifyLayerResolver } from "./layerResolver.js";
import { 
    appendToModuleGraphResolver, 
    buildModuleGraphResolver, 
    connectInModuleGraphResolver, 
    deleteInModuleGraphResolver, 
    disconnectInModuleGraphResolver, 
    validateModuleGraphResolver} from "./graphResolver.js";

let model: Model;
export let tensorboardProcess: ChildProcess = null;

export function setModel(updated_model: Model){
    model = updated_model;
}

export function setModuleGraph(module_graph: ModuleGraph){
    model.module_graph = module_graph;
}

export const resolvers = {
    // graphql interface inferring for Layer
    LayerConfig: {
        // for inferring underlying concrete type
        __resolveType(layer_config: LayerConfig, _: unknown){
            
            if(layer_config.type === "linear"){
                return 'LinearLayerConfig';
            }
            else if(layer_config.type == "conv2d"){
                return 'Conv2dLayerConfig';
            }
            else if(layer_config.type == "convtranspose2d"){
                return 'ConvTranspose2dLayerConfig';
            }
            else if(layer_config.type == "conv1d"){
                return 'Conv1dLayerConfig';
            }
            else if(layer_config.type == "maxpool2d"){
                return 'MaxPool2dLayerConfig';
            }
            else if(layer_config.type == "maxpool1d"){
                return 'MaxPool1dLayerConfig';
            }
            else if(layer_config.type == "avgpool2d"){
                return 'AvgPool2dLayerConfig';
            }
            else if(layer_config.type == "avgpool1d"){
                return 'AvgPool1dLayerConfig';
            }
            else if(layer_config.type == "batchnorm2d"){
                return 'BatchNorm2dLayerConfig';
            }
            else if(layer_config.type == "batchnorm1d"){
                return 'BatchNorm1dLayerConfig';
            }
            else if(layer_config.type == "flatten"){
                return 'FlattenLayerConfig';
            }
            else if(layer_config.type == "dropout"){
                return 'DropoutLayerConfig';
            }
            else if(layer_config.type == "dropout2d"){
                return 'Dropout2dLayerConfig';
            }
            else if(layer_config.type == "elu"){
                return 'ELULayerConfig';
            }
            else if(layer_config.type == "relu"){
                return 'ReLULayerConfig';
            }
            else if(layer_config.type == "leakyrelu"){
                return 'LeakyReLULayerConfig';
            }
            else if(layer_config.type == "sigmoid"){
                return 'SigmoidLayerConfig';
            }
            else if(layer_config.type == "logsigmoid"){
                return 'LogSigmoidLayerConfig';
            }
            else if(layer_config.type == "tanh"){
                return 'TanhLayerConfig';
            }
            else if(layer_config.type == "cat"){
                return 'CatLayerConfig';
            }
            return null;
        }
    },
    // graphql interface inferring for Dataset
    DatasetConfig: {
        // for inferring underlying concrete type
        __resolveType(dataset_config: DatasetConfig, _: unknown){

            if(dataset_config.name === "mnist"){
                // must match the one in schema
                return 'MNISTDatasetConfig';
            }
            else if(dataset_config.name === "cifar10"){
                return 'CIFAR10DatasetConfig';
            }
            else if(dataset_config.name == "image_folder"){
                return 'ImageFolderDatasetConfig';
            }
            else if(dataset_config.name == "custom_csv"){
                return 'CustomCSVDatasetConfig';
            }
            return null;
        }
    },
    // graphql queries
    Query: {
        // getModel query
        // return the model based on id
        getModel: () =>{
            return model;
        },
        // getTrainStatus query
        // return the status (pop from redis queue)
        getTrainingStatus: () => dequeueMessage(),
        // validateModel query
        validateModel: async (_: unknown, {in_dimension} : {in_dimension:number[]}) => {
            return validateModelResolver(model, in_dimension);
        }
    },
    // graphql mutations
    Mutation: {
        // createModel mutation
        createModel: async (_: unknown, args: CreateModelArgs) => {
            return await createModelResolver(args);
        },
        // appendLayer mutation
        appendLayer: async (_: unknown, args : AppendLayerArgs) => {
            return await appendLayerResolver(model, args);
        },
        // deleteLayer mutation
        deleteLayer: async(_:unknown, args: DeleteLayerArgs) => {
            return await deleteLayerResolver(model, args)
        },
        // modifyLayer mutation
        modifyLayer: async(_:unknown, args: ModifyLayerArgs) => {
            return await modifyLayerResolver(model, args);
        },
        // appendToModuleGraph mutation
        appendToModuleGraph: async (_:unknown, args: AppendToModuleGraphArgs) => {
            return await appendToModuleGraphResolver(args);
        },
        // deleteInModuleGraph mutation
        deleteInModuleGraph: async(_:unknown, args: DeleteInModuleGraphArgs) => {
            return await deleteInModuleGraphResolver(args);
        },
        // connectInModuleGraph mutation
        connectInModuleGraph: async(_:unknown, args: ConnectInModuleGraphArgs) => {
            return await connectInModuleGraphResolver(args);
        },
        // diconnectInModuleGraph mutation
        disconnectInModuleGraph: async(_:unknown, args: DisconnectInModuleGraphArgs) => {
            return await disconnectInModuleGraphResolver(args);
        },
        buildModuleGraph: async(_:unknown) => {
            return await buildModuleGraphResolver(model);
        },
        validateModuleGraph: async(_:unknown, {in_dimension}: {in_dimension: number[]}) => {
            return await validateModuleGraphResolver(model, in_dimension);
        },
        // setTrainConfig mutation
        setTrainConfig: async (_:unknown, args: SetTrainConfigArgs) => {
            return await setTrainConfigResolver(model, args);
        },
        // setDataset mutation
        setDataset: async (_:unknown, args: SetDatasetArgs) => {
            return await setDatasetResolver(model, args);
        },
        // train mutation
        train: async (_:unknown, args: TrainArgs) => {
            return await trainResolver(model, args);
        },
        // save mutation
        saveModel: async(_:unknown) => {
            return await saveResolver(model);
        },
        // load mutation
        loadModel: async(_:unknown, {model_id} : {model_id:string}) => {
            return await loadModelResolver(model_id);
        },
        // startTensorboard mutation
        startTensorboard: async(_:unknown) => {
            tensorboardProcess = spawn('tensorboard', ['--logdir', './tbsummary']);

            tensorboardProcess.stdout.on('data', (data) => {
            console.log(`[TensorBoard]: ${data}`);
            });

            tensorboardProcess.stderr.on('data', (data) => {
            console.error(`[TensorBoard Error]: ${data}`);
            });

            tensorboardProcess.on('close', (code) => {
            console.log(`[TensorBoard exited with code ${code}]`);
            });

            return `http://localhost:6006`;
        }
    }
}