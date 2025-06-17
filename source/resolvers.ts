import { v4 as uuidv4 } from "uuid";
import { enqueueMessage } from './redisClient.js'; 

// ----- Type definitions for graphql Objects
interface Layer {
    id: string;
    type: string;
    name?: string;
}
interface LinearLayer extends Layer {
    type: 'linear';
    in_features: number;
    out_features: number;
}

type LinearLayerConfig = {
  name?: string;
  in_features: number;
  out_features: number;
};

type LayerConfig = {
  type: string; 
  linear?: LinearLayerConfig;
};

type AppendLayerArgs = {
    modelId: string;
    layerConfig: LayerConfig;
};

type OptimizerConfig = {
    name: string;
    lr: number;
}  

type TrainConfig = {
    epochs: number;
    batch_size: number;
    optimizerConfig: OptimizerConfig
    loss_function: string;
}

type TrainConfigArgs = {
    modelId: string;
    trainConfig: TrainConfig;
}

type Model  = {
  id: string;
  name: string;
  layers: Layer[];
  trainConfig: TrainConfig;
}

// in memory model
const models: Model[] = [];

export const resolvers = {
    // graphql interface inferring
    Layer: {
        // for inferring underlying concrete type
        __resolveType(layer: Layer, _: unknown){
            
            if(layer.type === "linear"){
                // must match the one in schema
                return 'LinearLayer';
            }
            return null;
        }
    },
    // graphql queries
    Query: {
        // getModel query
        // return the model based on id
        getModel: (_: unknown, {id}: {id:string}) =>{
            return models.find(m => m.id === id);
        },
        // getModels query
        // return the models list
        getModels: () => models,
    },
    // graphql mutations
    Mutation: {
        /*
        createModel mutation
        
        Args:
            name:string, name of the model
        */
        createModel: async (_: unknown, {name}: {name:string}) => {
            // create a new model
            const newModel: Model = {
                id: uuidv4(), // generate unique uuid
                name: name, 
                layers: [], // initialize model with no layers to begin with
                trainConfig : {
                    epochs: 50, 
                    batch_size: 64, 
                    optimizerConfig: {
                        name: "adam",
                        lr: 3e-4
                    }, 
                    loss_function: "ce"
                } // default train config
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
        },


        /* 
        appendLayer mutation

        Args:
            modelId: Id, unique id of the model,
            layerConfig: any, configuration of the layer
        */
        appendLayer: async (_: unknown, {modelId, layerConfig} : AppendLayerArgs) => {
            // find the model 
            const model = models.find(m => m.id === modelId);
            // handle model doesn't exist 
            if(!model){
                throw new Error(`[synapse][graphql]: Model with ID ${modelId} not found`)
            }

            // handle linear model
            if(layerConfig.type == "linear"){
                // destructure layerConfig
                const {type, linear} = layerConfig;

                // initialize new layer & its uuid
                let newLayer: LinearLayer;
                const layerId = uuidv4();

                if(!linear) throw new Error(`[synapse][graphql]: Linear layer config is missing`);
                
                newLayer = {
                    id: layerId,
                    type: "linear",
                    name: linear.name || `linear_${layerId.substring(0, 4)}`,
                    in_features: linear.in_features,
                    out_features: linear.out_features
                };

                // push layer to model
                model.layers.push(newLayer)
                console.log(`[synapse][graphql]: Appended ${type} layer (ID: ${newLayer.id}) to model ${model.name} (Model ID: ${model.id})`);
            }
            else{
                throw new Error(`[synapse][graphql]: Unsupported layer type: ${layerConfig.type}`)
            }
            
            console.log(`[synapse][graphql]: Appending to redis message Queue`)
            // push message to redis
            const message = {
                eventType: "LAYER_ADDED",
                modelId: model.id,
                layerData: model.layers.at(-1),
                timestamp: new Date().toISOString()
            };

            await enqueueMessage(message);
            // return model
            return model;
        },


        /*
        setTrainConfig mutation

        Args:
            modelId: Id, unique id of the model
            trainConfig: TrainConfig, training configurations
        */
        setTrainConfig: async (_:unknown, {modelId, trainConfig}:TrainConfigArgs) => {

            // find the model 
            const model = models.find(m => m.id === modelId);
            // handle model doesn't exist 
            if(!model){
                throw new Error(`[synapse][graphql]: Model with ID ${modelId} not found`)
            }

            model.trainConfig = trainConfig

            console.log(`[synapse][graphql]: Appending to redis message Queue`)
            // push message to redis
            const message = {
                eventType: "SET_TRAIN_CONFIG",
                modelId: model.id,
                trainConfig: trainConfig,
                timestamp: new Date().toISOString()
            };
            await enqueueMessage(message);
            return model;
        }
    }
}