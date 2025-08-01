from config import logging
import redis
import json
import traceback

from modelManager import ModelManager
from backendUtils import parseFromLayerConfig, \
                        parseFromTrainConfig, \
                        parseFromDataset 
from backendModuleGraphBuilder import parseModuleGraphInput, buildModuleGraph
from serializationUtils import serialize_model_manager, deserialize_model_manager

from typedefs import *
from typing import cast, Optional

def processMessage(message_data, model: ModelManager, redis_client: redis.Redis) -> Optional[ModelManager]:
    """Processes a single message received from Redis."""
    try:
        # load the string (Json.stringify is called from server-side in typescript)
        message = json.loads(message_data)
        # get event type
        event_type = message.get("event_type")
        # handle LAYER_ADDED event 
        if event_type == "LAYER_ADDED":
            id = message.get("model_id")
            layer_config = cast(TSLayerInput, message.get("layer_config"))
            parsed_layer_config: LayerConfig = parseFromLayerConfig(layer_config)
            model.appendLayer(layer_config=parsed_layer_config)
        # handle LAYER_DELETED event
        elif event_type == "LAYER_DELETED":
            id = message.get("model_id")
            layer_id = message.get("layer_id")
            model.deleteLayer(layer_id=layer_id)
        # handle LAYER_MODIFIED_EVENT
        elif event_type == "LAYER_MODIFIED":
            id = message.get("model_id")
            layer_id = message.get("layer_id")
            layer_config = message.get("layer_config")
            parsed_layer_config: LayerConfig = parseFromLayerConfig(layer_config)
            model.modifyLayer(parsed_layer_config)
        # handle CONSTRUCT_MODULE_GRAPH event
        elif event_type == "CONSTRUCT_MODULE_GRAPH":
            module_graph = message.get("module_graph")
            module_graph_input = cast(ModuleGraph, parseModuleGraphInput(module_graph))
            logging.info(f"Received Module Graph: {json.dumps(module_graph, indent=4)}")
            module = buildModuleGraph(module_graph_input)
            model.setModule(module)
        # handle MODEL_CREATED event
        elif event_type == "MODEL_CREATED":
            id = message.get("model_id")
            name = message.get("name")
            model = ModelManager(model_id=id, model_name=name)
            return model
        # handle SET_TRAIN_CONFIG
        elif event_type == "SET_TRAIN_CONFIG":
            train_config = message.get("train_config")
            parsed_train_config = parseFromTrainConfig(train_config)
            model.setTrainingConfig(parsed_train_config) 
        # handle SET_DATASET
        elif event_type == "SET_DATSET":
            dataset_config = message.get("dataset_config")
            parsed_dataset_config = parseFromDataset(dataset_config)
            model.setDatasetConfig(parsed_dataset_config)        
        # handle TRAIN_MODEL
        elif event_type == "TRAIN_MODEL":
            args = cast(TSTrainArgsInput, message.get("args"))
            logging.info(f"Received Train Args: {json.dumps(args, indent=4)}")
            model.train(redis_client=redis_client, args=args)
        # handle SERIALIZE_MODEL
        elif event_type == "SERIALIZE_MODEL":
            serialize_model_manager(model)
            logging.info("Serialized Models!")
        # handle DESERIALIZE_MODEL
        elif event_type == "DESERIALIZE_MODEL":
            id = message.get("model_id")
            model = deserialize_model_manager(id)
            logging.info(f"{model.__getstate__()}")
            return model
        else:
            print(f"[synapse][redis]: Unknown event type: {event_type}")
    # ----- exceptions
    except Exception as e:
        logging.error(traceback.format_exc())