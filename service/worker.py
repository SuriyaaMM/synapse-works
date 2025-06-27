import redis
import json
import traceback
from config import logging, \
                    REDIS_HOST, REDIS_MAIN_QUEUE_NAME, REDIS_PORT, REDIS_TRAIN_QUEUE_NAME

from modelManager import ModelManager
from backendUtils import parseFromLayerConfig, \
                        parseFromTrainConfig, \
                        parseFromDataset
from serializationUtils import serialize_model_manager, deserialize_model_manager

from typedefs import *


def processMessage(message_data, models: list[ModelManager], redis_client: redis.Redis):
    """Processes a single message received from Redis."""
    try:
        # load the string (Json.stringify is called from server-side in typescript)
        message = json.loads(message_data)
        # get event type
        event_type = message.get("event_type")
        
        # handle LAYER_ADDED event 
        if event_type == "LAYER_ADDED":
            id = message.get("model_id")
            layer_config = message.get("layer_config")
            for model in models:
                if model.id == id:
                    parsed_layer_config: LayerConfig = parseFromLayerConfig(layer_config)
                    model.appendLayer(layer_config=parsed_layer_config)
        # handle LAYER_DELETED event
        elif event_type == "LAYER_DELETED":
            id = message.get("model_id")
            layer_id = message.get("layer_id")
            for model in models:
                if model.id == id:
                    model.deleteLayer(layer_id=layer_id)
        # handle LAYER_MODIFIED_EVENT
        elif event_type == "LAYER_MODIFIED":
            id = message.get("model_id")
            layer_id = message.get("layer_id")
            layer_config = message.get("layer_config")
            for model in models:
                if model.id == id:
                    parsed_layer_config: LayerConfig = parseFromLayerConfig(layer_config)
                    model.modifyLayer(parsed_layer_config)
        # handle MODEL_CREATED event
        elif event_type == "MODEL_CREATED":
            id = message.get("model_id")
            name = message.get("name")
            model = ModelManager(model_id=id, model_name=name)
            models.append(model)
        # handle SET_TRAIN_CONFIG
        elif event_type == "SET_TRAIN_CONFIG":
            id = message.get("model_id")
            train_config = message.get("train_config")
            for model in models:
                if model.id == id:
                    parsed_train_config = parseFromTrainConfig(train_config)
                    model.setTrainingConfig(parsed_train_config)
        # handle SET_DATASET
        elif event_type == "SET_DATSET":
            id = message.get("model_id")
            dataset_config = message.get("dataset_config")
            for model in models:
                if model.id == id:
                    parsed_dataset_config = parseFromDataset(dataset_config)
                    model.setDatasetConfig(parsed_dataset_config)
        # handle TRAIN_MODEL
        elif event_type == "TRAIN_MODEL":
            id = message.get("model_id")
            for model in models:
                if model.id == id:
                    model.train(redis_client=redis_client)
        # handle SERIALIZE_MODEL
        elif event_type == "SERIALIZE_MODEL":
            serialize_model_manager(models)
            logging.info("Serialized Models!")
        # handle DESERIALIZE_MODEL
        elif event_type == "DESERIALIZE_MODEL":
            models = deserialize_model_manager()
            logging.info(f"{[model.__getstate__() for model in models]}")
        else:
            print(f"[synapse][redis]: Unknown event type: {event_type}")
    # ----- exceptions
    except Exception as e:
        logging.error(traceback.format_exc())

def start(models: list[ModelManager]):
    """Connects to Redis and starts consuming messages from the list."""
    print("[synapse][redis]: Connecting to Redis...")
    try:
        # try connecting to redis-server
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
        # test connection
        r.ping()
        print("[synapse][redis]: Connected to Redis.")
        print(f"[synapse][redis]: Waiting for messages in '{REDIS_MAIN_QUEUE_NAME}'")
        while True:
            # blocking right tail pop, we left pushed to the queue
            _, message_data = r.brpop([REDIS_MAIN_QUEUE_NAME], timeout=0) # type:ignore
            if message_data:
                processMessage(message_data.decode('utf-8'), models=models, redis_client=r)
    # ----- exceptions
    except Exception as e:
        print(f"[synapse][redis]: unexpected exception: {e}")
    finally:
        logging.info(f"Deleting {REDIS_TRAIN_QUEUE_NAME} & {REDIS_MAIN_QUEUE_NAME}")
        r.delete(REDIS_MAIN_QUEUE_NAME)
        r.delete(REDIS_TRAIN_QUEUE_NAME)
        if 'r_main' in locals() and r.ping():
            print("[synapse][redis]: Redis connection closed.")

if __name__ == '__main__':
    models = []
    start(models)