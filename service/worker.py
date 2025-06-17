import redis
import json

from modelManager import ModelManager
from backendUtils import parseFromLayerConfig, \
                        parseFromTrainConfig

REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_QUEUE_NAME = 'model_layer_updates_queue'

def processMessage(messageData, models: list[ModelManager]):
    """Processes a single message received from Redis."""
    try:
        # load the string (Json.stringify is called from server-side in typescript)
        message = json.loads(messageData)
        print(f"[synapse][redis] : Received message: {message}")
        # get event type
        eventType = message.get("eventType")
        
        # handle LAYER_ADDED event 
        if eventType == "LAYER_ADDED":
            id = message.get("modelId")
            layerConfig = message.get("layerData")
            for model in models:
                if model.id == id:
                    layerConfigObj = parseFromLayerConfig(layer_config=layerConfig)
                    model.appendLayer(layer_config=layerConfigObj)
        # handle MODEL_CREATED event
        elif eventType == "MODEL_CREATED":
            id = message.get("modelId")
            name = message.get("name")
            model = ModelManager(model_id=id, model_name=name)
            models.append(model)
        # handle SET_TRAIN_CONFIG
        elif eventType == "SET_TRAIN_CONFIG":
            id = message.get("modelId")
            trainConfig = message.get("trainConfig")
            for model in models:
                if model.id == id:
                    trainConfigObj = parseFromTrainConfig(train_config=trainConfig)
                    model.setTrainingConfig(trainConfigObj)
                    model.dumpNetworkForTraining()
        else:
            print(f"[synapse][redis]: Unknown event type: {eventType}")
    # ----- exceptions
    except json.JSONDecodeError:
        print(f"[synapse][redis]: Invalid JSON received: {messageData.decode()}")
    except Exception as e:
        print(f"[synapse][redis]: processing message: {e}")

def start(models: list[ModelManager]):
    """Connects to Redis and starts consuming messages from the list."""
    print("[synapse][redis]: Connecting to Redis...")
    try:
        # try connecting to redis-server
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
        # test connection
        r.ping()
        print("[synapse][redis]: Connected to Redis.")
        print(f"[synapse][redis]: Waiting for messages in '{REDIS_QUEUE_NAME}'")

        while True:
            # blocking right tail pop, we left pushed to the queue
            _, messageData = r.brpop([REDIS_QUEUE_NAME], timeout=0) # type:ignore
            if messageData:
                processMessage(messageData.decode('utf-8'), models=models)
    # ----- exceptions
    except Exception as e:
        print(f"[synapse][redis]: unexpected exception: {e}")
    finally:
        if 'r' in locals() and r.ping():
            print("[synapse][redis]: Redis connection closed.")

if __name__ == '__main__':
    models = []
    start(models)