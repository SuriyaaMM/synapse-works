import redis
import json
import time

REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_QUEUE_NAME = 'model_layer_updates_queue'

def processMessage(messageData):
    """Processes a single message received from Redis."""
    try:
        # load the string (Json.stringify is called from server-side in typescript)
        message = json.loads(messageData)
        print(f"[synapse][redis] : Received message: {message}")

        eventType = message.get("eventType")
        modelId = message.get("modelId")
        layerData = message.get("layerData")
        # handle LAYER_ADDED event 
        if eventType == "LAYER_ADDED":
            print(f"[synapse][redis]: Layer added to model {modelId}: {layerData.get('type')} - {layerData.get('name')}")
        else:
            print(f"[synapse][redis]: Unknown event type: {eventType}")
    # ----- exceptions
    except json.JSONDecodeError:
        print(f"[synapse][redis]: Invalid JSON received: {messageData.decode()}")
    except Exception as e:
        print(f"[synapse][redis]: processing message: {e}")

def start():
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
                processMessage(messageData.decode('utf-8'))
    # ----- exceptions
    except Exception as e:
        print(f"[synapse][redis]: unexpected exception: {e}")
    finally:
        if 'r' in locals() and r.ping():
            print("[synapse][redis]: Redis connection closed.")

if __name__ == '__main__':
    start()