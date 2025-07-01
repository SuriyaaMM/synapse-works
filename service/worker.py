import redis
from config import logging, \
                    REDIS_HOST, REDIS_MAIN_QUEUE_NAME, REDIS_PORT, REDIS_TRAIN_QUEUE_NAME

from modelManager import ModelManager
from workerUtils import processMessage

def start(model: ModelManager):
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
                updated_model = processMessage(message_data.decode('utf-8'), model=model, redis_client=r)
                if(isinstance(updated_model,ModelManager)):
                    model = updated_model
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
    model = ModelManager("", "")
    start(model)