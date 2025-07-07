import redis
import os
from config import logging, REDIS_MAIN_QUEUE_NAME, REDIS_TRAIN_QUEUE_NAME
from modelManager import ModelManager
from workerUtils import processMessage
from dotenv import load_dotenv
load_dotenv()

def start(model: ModelManager):
    """Connects to Redis and starts consuming messages from the list."""
    print("[synapse][redis]: Connecting to Redis...")

    try:
        # Load REDIS_URL from environment
        REDIS_URL = os.getenv("REDIS_URL")

        if not REDIS_URL:
            raise ValueError("REDIS_URL not set in environment!")

        # Connect to Upstash Redis using TLS
        r = redis.from_url(REDIS_URL, decode_responses=True)
        r.ping()

        print("[synapse][redis]: Connected to Redis.")
        print(f"[synapse][redis]: Waiting for messages in '{REDIS_MAIN_QUEUE_NAME}'")

        while True:
            # blocking right tail pop, we left pushed to the queue
            queue, message_data = r.brpop([REDIS_MAIN_QUEUE_NAME], timeout=0)
            if message_data:
                print(f"[synapse][worker]: Received message: {message_data}")
                updated_model = processMessage(message_data, model=model, redis_client=r)
                if isinstance(updated_model, ModelManager):
                    model = updated_model

    except Exception as e:
        print(f"[synapse][redis]: unexpected exception: {e}")

    finally:
        logging.info(f"Deleting {REDIS_TRAIN_QUEUE_NAME} & {REDIS_MAIN_QUEUE_NAME}")
        r.delete(REDIS_MAIN_QUEUE_NAME)
        r.delete(REDIS_TRAIN_QUEUE_NAME)
        if 'r' in locals():
            try:
                r.ping()
                print("[synapse][redis]: Redis connection closed.")
            except:
                pass

if __name__ == '__main__':
    model = ModelManager("", "")
    start(model)