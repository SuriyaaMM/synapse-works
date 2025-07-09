import redis
import os
import argparse
import json

from config import logging, REDIS_HOST, REDIS_PORT, REDIS_MAIN_QUEUE_NAME, REDIS_TRAIN_QUEUE_NAME
from modelManager import ModelManager
from workerUtils import processMessage


def start(model: ModelManager, args: argparse.Namespace):
    """Connects to Redis and starts consuming messages from the list."""
    logging.info("Connecting to Redis")
    
    with open("environment.json", "r") as file:
        env = json.load(file)

    try:
        if args.remote:
            # Load REDIS_URL from environment
            REDIS_URL = env["REDIS_URL"]

            # Connect to Upstash Redis using TLS
            r = redis.from_url(REDIS_URL, decode_responses=True)
            r.ping()
            logging.info(f"Connected to Redis {REDIS_URL}")
        else:
            # Connect to Redis Localhost
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
            r.ping()
            logging.info(f"Connected to Redis {REDIS_HOST}:{REDIS_PORT}")

        while True:
            # blocking right tail pop, we left pushed to the queue
            queue, message_data = r.brpop([REDIS_MAIN_QUEUE_NAME], timeout=0) # type:ignore
            if message_data:
                print(f"[synapse][worker]: Received message: {message_data}")
                updated_model = processMessage(message_data, model=model, redis_client=r)
                if isinstance(updated_model, ModelManager):
                    model = updated_model

    except Exception as e:
        logging.info(f"unexpected exception: {e}")

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

    parser = argparse.ArgumentParser()

    parser.add_argument("--remote", action = "store_true", help="Connects server to remove URL")
    args = parser.parse_args()

    model = ModelManager("", "")
    start(model, args)