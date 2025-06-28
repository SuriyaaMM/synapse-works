from config import logging

import os
import pickle

from modelManager import ModelManager

def serialize_model_manager(model_manager: ModelManager):
    os.makedirs("./savefile", exist_ok=True)
    if(model_manager.id == ""):
        logging.error("model_manager.id is empty, create model first!")
    else:
        with open(f"./savefile/{model_manager.id}.bin", "wb") as file:
            pickle.dump(model_manager, file)

def deserialize_model_manager(model_id: str) -> ModelManager :
    
    with open(f"./savefile/{model_id}.bin", "rb") as file:
        model_managers = pickle.load(file)
        return model_managers