import os
import pickle

from modelManager import ModelManager

def serialize_model_manager(model_managers: list[ModelManager]):
    os.makedirs("./savefile", exist_ok=True)
    
    with open("./savefile/model_manager.bin", "wb") as file:
        pickle.dump(model_managers, file)

def deserialize_model_manager() -> list[ModelManager] :
    
    with open("./savefile/model_manager.bin", "rb") as file:
        model_managers = pickle.load(file)
        return model_managers