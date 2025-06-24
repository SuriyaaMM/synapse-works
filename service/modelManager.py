from config import logging
from typedefs import *
import json
import redis

from backendTorch import TorchModelManager, TorchTrainManager, train, nn

class ModelManager(object):

    def __init__(self, model_id: str, model_name: str, backend: str = "torch"):
        R"""ModelManager handles model creation, deletion & modification

        Args:
            model_id: str, graphql object modelId
            model_name: str, graphql object modelName
            backend: str, graphql object backend
        """
        self.debug = True
        self.id = model_id
        self.name = model_name
        self.backend = backend

        if backend == "torch":
            self._internal_manager = TorchModelManager()
        else:
            raise NotImplementedError(f"{backend} is not implemented yet")
        
        logging.info(f"ModelManager (name = {self.name}) initialized with (id = {self.id})")

    def dumpNetworkForTraining(self):
        if(self.backend == "torch"):
<<<<<<< HEAD
            self._train_manager = TorchTrainManager(self._internal_manager.layers,
=======
            layers: list[nn.Module] = []
            for torch_layer in self._internal_manager.layers:
                layers.append(torch_layer.layer)

            self._train_manager = TorchTrainManager(layers,
>>>>>>> eadd141d39de380594e338dabde6ad5b6ea6a2b0
                                                   self._internal_manager.train_config,
                                                   self._internal_manager.dataset_config)
        else:
            raise NotImplementedError(f"training using {self.backend} is not implemented yet")

    def appendLayer(self, layer_config: LayerConfig):
        R"""Appends the layer to model
        
        Args:
            layer_name: str, graphql object layerName
        """
        self._internal_manager.appendLayer(layer_config=layer_config)
        logging.info(f"Appended layer({layer_config['type']} to model(id = {self.id}) with kwargs\n{json.dumps(layer_config["kwargs"], indent=4)})")
    
    def deleteLayer(self, layer_id: str):
        R"""Deletes the layer in model

        Args:
            layer_name: str, graphql object layerName
        """
<<<<<<< HEAD
        self._internal_manager.deleteLayer(layer_config=layer_config)
        logging.info(f"Deleted layer({layer_config['type']} from model(id = {self.id}) with kwargs\n{json.dumps(layer_config["kwargs"], indent=4)})")
=======
        self._internal_manager.deleteLayer(layer_id=layer_id)
        
>>>>>>> eadd141d39de380594e338dabde6ad5b6ea6a2b0

    def setTrainingConfig(self, train_config_td: TrainConfig):
        R"""Sets training configuration for the model

        Args:
            train_config_td: TrainConfig graphql object
        """
        self._internal_manager.setTrainConfig(train_config=train_config_td, debug=self.debug)

    def setDatasetConfig(self, dataset_config_td: DatasetConfig):
        R"""Sets dataset configuration for the model

        Args:
            dataset_config_td: DatasetConfig graphql object
        """
        self._internal_manager.setDatasetConfig(dataset_config=dataset_config_td, debug=self.debug)
        logging.info(f"Set DatasetConfig to Model({self.id}) with config: {json.dumps(dataset_config_td['kwargs'], indent=4, default=custom_json_encoder)}")
    
    def train(self, redis_client: redis.Redis):
        logging.info("training model started!")
        self.dumpNetworkForTraining()
        logging.info("sucessfully dumped neural network")
        train(self._train_manager, redis_client=redis_client)