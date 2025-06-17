from config import logging
from typedefs import *

from backendTorch import TorchModelManager, TorchTrainManager

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
            self._InternalManager = TorchModelManager()
        else:
            raise NotImplementedError(f"{backend} is not implemented yet")
        
        logging.info(f"ModelManager (name = {self.name})initialized with (id = {self.id})")

    def dumpNetworkForTraining(self):
        if(self.backend == "torch"):
            self._TrainManager = TorchTrainManager(self._InternalManager.layers,
                                                   self._InternalManager.trainConfig,
                                                   self.debug)
        else:
            raise NotImplementedError(f"training using {self.backend} is not implemented yet")

    def appendLayer(self, layer_config: LayerConfig):
        R"""Appends the layer to model
        
        Args:
            layer_name: str, graphql object layerName
        """
        self._InternalManager.appendLayer(layer_config=layer_config)
        logging.info(f"appended layer({layer_config['name']} to model(id = {self.id}) with kwargs\n{layer_config["kwargs"]})")
    
    def deleteLayer(self, layer_config: LayerConfig):
        R"""Deletes the layer in model

        Args:
            layer_name: str, graphql object layerName
        """
        self._InternalManager.deleteLayer(layer_config=layer_config)

    def setTrainingConfig(self, train_config_td: TrainConfig):
        R"""Sets training configuration for the model

        Args:
            train_config_td: TrainConfig graphql object
        """
        self._InternalManager.setTrainConfig(train_config_td=train_config_td, debug=self.debug)

    def setDatasetConfig(self, dataset_config_td: DatasetConfig):
        R"""Sets dataset configuration for the model

        Args:
            dataset_config_td: DatasetConfig graphql object
        """
        self._InternalManager.setDatasetConfig(dataset_config_td=dataset_config_td, debug=self.debug)
