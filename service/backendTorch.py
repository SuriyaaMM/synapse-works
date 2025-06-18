from config import logging
from abstractManager import AbstractModelManager
from typedefs import *

import torch

from torch import nn
from torch.utils import data as tdu

from backendTorchUtils import torch_dataset_name_map, torch_layer_name_map, \
                                torch_loss_function_name_map, torch_optimizer_name_map

"""
Implementation of AbstractManager for pytorch backend
"""
class TorchModelManager(AbstractModelManager):
    # TODO(mms): default trainConfig get's passed when the model is created, handle that!
    def __init__(self):
        super().__init__()
        self.layers = []

    def appendLayer(self, layer_config: LayerConfig, debug: bool = True):
        if(debug):
            try:
                layer = torch_layer_name_map(layer_config["name"])(**layer_config["kwargs"])
            # ----- exceptions
            except TypeError as e:
                logging.error(f"type error {e}")
        else:
            layer = torch_layer_name_map(layer_config["name"])(**layer_config["kwargs"])

        # append to existing layers
        self.layers.append(layer)

    def deleteLayer(self, layer_config: LayerConfig, debug: bool = True):
        if(debug):
            try:
                layer = torch_layer_name_map(layer_config["name"])(layer_config["kwargs"])
            # ----- exceptions
            except TypeError as e:
                logging.error(f"type error {e}")
        else:
            layer = torch_layer_name_map(layer_config["name"])(layer_config["kwargs"])

        # search in layers & remove it
        if layer in self.layers:
            self.layers.remove(layer)
            logging.info(f"removed layer {layer}")

    def setDatasetConfig(self, dataset_config_td: Dataset, debug: bool = True):
        # get configurations
        self.datasetConfig = dataset_config_td
        # configure torch for specified
        self.dataset = torch_dataset_name_map(self.datasetConfig["name"], debug)(**dataset_config_td["kwargs"]) # type:ignore
        self.trainDataset, self.testDataset = tdu.random_split(self.dataset, self.datasetConfig["split_length"]) # type:ignore
        logging.info(f"dataset state: {self.dataset.__str__}")
        logging.info(f"train dataset & test dataset: {self.trainDataset, self.testDataset}")
        
    def setTrainConfig(self, train_config_td: TrainConfig,  debug: bool = True):
        # get configurations
        self.trainConfig: TrainConfig = train_config_td
        
        logging.info(fR"""Set Training Configuration with
                        
        epochs: {self.trainConfig["epochs"]}
        batch size: {self.trainConfig["batch_size"]}""")

class TorchTrainManager(nn.Module):
    
    def __init__(self, 
                 layers: list[nn.Module],
                 train_config: TrainConfig, 
                 debug: bool = True):
        super().__init__()
        self.layers = layers
        self.neuralNet = nn.Sequential(*layers)

        optimizerType = torch_optimizer_name_map(train_config["optimizer"])

        self.optimizer = optimizerType(self.neuralNet.parameters(), **train_config["optimizer_kwargs"]) # type:ignore
        self.lossFunction = torch_loss_function_name_map(train_config["loss_function"], debug)()

        logging.info(f"""Initialized TorchTrainManager with parameters

        number of layers: {len(self.layers)}
        optimizer : {optimizerType}\n{self.optimizer.state_dict()}\n
        loss function : {self.lossFunction})""")

    def forward(self, x: torch.Tensor):
        return self.neuralNet(x)


            

