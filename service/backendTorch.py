from config import logging
from abstractManager import AbstractModelManager
from typedefs import *

import torch

from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

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

    def appendLayer(self, layer_name: str, kwargs: LayerKwargs, debug: bool = True):
        if(debug):
            try:
                layer = torch_layer_name_map(layer_name)(**kwargs)
            # ----- exceptions
            except TypeError as e:
                logging.error(f"type error {e}")
        else:
            layer = torch_layer_name_map(layer_name)(**kwargs)

        # append to existing layers
        self.layers.append(layer)

    def deleteLayer(self, layer_name: str, kwargs: LayerKwargs, debug: bool = True):
        if(debug):
            try:
                layer = torch_layer_name_map(layer_name)(**kwargs)
            # ----- exceptions
            except TypeError as e:
                logging.error(f"type error {e}")
        else:
            layer = torch_layer_name_map(layer_name)(**kwargs)

        # search in layers & remove it
        if layer in self.layers:
            self.layers.remove(layer)
            logging.info(f"removed layer {layer}")

    def setDatasetConfig(self, dataset_config_td: DatasetConfig, debug: bool = True):
        # get configurations
        self.datasetName = dataset_config_td["name"]
        self.splitOptions : list = dataset_config_td["split_length"]
        self.shuffle = dataset_config_td["shuffle"]
        # configure torch for specified
        self.dataset = torch_dataset_name_map(self.datasetName, debug)(dataset_config_td["kwargs"]) # type:ignore
        self.trainDataset, self.testDataset = random_split(self.dataset, lengths=self.splitOptions) # type:ignore

        logging.info(f"""Set Dataset Configuration with
                        
                        name: {self.datasetName}
                        split : ({self.splitOptions[0] * 100} % train) & ({self.splitOptions[1] * 100} % test)
                        shuffle: {self.shuffle}""")

    def setTrainConfig(self, train_config_td: TrainConfig,  debug: bool = True):
        # get configurations
        self.optimizerConfig = train_config_td["optimizerConfig"]
        self.lossFunctionName = train_config_td["loss_function"]
        self.epochs = train_config_td["epochs"]
        self.batchSize = train_config_td["batch_size"]
        # initialize train & test loaders
        #self.trainLoader = DataLoader(self.trainDataset, batch_size=self.batchSize, shuffle=self.shuffle)
        #self.testLoader = DataLoader(self.testDataset, batch_size=self.batchSize, shuffle=self.shuffle)

        logging.info(f"""Set Training Configuration with
                        
                        epochs: {self.epochs}
                        batch size: {self.batchSize}""")

class TorchTrainManager(nn.Module):
    
    def __init__(self, layers: list[nn.Module], optimizer_config: OptimizerConfig, loss_function: str, debug: bool = True):
        self.layers = layers
        self.neuralNet = nn.Sequential(*layers)

        optimizerType = torch_optimizer_name_map(optimizer_config["name"])

        optimizerKwargs = {
            "lr" : optimizer_config["lr"]
        }

        self.optimizer = optimizerType(self.neuralNet.parameters(), optimizerKwargs)
        self.lossFunction = torch_loss_function_name_map(loss_function, debug)()

        logging.info(f"""Initialized TorchTrainManager with parameters

                        number of layers: {len(self.layers)}
                        optimizer : {optimizerType}
                        loss function : {self.lossFunction}({loss_function})""")

    def forward(self, x: torch.Tensor):
        return self.neuralNet(x)


            

