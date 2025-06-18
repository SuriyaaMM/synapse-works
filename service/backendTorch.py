from config import logging
from abstractManager import AbstractModelManager
from typedefs import *

import torch

from torch import nn
from torch.utils import data as tdu
from torchvision.transforms import ToTensor

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
        # set configurations
        self.datasetConfig = dataset_config_td
        logging.info(f"Dataset ({self.datasetConfig["name"]}) is configured")
        
    def setTrainConfig(self, train_config_td: TrainConfig,  debug: bool = True):
        # set configurations
        self.trainConfig: TrainConfig = train_config_td
        logging.info(f"epochs: {self.trainConfig["epochs"]} \
                     batch size: {self.trainConfig["batch_size"]}")

class TorchTrainManager(nn.Module):
    
    def __init__(self, 
                 layers: list[nn.Module],
                 train_config: TrainConfig,
                 dataset: Dataset,
                 debug: bool = True):
        super().__init__()
        self.layers = layers
        self.neuralNet = nn.Sequential(*layers)

        optimizerType = torch_optimizer_name_map(train_config["optimizer"])

        self.optimizer = optimizerType(self.neuralNet.parameters(), **train_config["optimizer_kwargs"]) # type:ignore
        logging.info(f"Set Optimizer {self.optimizer.state_dict()}")
        self.lossFunction = torch_loss_function_name_map(train_config["loss_function"], debug)()
        logging.info(f"Set LossFunction {self.lossFunction.state_dict()}")
        self.epochs = train_config["epochs"]
        self.torchDataset = torch_dataset_name_map(dataset["name"], debug)(transform=ToTensor(), **dataset["kwargs"]) # type:ignore
        logging.info(f"Set dataset {self.torchDataset.__str__}")
        logging.info(f"Split Length = {dataset["split_length"]}")
        self.trainDataset, self.testDataset = tdu.random_split(self.torchDataset, dataset["split_length"]) # type:ignore
        self.trainLoader : tdu.DataLoader = tdu.DataLoader(self.trainDataset, train_config["batch_size"])
        logging.info(f"Train Loader configured, {self.trainLoader.__str__}")
        self.testLoader : tdu.DataLoader = tdu.DataLoader(self.testDataset, batch_size=train_config["batch_size"])
        logging.info(f"Test Loader configured, {self.trainLoader.__str__}")

    def forward(self, x: torch.Tensor):
        x = x.view(x.shape[0], -1)
        return self.neuralNet(x)

def _TrainEpoch(trainManager: TorchTrainManager) -> tuple[float, float, int]:
    # set to train mode
    trainManager.train()
    runningLoss = 0.0
    correctPredictions = 0
    totalSamples = 0
    # refer: https://github.com/SuriyaaMM/dl-analysis/blob/main/analysis/regularization/train.py
    features : torch.Tensor
    labels: torch.Tensor
    for features, labels in trainManager.trainLoader:

        output: torch.Tensor = trainManager.forward(features)
        loss: torch.Tensor = trainManager.lossFunction(output, labels)
        trainManager.optimizer.zero_grad()
        loss.backward()
        trainManager.optimizer.step()
        runningLoss += loss.item() * features.size(0)
        predictions = torch.argmax(output, dim = 1)
        correctPredictions += (predictions == labels).sum().item()
        totalSamples += labels.size(0)

    return runningLoss, correctPredictions, totalSamples


def train(train_manager: TorchTrainManager):

    for epochs in range(train_manager.epochs):
        runningLoss, correctPredictions, totalSamples = _TrainEpoch(trainManager= train_manager)
        logging.info(f"epoch: {epochs + 1}, loss: {runningLoss}")
