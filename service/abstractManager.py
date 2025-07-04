from abc import ABC, abstractmethod
from typedefs import *

R"""
Abstract base class for implementation of backend managers
"""
class AbstractModelManager(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def appendLayer(self, layer_config: LayerConfig, debug: bool = True):
        pass

    @abstractmethod
    def deleteLayer(self, layer_id: str, debug: bool = True):
        pass

    @abstractmethod
    def modifyLayer(self, layer_config: LayerConfig, debug: bool = True):
        pass

    @abstractmethod
    def setDatasetConfig(self, dataset_config: DatasetConfig, debug: bool = True):
        pass
    
    @abstractmethod
    def setTrainConfig(self, train_config: TrainConfig, debug: bool = True):
        pass
    
    @abstractmethod
    def setModule(self, module: torch.nn.Module):
        pass