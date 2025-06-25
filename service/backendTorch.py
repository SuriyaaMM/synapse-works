import torch.utils.tensorboard
import torch.version
from config import logging, REDIS_TRAIN_QUEUE_NAME
from abstractManager import AbstractModelManager
from typedefs import *

import json
import redis
import datetime
import torch

from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import ToTensor

from backendTorchUtils import torch_dataset_name_map, torch_layer_name_map, \
                                torch_loss_function_name_map, torch_optimizer_name_map

class TorchLayer:
    layer: nn.Module
    id: str


"""
Implementation of AbstractManager for pytorch backend
"""
class TorchModelManager(AbstractModelManager):
    def __init__(self):
        super().__init__()
        self.layers: list[TorchLayer] = []

    def appendLayer(self, layer_config: LayerConfig, debug: bool = True):
        if(debug):
            try:
                layer = torch_layer_name_map(layer_config["type"])(**layer_config["kwargs"])
            # ----- exceptions
            except TypeError as e:
                logging.error(f"type error {e}")
        else:
            layer = torch_layer_name_map(layer_config["type"])(**layer_config["kwargs"])

        torch_layer = TorchLayer()
        torch_layer.layer = layer
        torch_layer.id = layer_config["id"]
        # append to existing layers
        self.layers.append(torch_layer)

    def deleteLayer(self, layer_id: str, debug: bool = True):
        for layer in self.layers:
            if layer.id == layer_id:
                self.layers.remove(layer)
                logging.info(f"Deleted layer {layer.__class__.__name__} with id({layer_id})")

    def setDatasetConfig(self, dataset_config: DatasetConfig, debug: bool = True):
        self.dataset_config = dataset_config
        logging.info(f"Set DatasetConfig {json.dumps(dataset_config, indent=4, default=custom_json_encoder)}")
        
    def setTrainConfig(self, train_config: TrainConfig,  debug: bool = True):
        self.train_config: TrainConfig = train_config
        logging.info(f"Set TrainConfig {json.dumps(train_config, indent=4)}")

class TorchTrainManager(nn.Module):
    
    def __init__(self, 
                 layers: list[nn.Module],
                 train_config: TrainConfig,
                 dataset_config: DatasetConfig,
                 debug: bool = True):
        # initialize nn.Module
        super().__init__()
        self.layers = layers
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.neuralNet = nn.Sequential(*layers).to(self.device)

        self.train_config = train_config
        self.dataset_config = dataset_config

        optimizerType = torch_optimizer_name_map(train_config["optimizer"])
        
        self.optimizer = optimizerType(self.neuralNet.parameters(), **train_config["optimizer_kwargs"]) # type:ignore
        self.loss_function = torch_loss_function_name_map(train_config["loss_function"], debug)()
        self.epochs = train_config["epochs"]

        self.dataset = torch_dataset_name_map(dataset_config["name"], debug)(**dataset_config["kwargs"]) # type:ignore
        # split datasets
        self.train_dataset, self.test_dataset = random_split(self.dataset, dataset_config["split_length"])
        # initialize dataloaders
        self.train_loader : DataLoader = DataLoader(self.train_dataset, **dataset_config["dataloader_config"])
        self.test_loader: DataLoader = DataLoader(self.test_dataset, **dataset_config["dataloader_config"])
        # dummy tensor for computation graph visualization
        self.dummy_tensor_for_computation_graph = next(iter(self.train_loader))[0]
        self.dummy_tensor_for_computation_graph = self.dummy_tensor_for_computation_graph.to(self.device)

    def forward(self, x: torch.Tensor):
        # TODO(mms) hardcoded flattening here
        #x = x.view(x.shape[0], -1)
        return self.neuralNet(x)

# refer: https://github.com/SuriyaaMM/dl-analysis/blob/main/analysis/regularization/train.py
def _TrainEpoch(train_manager: TorchTrainManager,
                current_epoch: int, 
                writer: torch.utils.tensorboard.SummaryWriter) -> tuple[float, float, int]:
    train_manager.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    features : torch.Tensor
    labels: torch.Tensor
    for batch_idx, (features, labels) in enumerate(train_manager.train_loader):
        features = features.to(train_manager.device)
        labels = labels.to(train_manager.device)
        output: torch.Tensor = train_manager.forward(features)
        loss: torch.Tensor = train_manager.loss_function(output, labels)
        train_manager.optimizer.zero_grad()
        loss.backward()

        # gradient visualization
        current_global_step = current_epoch * len(train_manager.train_loader) + batch_idx
        for name, param in train_manager.neuralNet.named_parameters():
            if param.grad is not None:
                writer.add_histogram(f"grads/{name.replace('.', '/')}", param.grad, current_global_step)
        
        # lr visualization
        writer.add_scalar("Learning Rate", train_manager.optimizer.param_groups[0]['lr'], current_global_step)

        # gradient norm visualization
        total_gradient_norm = 0.0
        for p in train_manager.neuralNet.parameters():
            if p.grad is not None:
                total_gradient_norm += p.grad.data.norm(2).item() ** 2
        
        total_gradient_norm = total_gradient_norm ** 0.5
        writer.add_scalar("Gradient Norm", total_gradient_norm, current_global_step)

        train_manager.optimizer.step()
        running_loss += loss.item() * features.size(0)
        predictions = torch.argmax(output, dim = 1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += labels.size(0)

    return running_loss, correct_predictions, total_samples

def _ValidateEpoch(train_manager: TorchTrainManager) -> tuple[float, float, int]:
    train_manager.neuralNet.eval()
    correct_predictions = 0
    total_samples = 0
    running_loss = 0

    features : torch.Tensor
    labels: torch.Tensor

    with torch.no_grad():
        for features, labels in train_manager.test_loader:
            features = features.to(train_manager.device)
            labels = labels.to(train_manager.device)
            output: torch.Tensor = train_manager.forward(features)
            loss: torch.Tensor = train_manager.loss_function(output, labels)
            running_loss += loss.item() * features.size(0)
            predictions = torch.argmax(output, dim = 1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    return running_loss, correct_predictions, total_samples

def train(train_manager: TorchTrainManager, redis_client: redis.Redis):
    writer = torch.utils.tensorboard.SummaryWriter("./tbsummary")
    logging.info(f"initialized writer {writer}")
    logging.info(f"using device {train_manager.device}")
    logging.info(f"using torch: {torch.__version__}")

    writer.add_graph(train_manager.neuralNet, train_manager.dummy_tensor_for_computation_graph)

    for epoch in range(train_manager.epochs):
        running_loss, correct_predictions, total_samples = _TrainEpoch(train_manager, epoch, writer)
        accuracy = correct_predictions/total_samples
        # write to tensorboard
        writer.add_scalar("Loss/train", running_loss, epoch)
        writer.add_scalar("Accuracy/train", accuracy, epoch)

        # weights & biases distribution
        for name, param in train_manager.neuralNet.named_parameters():
            # log weights
            writer.add_histogram(f"weights/{name.replace('.', '/')}", param, epoch)
            # log params if they exist
            if param.dim() == 1:
                writer.add_histogram(f"weights/{name.replace('.', '/')}", param, epoch)
        
        logging.info(f"epoch: {epoch + 1}, loss: {running_loss}, accuracy: {accuracy}")
        
        # TODO(mms) hardcoded validation period here to 5
        if epoch % 5 == 0:
            running_val_loss, correct_val_predictions, total_val_samples = _ValidateEpoch(train_manager)
            accuracy_val = correct_val_predictions/total_val_samples
            writer.add_scalar("Loss/validation", running_val_loss, epoch)
            writer.add_scalar("Accuracy/validation", accuracy_val, epoch)

        update_message = {
            "epoch" : epoch + 1,
            "loss" : running_loss,
            "accuracy" : (correct_predictions/total_samples),
            "completed" : epoch == train_manager.epochs - 1,
            "timestamp" : datetime.datetime.now().isoformat()
        }
        redis_client.lpush(REDIS_TRAIN_QUEUE_NAME, json.dumps(update_message))
        logging.info(f"pushed {json.dumps(update_message)} to redis queue")

    batch_size = 1
    if "batch_size" in train_manager.dataset_config.keys():
        batch_size = train_manager.dataset_config["dataloader_config"]["batch_size"] # type:ignore

    hparam_dict = {
        "learning_rate": train_manager.train_config["optimizer_kwargs"]["lr"],
        "optimizer": train_manager.train_config["optimizer"],
        "epochs": train_manager.epochs,
        "batch_size": batch_size,
        "loss_fn": train_manager.train_config["loss_function"],
    }
   
    metric_dict = {
        "hparam/accuracy_train": accuracy, 
        "hparam/loss_train": running_loss,
    }
    writer.add_hparams(hparam_dict, metric_dict)    
    writer.close()