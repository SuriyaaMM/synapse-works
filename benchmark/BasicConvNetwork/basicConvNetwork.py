# benchmark_like_app.py

import os
import json
import torch
import datetime
import logging
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, ProfilerActivity, record_function

# ----------------- Logger -----------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------- Model ------------------
class ConvMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 10)
        )

    def forward(self, x):
        return self.net(x)

# ----------------- Train Loop ------------------
def train_one_epoch(model, loader, optimizer, loss_fn, writer, device, epoch, config):
    # set model to train mode
    model.train()
    # metric variables
    total_loss, correct, total = 0.0, 0, 0

    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        # calculate metrics
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(dim=1) == y).sum().item()
        total += x.size(0)

        global_step = epoch * len(loader) + step

        # log gradients on right period
        if config['log_gradients'] and (epoch + 1) % config['log_period'] == 0:
            for name, p in model.named_parameters():
                if p.grad is not None:
                    writer.add_histogram(f"grads/{name}", p.grad, global_step)

        # log learning rate on right period
        if config['log_lr'] and (epoch + 1) % config['log_period'] == 0:
            writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], global_step)

        # log gradient norm on righ period
        if config['log_grad_norm'] and (epoch + 1) % config['log_period'] == 0:
            total_norm = sum((p.grad.norm(2) ** 2 for p in model.parameters() if p.grad is not None)) ** 0.5
            writer.add_scalar("GradientNorm", total_norm, global_step)

    # write metrics to tensorboard
    avg_loss = total_loss / total
    acc = correct / total
    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Accuracy/train", acc, epoch)
    logging.info(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")

# ----------------- Validation ------------------
def validate(model, loader, loss_fn, writer, device, epoch):
    # set model to evaluation
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            # calculate metrics
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(dim=1) == y).sum().item()
            total += x.size(0)
    
    # write metrics to tensorboard
    avg_loss = total_loss / total
    acc = correct / total
    writer.add_scalar("Loss/val", avg_loss, epoch)
    writer.add_scalar("Accuracy/val", acc, epoch)
    logging.info(f"[Epoch {epoch+1}] Val Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")

# ----------------- Main ------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # hyper parameters
    batch_size = 1024
    epochs = 50
    lr = 1e-4
    weight_decay = 1e-2
    
    config = {
        "log_gradients": True,
        "log_grad_norm": True,
        "log_weights": True,
        "log_lr": True,
        "log_period": 25,
        "profile": True,
    }

    # Data
    transform = transforms.ToTensor()
    dataset = datasets.MNIST("./data", train=True, transform=transform, download=True)
    # split into 70% 30%
    train_set, val_set = random_split(dataset, [0.7, 0.3])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # Model
    model = ConvMNIST().to(device)
    # initialize adamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    log_dir = f"./tbsummary/benchmark_like_app/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    writer = SummaryWriter(log_dir)
    writer.add_graph(model, torch.randn(1, 1, 28, 28).to(device))
    
    # initialize profiler while training
    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir)) as prof:
            for epoch in range(epochs):
                with record_function("epoch"):
                    # validate on right period
                    if (epoch + 1) % 10 == 0:
                        validate(model, val_loader, loss_fn, writer, device, epoch)
                    
                    train_one_epoch(model, train_loader, optimizer, loss_fn, writer, device, epoch, config)
                    prof.step()
    # cleanup tensorboard writer
    writer.close()
    torch.save(model.state_dict(), f"savefile/model/benchmark_conv_mnist.pt")
    logging.info("Model saved to savefile/model/benchmark_conv_mnist.pt")

if __name__ == "__main__":
    main()
