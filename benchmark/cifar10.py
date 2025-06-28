import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import time
import os
import datetime
import logging
from torch.utils.tensorboard import SummaryWriter

# Configure logging to match your application's format for consistency
logging.basicConfig(level=logging.INFO, format='[synapse][%(asctime)s](%(filename)s:%(lineno)d): %(message)s')

# --- 1. Define the simple neural network architecture ---
class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, 1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

# --- 2. Configuration for benchmarking ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

INPUT_SIZE = 3 * 32 * 32
NUM_CLASSES = 10
NUM_EPOCHS = 10
BATCH_SIZE = 1024
LEARNING_RATE = 0.001

# --- 3. Prepare CIFAR-10 Data ---
logging.info("Loading CIFAR-10 dataset...")

transform = transforms.Compose([
    transforms.ToTensor()])

data_root = "../data"
os.makedirs(data_root, exist_ok=True)
full_dataset = datasets.CIFAR10(root=data_root, download = True, train=True, transform=transform)

# Split dataset into train and test sets
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=8)

logging.info(f"Training dataset size: {len(train_dataset)} samples")
logging.info(f"Batch size: {BATCH_SIZE}")
logging.info(f"Number of batches per epoch: {len(train_loader)}")
logging.info("-" * 30)

# --- 4. Initialize Model, Loss, and Optimizer ---
model = SimpleNN(INPUT_SIZE, NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# --- 5. TensorBoard Writer and Profiler Setup ---
writer_filename = (
    f"./tbsummary/cifar10/standalone_"
    f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
)
writer = SummaryWriter(writer_filename)
logging.info(f"initialized writer {writer}")
logging.info(f"using torch: {torch.__version__}")

# Add graph (needs a dummy input of the correct shape)
dummy_input_for_graph = torch.randn(1, 3, 32, 32).to(device)
writer.add_graph(model, dummy_input_for_graph)

logging.info("Starting researcher-style training benchmark...")
start_time_total = time.time()

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(writer_filename)
) as prof:
    for epoch in range(NUM_EPOCHS):
        # --- Training Loop ---
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        epoch_start_time = time.time()

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()

            # Gradient visualization (similar to your backendTorch.py)
            current_global_step = epoch * len(train_loader) + batch_idx
            for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f"grads/{name.replace('.', '/')}", param.grad, current_global_step)
            
            # LR visualization
            writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], current_global_step)

            # Gradient norm visualization (every 25 epochs, as in your backendTorch.py)
            if (epoch + 1) % 25 == 0:
                total_gradient_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_gradient_norm += p.grad.data.norm(2).item() ** 2
                total_gradient_norm = total_gradient_norm ** 0.5
                writer.add_scalar("Gradient Norm", total_gradient_norm, current_global_step)

            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            predictions = torch.argmax(outputs, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

        # End of Training Epoch Metrics
        train_accuracy = correct_predictions / total_samples
        train_avg_loss = running_loss / total_samples

        writer.add_scalar("Loss/train", train_avg_loss, epoch)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)

        # Weights & Biases distribution (every 25 epochs, as in your backendTorch.py)
        if (epoch + 1) % 25 == 0:
            for name, param in model.named_parameters():
                writer.add_histogram(f"weights/{name.replace('.', '/')}", param, epoch)
                if param.dim() == 1:
                    writer.add_histogram(f"weights/{name.replace('.', '/')}", param, epoch)

        prof.step() # Profiler step

        # --- Validation Loop (every 25 epochs, as in your backendTorch.py) ---
        val_loss = 0.0
        val_correct_predictions = 0
        val_total_samples = 0
        if (epoch + 1) % 25 == 0:
            model.eval()
            with torch.no_grad():
                for inputs_val, labels_val in test_loader:
                    inputs_val = inputs_val.to(device)
                    labels_val = labels_val.to(device)

                    outputs_val = model(inputs_val)
                    loss_val = criterion(outputs_val, labels_val)

                    val_loss += loss_val.item() * inputs_val.size(0)
                    predictions_val = torch.argmax(outputs_val, dim=1)
                    val_correct_predictions += (predictions_val == labels_val).sum().item()
                    val_total_samples += labels_val.size(0)
            
            val_avg_loss = val_loss / val_total_samples
            val_accuracy = val_correct_predictions / val_total_samples
            writer.add_scalar("Loss/validation", val_avg_loss, epoch)
            writer.add_scalar("Accuracy/validation", val_accuracy, epoch)

        epoch_duration = time.time() - epoch_start_time
        logging.info(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        logging.info(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        logging.info(f"epoch: {epoch + 1}, loss: {train_avg_loss:.4f}, accuracy: {train_accuracy:.4f}, time: {epoch_duration:.2f}s")

# Final HParams and Metrics for TensorBoard
final_train_accuracy = train_accuracy if 'train_accuracy' in locals() else 0.0
final_train_avg_loss = train_avg_loss if 'train_avg_loss' in locals() else 0.0

hparam_dict = {
    "learning_rate": LEARNING_RATE,
    "optimizer": "Adam",
    "epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "loss_fn": "CrossEntropyLoss",
}
metric_dict = {
    "hparam/accuracy_train": final_train_accuracy,
    "hparam/loss_train": final_train_avg_loss,
}
writer.add_hparams(hparam_dict, metric_dict)
writer.close()

end_time_total = time.time()
total_duration = end_time_total - start_time_total

logging.info("-" * 30)
logging.info(f"Researcher-style benchmark complete for {NUM_EPOCHS} epochs.")
logging.info(f"Total training time: {total_duration:.2f} seconds")
logging.info(f"Average time per epoch: {total_duration / NUM_EPOCHS:.2f} seconds")