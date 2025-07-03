import os
import torch
import logging
import datetime
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, ProfilerActivity, record_function
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Configure logging for better visibility during training
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LinearMNISTModel(nn.Module):
    """
    A simple neural network for MNIST classification using only linear layers,
    as per the requested architecture.
    Note: For better performance on image data, non-linear activation functions
    (like ReLU) are typically added between linear layers. This architecture
    is kept strictly linear as per the user's explicit request.
    """
    def __init__(self):
        super().__init__()
        # Flattens the 28x28 image into a 784-element vector (28 * 28)
        self.flatten = nn.Flatten()
        # First linear layer: 784 input features to 1024 hidden features
        self.fc1 = nn.Linear(784, 1024)
        # Second linear layer: 1024 hidden features to 2048 hidden features
        self.fc2 = nn.Linear(1024, 2048)
        # Output layer: 2048 hidden features to 10 output classes (digits 0-9)
        self.out = nn.Linear(2048, 10)

    def forward(self, x):
        # Apply flatten
        x = self.flatten(x)
        # Apply first linear layer
        x = self.fc1(x)
        # Apply second linear layer
        x = self.fc2(x)
        # Apply final output layer
        x = self.out(x)
        return x

def train_epoch(model, train_loader, device, loss_fn, optimizer, current_epoch, writer, log_frequency_epoch):
    """
    Performs a single training epoch.
    Calculates loss, performs backpropagation, and updates model weights.
    Logs metrics to TensorBoard.
    `log_frequency_epoch` is used to control frequency of some logs.
    """
    model.train() # Ensure model is in training mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch_idx, (features, labels) in enumerate(train_loader):
        features = features.to(device) # Move features to device (CPU/GPU)
        labels = labels.to(device)     # Move labels to device

        # Forward pass: compute model output
        output = model(features)
        # Calculate loss
        loss = loss_fn(output, labels)

        # Zero the gradients before backpropagation
        optimizer.zero_grad()
        # Backpropagation: compute gradients
        loss.backward()

        # Calculate current global step for TensorBoard logging
        current_global_step = current_epoch * len(train_loader) + batch_idx

        # --- Reverted: Log gradients for every batch, as per the second script ---
        for name, param in model.named_parameters():
            if param.grad is not None:
                writer.add_histogram(f"grads/{name.replace('.', '/')}", param.grad, current_global_step)

        # Log learning rate to TensorBoard
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], current_global_step)

        # Log total gradient norm (still controlled by epoch frequency, as per second script's example of 25 epochs)
        if (current_epoch + 1) % log_frequency_epoch == 0 and batch_idx == 0: # Log once per relevant epoch start
            total_gradient_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_gradient_norm += p.grad.data.norm(2).item() ** 2
            total_gradient_norm = total_gradient_norm ** 0.5
            writer.add_scalar("Gradient Norm", total_gradient_norm, current_global_step)

        # Optimizer step: update model weights
        optimizer.step()

        # Accumulate loss and correct predictions
        running_loss += loss.item() * features.size(0)
        predictions = torch.argmax(output, dim=1) # Get the predicted class
        correct_predictions += (predictions == labels).sum().item()
        total_samples += labels.size(0)

    # Return epoch-level metrics
    return running_loss, correct_predictions, total_samples

def validate_epoch(model, test_loader, device, loss_fn):
    """
    Performs a single validation epoch.
    Evaluates the model on the test set without updating weights.
    """
    model.eval() # Set model to evaluation mode
    correct_predictions = 0
    total_samples = 0
    running_loss = 0

    with torch.no_grad(): # Disable gradient calculation during validation
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            output = model(features)
            loss = loss_fn(output, labels)
            running_loss += loss.item() * features.size(0)
            predictions = torch.argmax(output, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    # Return validation metrics
    return running_loss, correct_predictions, total_samples

def run_training(model, train_loader, test_loader, device, loss_fn, optimizer, epochs, model_id="linear-mnist", dataset_name="MNIST", export_to=None):
    """
    Orchestrates the entire training and validation process.
    Handles TensorBoard setup, profiling, and model saving.
    """
    # Create a unique TensorBoard log directory
    writer_filename = (
        f"./tbsummary/StandAloneScript_{dataset_name}/"
        f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    writer = SummaryWriter(writer_filename)
    logging.info(f"TensorBoard logs will be saved to: {writer_filename}")

    # Dummy tensor for tracing the computation graph in TensorBoard
    dummy_tensor_for_computation_graph = torch.randn(1, 1, 28, 28).to(device)
    # Add the model's computation graph to TensorBoard
    writer.add_graph(model, dummy_tensor_for_computation_graph)

    # --- Profiler remains as before, but with record_shapes=False by default for less overhead ---
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(writer_filename)) as prof:
        with record_function("model_training"): # Label for the profiling section
            # Frequency for logging weight histograms and performing validation
            # The second script uses 25 for these checks.
            LOG_EPOCH_HIST_AND_VAL_FREQ = 25

            for epoch in range(epochs):
                # Train for one epoch
                train_loss, train_correct, train_total = train_epoch(
                    model, train_loader, device, loss_fn, optimizer, epoch, writer,
                    LOG_EPOCH_HIST_AND_VAL_FREQ # Pass the epoch frequency for relevant logs
                )
                train_accuracy = train_correct / train_total

                # Log training loss and accuracy to TensorBoard
                writer.add_scalar("Loss/train", train_loss, epoch)
                writer.add_scalar("Accuracy/train", train_accuracy, epoch)

                # Log weights histograms and perform validation as per the second script's frequency (every 25 epochs)
                if (epoch + 1) % LOG_EPOCH_HIST_AND_VAL_FREQ == 0:
                    for name, param in model.named_parameters():
                        writer.add_histogram(f"weights/{name.replace('.', '/')}", param, epoch)

                    # Perform validation
                    val_loss, val_correct, val_total = validate_epoch(model, test_loader, device, loss_fn)
                    val_accuracy = val_correct / val_total
                    # Log validation loss and accuracy to TensorBoard
                    writer.add_scalar("Loss/validation", val_loss, epoch)
                    writer.add_scalar("Accuracy/validation", val_accuracy, epoch)
                    logging.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

                # Step the profiler to record metrics for the current epoch
                prof.step()
                logging.info(f"Epoch: {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

    # Save profiler results to a file
    os.makedirs(".profile", exist_ok=True)
    profile_save_path = f".profile/TorchProfile_StandAloneScript_{dataset_name}_{model_id}.txt"
    with open(profile_save_path, "w") as f:
        f.write(prof.key_averages().table(sort_by="cpu_time_total"))
    logging.info(f"Profiler results saved to {profile_save_path}")

    # Log hyperparameters and final metrics to TensorBoard's HParams tab
    hparam_dict = {
        "learning_rate": optimizer.param_groups[0]['lr'],
        "optimizer": type(optimizer).__name__,
        "epochs": epochs,
        "batch_size": train_loader.batch_size,
        "loss_fn": type(loss_fn).__name__,
    }
    # Note: Accuracy and loss here reflect the *last* epoch's training metrics.
    # For a more comprehensive HParam summary, you might average over epochs or use best metrics.
    metric_dict = {
        "hparam/accuracy_train": train_accuracy,
        "hparam/loss_train": train_loss
    }
    writer.add_hparams(hparam_dict, metric_dict)
    writer.close()
    logging.info("TensorBoard summary writer closed.")

    # Save the trained model's state dictionary if export_to is "TorchTensor"
    os.makedirs("savefile/model", exist_ok=True)
    if export_to == "TorchTensor":
        model_save_path = f"savefile/model/StandAloneScript_{model_id}.pt"
        torch.save(model.state_dict(), model_save_path)
        logging.info(f"Model state dictionary saved to {model_save_path}")

def main():
    """
    Main function to set up and run the MNIST training.
    This function orchestrates the entire training pipeline.
    """
    # Determine the device to use (CUDA if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Define transformations for the MNIST dataset
    # transforms.ToTensor() converts PIL Image to Tensor and scales pixel values to [0, 1]
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Load MNIST training and test datasets
    # root="./data" ensures data is downloaded to a 'data' directory in the current working directory
    logging.info("Downloading MNIST datasets...")
    train_set = datasets.MNIST(root="./benchmark/data", train=True, transform=transform)
    test_set = datasets.MNIST(root="./benchmark/data", train=False, transform=transform)
    logging.info("MNIST datasets downloaded.")

    # Define batch size for data loaders
    batch_size = 1024

    # Create data loaders for batching and shuffling
    # Added num_workers and pin_memory for potential performance improvements
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                              num_workers=8, pin_memory=True if device.type == 'cuda' else False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, 
                             num_workers=8, pin_memory=True if device.type == 'cuda' else False)

    # Instantiate the model, loss function, and optimizer
    model = LinearMNISTModel()
    model.to(device) # Move model to the selected device
    loss_fn = nn.CrossEntropyLoss() # Suitable for multi-class classification
    learning_rate = 1e-4 # Learning rate for the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Adam optimizer
    epochs = 50 # Number of training epochs

    logging.info("Starting model training...")
    # Run the training process with all parameters passed directly
    run_training(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=epochs,
        model_id="linear-mnist",        # Identifier for the model
        dataset_name="MNIST",           # Name of the dataset
        export_to="TorchTensor"         # Option to export the trained model
    )
    logging.info("Model training finished.")

if __name__ == "__main__":
    main()