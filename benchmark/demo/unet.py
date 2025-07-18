import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

torch.cuda.empty_cache()


class MiniUNet(nn.Module):

    def __init__(self):
        super().__init__()
        # Encoder 1
        self.enc1_conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.enc1_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)

        # Encoder 2
        self.enc2_conv1 = nn.Conv2d(64, 128, 3, padding=1)
        self.enc2_conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck_conv1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bottleneck_conv2 = nn.Conv2d(256, 256, 3, padding=1)

        # Decoder 1
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1_conv1 = nn.Conv2d(
            256, 128, 3, padding=1
        )  # Input channels are 256 due to concatenation (128 from upsample + 128 from skip)
        self.dec1_conv2 = nn.Conv2d(128, 128, 3, padding=1)

        # Decoder 2
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2_conv1 = nn.Conv2d(
            128, 64, 3, padding=1
        )  # Input channels are 128 due to concatenation (64 from upsample + 64 from skip)
        self.dec2_conv2 = nn.Conv2d(64, 64, 3, padding=1)

        # Final output
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(28 * 28, 10)  # For MNIST output (10 classes)

    def forward(self, x):
        # Encoder 1
        e1_c1 = F.relu(self.enc1_conv1(x))
        e1 = F.relu(self.enc1_conv2(e1_c1))
        p1 = self.pool1(e1)

        # Encoder 2
        e2_c1 = F.relu(self.enc2_conv1(p1))
        e2 = F.relu(self.enc2_conv2(e2_c1))
        p2 = self.pool2(e2)

        # Bottleneck
        b_c1 = F.relu(self.bottleneck_conv1(p2))
        b = F.relu(self.bottleneck_conv2(b_c1))

        # Decoder 1 (up-convolution + skip connection)
        up_b = self.up1(b)
        # Pad e2 if its dimensions don't exactly match up_b's
        # (This can happen if input dimensions are odd, but for 28x28, it should be fine)
        # if up_b.size()[2:] != e2.size()[2:]:
        #     up_b = F.interpolate(up_b, size=e2.size()[2:], mode='nearest')
        d1_cat = torch.cat([up_b, e2],
                           dim=1)  # Concatenate along channel dimension
        d1_c1 = F.relu(self.dec1_conv1(d1_cat))
        d1 = F.relu(self.dec1_conv2(d1_c1))

        # Decoder 2 (up-convolution + skip connection)
        up_d1 = self.up2(d1)
        # if up_d1.size()[2:] != e1.size()[2:]:
        #     up_d1 = F.interpolate(up_d1, size=e1.size()[2:], mode='nearest')
        d2_cat = torch.cat([up_d1, e1],
                           dim=1)  # Concatenate along channel dimension
        d2_c1 = F.relu(self.dec2_conv1(d2_cat))
        d2 = F.relu(self.dec2_conv2(d2_c1))

        # Final output for segmentation-like task (1 channel)
        out_segmentation = self.final_conv(
            d2
        )  # No ReLU here as it's typically before the final output layer for classification/regression

        # Flatten and Linear for MNIST classification
        flat = self.flatten(out_segmentation)
        out_classification = self.linear(
            flat)  # No activation here, as CrossEntropyLoss expects logits

        return out_classification


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MiniUNet().to(device)
    dummy_input = torch.randn(1, 1, 28, 28).to(device)

    # *** CRITICAL CHANGES FOR BENCHMARKING ***
    # Match the optimizer configuration from GraphQL
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(),
        lr=0.0003,
        momentum=0.9,
        weight_decay=0.0001,
        nesterov=True)  # Added momentum, weight_decay, nesterov
    writer = SummaryWriter()

    # Load MNIST dataset with Normalize transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ),
                             (0.3081, ))  # Standard MNIST mean and std
    ])
    train_dataset = datasets.MNIST(root="./data",
                                   train=True,
                                   download=True,
                                   transform=transform)
    loader = DataLoader(train_dataset,
                        batch_size=1024,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True)

    writer.add_graph(net, dummy_input)

    for epoch in range(10):
        net.train()
        start_epoch = time.time()
        total_loss = 0
        correct = 0
        total = 0

        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            t0 = time.time()
            optimizer.zero_grad()
            out = net(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            batch_time = time.time() - t0
            writer.add_scalar("Batch Time", batch_time,
                              epoch * len(loader) + i)
            total_loss += loss.item()

            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        end_epoch = time.time()
        epoch_time = end_epoch - start_epoch
        acc = correct / total
        print(
            f"Epoch {epoch + 1}: Loss = {total_loss:.4f}, Accuracy = {acc*100:.2f}%, Time = {epoch_time:.4f}s"
        )
        writer.add_scalar("Epoch Time", epoch_time, epoch)
        writer.add_scalar("Loss/Epoch", total_loss, epoch)
        writer.add_scalar("Accuracy/Epoch", acc, epoch)

    writer.close()


if __name__ == "__main__":
    main()
