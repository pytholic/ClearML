import argparse
import os
from datetime import datetime

import clearml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from clearml import Logger, Task
from torchvision import datasets, transforms


# Define the network architecture
class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding="same")
        self.conv2 = nn.Conv2d(32, 64, 5, padding="same")
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        x = torch.randn(3, 32, 32).view(-1, 3, 32, 32)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, 10)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = self.dropout1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = self.dropout2(x)

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding="same")
        self.conv2 = nn.Conv2d(32, 64, 5, padding="same")
        self.conv3 = nn.Conv2d(64, 128, 5, padding="same")
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)
        self.dropout3 = nn.Dropout2d(0.5)

        x = torch.randn(3, 32, 32).view(-1, 3, 32, 32)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, 10)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = self.dropout1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = self.dropout2(x)
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = self.dropout3(x)

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Defien train loop
def train(args, model, device, train_loader, epoch):
    model.train()
    correct = 0
    train_loss = 0
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for batch_idx, (images, target) in enumerate(train_loader, 0):
        # Move the data to the GPU if available
        images, target = images.to(device), target.to(device)
        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        output = model(images)
        loss = F.nll_loss(output, target)
        # Backward pass
        loss.backward()
        optimizer.step()
        # Accumulate the training loss
        train_loss += loss.item()
        # Calculate accuracy
        pred = output.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    train_loss /= len(train_loader)
    train_acc = 100.0 * correct / len(train_loader.dataset)
    Logger.current_logger().report_scalar(
        "train", "loss", value=train_loss, iteration=epoch
    )
    Logger.current_logger().report_scalar(
        "train", "accuracy", value=train_acc, iteration=epoch
    )
    if epoch % args.log_interval == 0:
        print(
            f"Train Epoch: {epoch} [{epoch}/{args.epochs} ({100. * epoch / args.epochs}%)]\tLoss: {loss.item()}\tTrain accuracy: {train_acc}"
        )


# Define the test loop
def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100.0 * correct / len(test_loader.dataset)

    Logger.current_logger().report_scalar(
        "test", "loss", iteration=epoch, value=test_loss
    )
    Logger.current_logger().report_scalar(
        "test", "accuracy", iteration=epoch, value=(correct / len(test_loader.dataset))
    )
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({test_acc:.0f}%)\n"
    )


def train_and_evaluate_model(
    args, model, model_name, device, train_loader, test_loader
):
    task = Task.init(
        project_name="pytorch-testing",
        task_name=f"PyTorch-CIFAR10-{model_name}-100epochs-{datetime.now()}",
    )

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, epoch)
        test(model, device, test_loader, epoch)

    if args.save_model:
        torch.save(model.state_dict(), f"models/cifar10-{datetime.now()}.pt")

    task.close()


# The main loop
def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch CIFAR-10 Multi-run Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        metavar="N",
        help="how many batches to wait before logging training status",
    )

    parser.add_argument(
        "--save-model",
        action="store_true",
        default=True,
        help="For Saving the current Model",
    )
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load the CIFAR-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    testset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    model1 = Net1().to(device)
    model2 = Net2().to(device)

    # Run models one by one
    train_and_evaluate_model(args, model1, "Model 1", device, train_loader, test_loader)
    train_and_evaluate_model(args, model2, "Model 2", device, train_loader, test_loader)


if __name__ == "__main__":
    main()
