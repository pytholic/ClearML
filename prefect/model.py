import torch
import torchmetrics
from config import config
from config.args import Args
from simple_parsing import ArgumentParser
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl

parser = ArgumentParser()
parser.add_arguments(Args, dest="options")
args_namespace = parser.parse_args()
args = args_namespace.options

# Model class
class cifar10Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        x = torch.randn(3, 32, 32).view(-1, 3, 32, 32)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, args.num_classes)

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


class LightningCIFAR10Classifier(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = cifar10Classifier()
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=args.num_classes
        )

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.model(x)
        loss = self.cross_entropy_loss(logits, y)
        acc = self.accuracy(logits, y)
        self.log("accuracy/train_accuracy", acc)
        self.log("loss/train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.model(x)
        loss = self.cross_entropy_loss(logits, y)
        acc = self.accuracy(logits, y)
        self.log("accuracy/val_accuracy", acc)
        self.log("loss/val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate)
        return optimizer
