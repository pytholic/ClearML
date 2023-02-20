import os
from argparse import ArgumentParser
from datetime import datetime

import torch
import torchmetrics
from clearml import Logger, Task
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


class LightningCIFAR10Classifier(pl.LightningModule):
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

        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=args.num_classes
        )

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

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        acc = self.accuracy(logits, y)
        self.log("train_accuracy", acc)
        self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack(tuple(output["loss"] for output in outputs)).mean()
        self.log("avg_epoch_train_loss: ", avg_train_loss, logger=True, prog_bar=True)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        acc = self.accuracy(logits, y)
        self.log("val_accuracy", acc)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class CIFAR10DataModule(pl.LightningDataModule):
    def setup(self, stage):
        # transforms for images
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        # prepare transforms standard to MNIST
        self.cifar10_train = CIFAR10(
            root="../data", train=True, download=False, transform=transform
        )
        self.cifar10_test = CIFAR10(
            root="../data", train=False, download=False, transform=transform
        )

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=args.batch_size)

    def val_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=args.batch_size)


if __name__ == "__main__":

    # Connecting ClearML with the current process,
    # from here on everything is logged automatically
    task = Task.init(
        project_name="experimentation/pytorch-lightning",
        task_name=f"pytorch-lightning-epochloss-{datetime.now()}",
    )

    pl.seed_everything(0)

    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_classes", default=10, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(max_epochs=20)
    args = parser.parse_args()

    # saves top-K checkpoints based on "val_loss" metric
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        filename="cifar10-{epoch:02d}-{val_loss:.2f}",
    )

    # data
    data_module = CIFAR10DataModule()

    # train
    model = LightningCIFAR10Classifier()
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
    trainer.fit(model, data_module)
