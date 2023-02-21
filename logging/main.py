import os
from argparse import Namespace
from datetime import datetime

import torch
import torchmetrics
import utils
from clearml import Task
from config import config
from config.config import console_logger
from model import *
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


class LightningCIFAR10Classifier(pl.LightningModule):
    def __init__(self, model):
        super().__init__()

        self.model = model
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


class CIFAR10DataModule(pl.LightningDataModule):
    def setup(self, stage):
        # transforms for images
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        # prepare transforms standard to MNIST
        self.cifar10_train = CIFAR10(
            root=config.DATA_DIR, train=True, download=False, transform=transform
        )
        self.cifar10_test = CIFAR10(
            root=config.DATA_DIR, train=False, download=False, transform=transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.cifar10_train, batch_size=args.batch_size, num_workers=24
        )

    def val_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=args.batch_size, num_workers=24)


if __name__ == "__main__":

    # Read args
    console_logger.info("Reading arguments...")
    args_path = config.CONFIG_DIR / "args.json"
    args = Namespace(**utils.load_dict(filepath=args_path))

    # Connecting ClearML with the current process,
    # from here on everything is logged automatically
    console_logger.info("Initilizaing clearML task...")
    task = Task.init(
        project_name="experimentation/logging",
        task_name=f"logging-example-{datetime.now()}",
    )

    # # saves top-K checkpoints based on "val_loss" metric
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="loss/val_loss",
        mode="min",
        filename="logging-example-{epoch:02d}-{val_loss:.2f}",
    )

    # Data
    data_module = CIFAR10DataModule()

    # Model
    model = cifar10Classifier()

    # Train
    classifier = LightningCIFAR10Classifier(model=model)
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=[checkpoint_callback], default_root_dir=config.LOGS_DIR
    )

    console_logger.info("Starting training...")
    trainer.fit(classifier, data_module)

    # logger.debug("Used for debugging your code.")
    # logger.info("Informative messages from your code.")
    # logger.warning("Everything works but there is something to be aware of.")
    # logger.error("There's been a mistake with the process.")
    # logger.critical("There is something terribly wrong and process may terminate.")
