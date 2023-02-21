import os
from argparse import Namespace
from datetime import datetime

import torch
import torchmetrics
import utils
from clearml import Task
from clearml.automation.controller import PipelineDecorator
from config import config
from config.config import logger
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
    def __init__(self):
        super().__init__()

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def prepare_data(self):

        # prepare transforms standard to MNIST
        self.cifar10_train = CIFAR10(
            root=config.DATA_DIR, train=True, download=False, transform=self.transform
        )
        self.cifar10_test = CIFAR10(
            root=config.DATA_DIR, train=False, download=False, transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.cifar10_train, batch_size=args.batch_size, num_workers=24
        )

    def val_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=args.batch_size, num_workers=24)

# Data
@PipelineDecorator.component()
def load_and_prepare_data()
    data_module = CIFAR10DataModule()
    return data_module

# Model


@PipelineDecorator.component()
def create_model(model=model):

    classifier = LightningCIFAR10Classifier(model=model)
    return classifier

@PipelineDecorator.component()
def read_args():

    logger.info("Reading arguments...")
    args_path = config.CONFIG_DIR / "args.json"
    args = Namespace(**utils.load_dict(filepath=args_path))
    return args

@PipelineDecorator.component()
def create_task():
    task = Task.init(
        project_name="experimentation/pipelines",
        task_name=f"pipeline-example-{datetime.now()}",
    )

@PipelineDecorator.component()
def train(args, callbacks, classifier, data_module):
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=[callbacks], default_root_dir=config.LOGS_DIR
    )
    trainer.fit(classifier, data_module)


@PipelineDecorator.pipeline(name="Test Pipeline", project="Pipeline Examples", version="0.1")
def main():
    
    # Data
    logger.info("Preparing data...")
    data_module = load_and_prepare_data()
    
    # Model
    logger.info("Creating model...")
    model = cifar10Classifier()
    classifer = create_model(model=model)

    # Read args
    logger.info("Reading arguments...")
    args = read_args()
    
    # Create task
    logger.info("Initilizaing clearML task...")
    create_task()

    # Saves top-K checkpoints
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="loss/val_loss",
        mode="min",
        filename="pipeline-example-{epoch:02d}-{val_loss:.2f}",
    )
    
    # Train
    logger.info("Starting training...")
    train(args, checkpoint_callback, classifer, data_module)
    

if __name__ == "__main__":

    main()
