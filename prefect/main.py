from datetime import datetime

from clearml import Task
from config import config
from config.args import Args
from config.config import logger
from model import LightningCIFAR10Classifier
from simple_parsing import ArgumentParser

import pytorch_lightning as pl
from prefect import flow, task
from pytorch_lightning.callbacks import ModelCheckpoint


# Get data
@task
def prepare_data(data_dir):
    from torchvision import transforms
    from torchvision.datasets import CIFAR10

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    trainset = CIFAR10(root=data_dir, train=True, download=False, transform=transform)
    validset = CIFAR10(root=data_dir, train=False, download=False, transform=transform)

    return trainset, validset


# Process data
@task
def create_subset(trainset, validset):
    from torch.utils.data import Subset

    train_subset = Subset(trainset, range(10000))
    valid_subset = Subset(validset, range(1000))

    return train_subset, valid_subset


# Create dataloaders
@task
def create_dataloaders(args, train_subset, valid_subset):
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        train_subset, batch_size=args.batch_size, num_workers=24
    )
    valid_dataloader = DataLoader(
        valid_subset, batch_size=args.batch_size, num_workers=24
    )
    return train_dataloader, valid_dataloader


# Train function
@task
def train(trainer, classifier, train_dataloader, valid_dataloader):
    trainer.fit(classifier, train_dataloader, valid_dataloader)


@flow(
    flow_run_name=f"cifar10-example-{datetime.now()}",
    description="An example flow with prefect.",
)
def main():

    # Read args
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="options")
    args_namespace = parser.parse_args()
    args = args_namespace.options

    # Initialize clearml task
    task = Task.init(
        project_name="experimentation/prefect",
        task_name=f"prefect-example-{datetime.now()}",
    )
    task.connect(args)

    # Saves top-K checkpoints based on "val_loss" metric
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="loss/val_loss",
        mode="min",
        filename="logging-example-{epoch:02d}-{val_loss:.2f}",
    )

    # Data
    trainset, validset = prepare_data(data_dir=config.DATA_DIR)

    train_subset, valid_subset = create_subset(trainset, validset)

    train_dataloader, valid_dataloader = create_dataloaders(
        args, train_subset, valid_subset
    )

    # Train
    # configure trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback],
        default_root_dir=config.LOGS_DIR,
        accelerator="gpu",
        devices=1,
    )

    # define classifier
    classifier = LightningCIFAR10Classifier()

    # fit the model
    train(trainer, classifier, train_dataloader, valid_dataloader)


if __name__ == "__main__":
    main()
