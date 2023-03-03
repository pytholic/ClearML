from clearml.automation.controller import PipelineDecorator


# Data processing
@PipelineDecorator.component(cache=True, return_values=["trainset, validset"])
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


@PipelineDecorator.component(cache=True, return_values=["train_subset, valid_subset"])
def create_subset(trainset, validset):
    from torch.utils.data import Subset

    train_subset = Subset(trainset, range(10000))
    valid_subset = Subset(validset, range(1000))

    return train_subset, valid_subset


# Create dataloaders
def create_dataloaders(args, train_subset, valid_subset):
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        train_subset, batch_size=args.batch_size, num_workers=24, shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_subset, batch_size=args.batch_size, num_workers=24, shuffle=False
    )
    return train_dataloader, valid_dataloader


@PipelineDecorator.pipeline(
    name="Test Pipeline", project="Pipeline Examples", version="0.1"
)
def main():
    from argparse import Namespace

    import utils
    from config import config
    from config.config import logger
    from model import LightningCIFAR10Classifier

    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint

    # Read args
    logger.info("Reading arguments...")
    args_path = config.CONFIG_DIR / "args.json"
    args = Namespace(**utils.load_dict(filepath=args_path))

    # Saves top-K checkpoints based on "val_loss" metric
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="loss/val_loss",
        mode="min",
        filename="logging-example-{epoch:02d}-{val_loss:.2f}",
    )

    # Data
    logger.info("Preparing data...")
    trainset, validset = prepare_data(data_dir=config.DATA_DIR)

    logger.info("Creating subsets...")
    train_subset, valid_subset = create_subset(trainset, validset)

    logger.info("Creating datalaoders...")
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
    logger.info("Starting training...")

    # define classifier
    classifier = LightningCIFAR10Classifier()

    # fit the model
    trainer.fit(classifier, train_dataloader, valid_dataloader)


if __name__ == "__main__":

    PipelineDecorator.run_locally()
    main()
