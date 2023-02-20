import os

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import MNIST


class LightningMNISTClassifier(pl.LightningModule):
    super().__init()

    # Mnist images are (1,28,28)
