from argparse import Namespace

import torch
import utils
from config import config
from torch import nn
from torch.nn import functional as F

args_path = config.CONFIG_DIR / "args.json"
args = Namespace(**utils.load_dict(filepath=args_path))

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
