print("importing libraries...")
import numpy as np
import torch
from torch import nn 
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torchvision
import torchvision.transforms as Transforms
from torchsummary import summary
from matplotlib import pyplot as plt
print("libraries imported")

#パラメーターの設定を行う
IMAGE_SIZE = 32
CHANNELS = 1
BATCH_SIZE = 128
EMBEDDING_DIM = 2
EPOCHS = 10
LEARNING_RATE = 1e-3

DEVICE = 'cude'if torch.cuda.is_available() else 'cpu'
print(DEVICE)

