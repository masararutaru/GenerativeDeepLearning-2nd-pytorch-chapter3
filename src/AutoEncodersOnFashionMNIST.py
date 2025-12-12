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

#データローダーの設定
def get_dataloaders():
    #データの前処理を行う。tensor形式に変換
    transform = Transforms.Compose([
        Transforms.ToTensor(),
        #Pad関数は上下左右2pxを黒（デフォルト）で埋める
        Transforms.Pad(2)
    ])
    print("loading dataset...")
    train_ds = torchvision.datasets.FashionMNIST(root = './data', train = True, download = True, transform = transform)
    test_ds = torchvision.datasets.FashionMNIST(root = './data', train = False, download = True, transform = transform)
    train_loader = DataLoader(dataset = train_ds, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)
    test_loader = DataLoader(dataset = test_ds, batch_size = BATCH_SIZE, shuffle = False, num_workers = 4)

    return train_loader, test_loader

train_loader, test_loader = get_dataloaders()
print(next(iter(train_loader))[0].shape)

