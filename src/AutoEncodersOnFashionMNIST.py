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

DEVICE = 'cuda'if torch.cuda.is_available() else 'cpu'
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

#autoencoderの定義
class Encoder(nn.Module):
    def __init__(self, latents):
        super().__init__()
        self.latents = latents
        self.model = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 2, padding = 1),
            #出力サイズの公式Conv2d:floor((in + 2 * padding -kernel_size)/stride) +1)
            #ReLU関数では出力サイズは変わらない
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features = 128 * 4 *4, out_features = self.latents)
        )#latentsがエンコーダーの出力になっている
        
    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, latents):
        super().__init__()
        self.latents = latents
        self.fc = nn.Linear(self.latents, 128 * 4 * 4)

        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 32, out_channels = 1, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
        )#latentsがデコーダーの入力になっている

    def forward(self, x):
        x = self.fc(x)
        x = x.reshape(x.shape[0], 128, 4, 4)
        x = self.model(x)
        return x

class AE(nn.Module):
    def __init__(self, latents):
        super().__init__()
        self.encoder = Encoder(latents)
        self.decoder = Decoder(latents)

    def forward(self, x):
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x

ae = AE(EMBEDDING_DIM).to(DEVICE)
#モデルの要約を表示＆設計があっていないとエラー表示になる！
summary(ae,(1,32,32))