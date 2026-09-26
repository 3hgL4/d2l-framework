
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parents[2]   
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))    #谁的最前面,sys path长什么样

from infra.minispec import MiniSpec

DATA_ROOT = ROOT / "data"    # 数据在 d2l/data/FashionMNIST/raw（2026-09-22 统一）


def dropout_layer(X, p):
    assert 0.0 <= p <= 1.0
    if p == 0.0:                #代码不完整,缺少p=1
        return X
    mask = (torch.rand(X.shape, device=X.device) > p).float()    #不写的话掩码默认建在 CPU 上，和 GPU 上的 X 相乘会报 device 不匹配.揭示了这两个硬件在跑算法时数据是什么关系.         
    return mask * X / (1.0 - p)


def log_softmax(X):
    X_max = X.max(dim=1, keepdim=True).values
    return X - X_max - (X - X_max).exp().sum(dim=1, keepdim=True).log()


def cross_entropy(y_hat, y):
    return -y_hat[range(len(y_hat)), y].mean()


class MLPScratch(nn.Module):
    def __init__(self, num_inputs: int = 784, num_hiddens: int = 256,
                 num_outputs: int = 10, dropout1: float = 0.0,
                 dropout2: float = 0.0):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens))
        self.W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(num_outputs))
        self.dropout1 = dropout1
        self.dropout2 = dropout2

    def forward(self, X):
        X = X.reshape((-1, self.W1.shape[0]))         # num_inputs: int = 784是不是需要自己算,1*28*28
        if self.training:                        
            X = dropout_layer(X, self.dropout1)         
        H1 = torch.relu(X @ self.W1 + self.b1)          
        if self.training:
            H1 = dropout_layer(H1, self.dropout2)      #为什么两个h1命名
        return log_softmax(H1 @ self.W2 + self.b2)      



def load_data(cfg):
    ds = torchvision.datasets.FashionMNIST
    out = []
    for train in (True, False):
        d = ds(root=str(DATA_ROOT), train=train, download=False)
        X = d.data.unsqueeze(1).float().div(255)     # unsqueeze(1),div(255)用法,参数什么意思
        out.append(TensorDataset(X, d.targets))            #数据类型和形状怎么变的
    return tuple(out)



class SGDScratch:
    def __init__(self, params, lr: float = 0.01):
        self.params = list(params)
        self.lr = float(lr)
        self.param_groups = [{"lr": self.lr, "params": self.params}]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        for p in self.params:
            if p.grad is not None:
                p -= self.lr * p.grad

    def state_dict(self):
        return {"lr": self.lr}

    def load_state_dict(self, state):
        self.lr = float(state["lr"])
        self.param_groups[0]["lr"] = self.lr


def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()



class MLPSpec(MiniSpec):
    name = "mlp"
    model = MLPScratch
    loss = cross_entropy
    datasets = load_data
    metrics = {"acc": accuracy}
    optimizer = SGDScratch                 # 手写版 SGD，lr 仍走 cfg.lr 可 override
    config = {"epochs": 20, "lr": 0.01, "batch_size": 256,
              "num_hiddens": 256, "dropout1": 0.0, "dropout2": 0.0,
              "patience": 0, "amp": False}

    def build_model(self, cfg):
        return MLPScratch(num_hiddens=int(cfg.num_hiddens),
                          dropout1=float(cfg.dropout1),
                          dropout2=float(cfg.dropout2))


SPEC = MLPSpec()
