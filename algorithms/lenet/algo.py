from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infra.datasets import load_fashion_mnist
from infra.minispec import MiniSpec


def _output_size(size, kernel_size, stride, padding):
    return (size + 2 * padding - kernel_size) // stride + 1


class _Conv2dFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding):
        batch, channels, height, width = x.shape
        out_channels, _, kernel_height, kernel_width = weight.shape
        padded = F.pad(x, (padding, padding, padding, padding))
        columns = F.unfold(padded, (kernel_height, kernel_width), stride=stride)
        output = torch.einsum("oc,ncl->nol", weight.reshape(out_channels, -1), columns) #
        output = output + bias.view(1, -1, 1)
        out_height = _output_size(height, kernel_height, stride, padding)
        out_width = _output_size(width, kernel_width, stride, padding)
        ctx.save_for_backward(x, weight, columns)
        ctx.stride = stride
        ctx.padding = padding
        ctx.output_size = (out_height, out_width)
        return output.reshape(batch, out_channels, out_height, out_width)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, columns = ctx.saved_tensors
        batch, channels, height, width = x.shape
        out_channels = weight.shape[0]
        grad_columns = grad_output.reshape(batch, out_channels, -1)
        grad_weight = torch.einsum("nol,ncl->oc", grad_columns, columns)  #
        grad_bias = grad_columns.sum(dim=(0, 2))#
        grad_input_columns = torch.einsum(
            "oc,nol->ncl", weight.reshape(out_channels, -1), grad_columns
        )         #
        padded_height = height + 2 * ctx.padding
        padded_width = width + 2 * ctx.padding
        grad_padded = F.fold(
            grad_input_columns,
            (padded_height, padded_width),
            weight.shape[-2:],
            stride=ctx.stride,
        )
        if ctx.padding:
            grad_input = grad_padded[
                :, :, ctx.padding:-ctx.padding, ctx.padding:-ctx.padding
            ]
        else:
            grad_input = grad_padded
        return grad_input, grad_weight.reshape_as(weight), grad_bias, None, None


class _MaxPoolFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kernel_size, stride):
        batch, channels, height, width = x.shape
        columns = F.unfold(x, (kernel_size, kernel_size), stride=stride)
        columns = columns.reshape(batch, channels, kernel_size * kernel_size, -1)
        values, indices = columns.max(dim=2)
        out_height = _output_size(height, kernel_size, stride, 0)
        out_width = _output_size(width, kernel_size, stride, 0)
        ctx.save_for_backward(indices)
        ctx.input_shape = x.shape
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.output_size = (out_height, out_width)
        return values.reshape(batch, channels, out_height, out_width)

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        batch, channels, height, width = ctx.input_shape
        grad_columns = torch.zeros(
            batch,
            channels,
            ctx.kernel_size * ctx.kernel_size,
            indices.shape[-1],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        grad_columns.scatter_(2, indices.unsqueeze(2), grad_output.reshape(batch, channels, -1).unsqueeze(2))
        return F.fold(grad_columns.reshape(batch, -1, indices.shape[-1]), (height, width),
                      ctx.kernel_size, stride=ctx.stride), None, None


class _AvgPoolFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kernel_size, stride):
        batch, channels, height, width = x.shape
        columns = F.unfold(x, (kernel_size, kernel_size), stride=stride)
        values = columns.reshape(batch, channels, kernel_size * kernel_size, -1).mean(dim=2)
        out_height = _output_size(height, kernel_size, stride, 0)
        out_width = _output_size(width, kernel_size, stride, 0)
        ctx.input_shape = x.shape
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.scale = kernel_size * kernel_size
        return values.reshape(batch, channels, out_height, out_width)

    @staticmethod
    def backward(ctx, grad_output):
        batch, channels, height, width = ctx.input_shape
        grad = grad_output.reshape(batch, channels, -1).unsqueeze(2).expand(
            -1, -1, ctx.scale, -1
        ) / ctx.scale
        return F.fold(grad.reshape(batch, -1, grad.shape[-1]), (height, width),
                      ctx.kernel_size, stride=ctx.stride), None, None


class Conv2dScratch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return _Conv2dFunc.apply(x, self.weight, self.bias, self.stride, self.padding)


class Pool2dScratch(nn.Module):
    def __init__(self, mode="avg", kernel_size=2, stride=2):
        super().__init__()
        self.mode = mode
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        function = _MaxPoolFunc if self.mode == "max" else _AvgPoolFunc
        return function.apply(x, self.kernel_size, self.stride)


class LinearScratch(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return x.reshape(x.shape[0], -1) @ self.weight + self.bias


class LeNetScratch(nn.Module):
    def __init__(self, pool="avg"):
        super().__init__()
        self.conv1 = Conv2dScratch(1, 6, 5, padding=2)
        self.pool1 = Pool2dScratch(pool)
        self.conv2 = Conv2dScratch(6, 16, 5)
        self.pool2 = Pool2dScratch(pool)
        self.fc1 = LinearScratch(16 * 5 * 5, 120)
        self.fc2 = LinearScratch(120, 84)
        self.fc3 = LinearScratch(84, 10)

    def forward(self, x):
        x = self.pool1(torch.tanh(self.conv1(x)))
        x = self.pool2(torch.tanh(self.conv2(x)))
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)


def cross_entropy(y_hat, y):
    return -torch.log_softmax(y_hat, dim=1)[torch.arange(y.shape[0], device=y.device), y].mean()


class SGDScratch:
    def __init__(self, params, lr=0.05, momentum=0.0):
        self.params = list(params)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.velocity = [torch.zeros_like(p) for p in self.params]
        self.param_groups = [{"lr": self.lr, "params": self.params}]

    def zero_grad(self):
        for parameter in self.params:
            parameter.grad = None

    @torch.no_grad()
    def step(self):
        for parameter, velocity in zip(self.params, self.velocity):
            if parameter.grad is not None:
                velocity.mul_(self.momentum).add_(parameter.grad)
                parameter.add_(velocity, alpha=-self.lr)

    def state_dict(self):
        return {"lr": self.lr, "momentum": self.momentum, "velocity": self.velocity}

    def load_state_dict(self, state):
        self.lr = float(state["lr"])
        self.momentum = float(state.get("momentum", 0.0))
        self.param_groups[0]["lr"] = self.lr
        self.velocity = [v.to(p.device) for v, p in zip(state["velocity"], self.params)]


def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()


class LeNetSpec(MiniSpec):
    name = "lenet"
    model = LeNetScratch
    loss = cross_entropy
    datasets = load_fashion_mnist
    optimizer = SGDScratch
    metrics = {"acc": accuracy}
    config = {
        "epochs": 10,
        "lr": 0.05,
        "momentum": 0.9,
        "batch_size": 128,
        "pool": "avg",
        "data_root": str(ROOT / "data"),
        "patience": 0,
        "amp": False,
    }

    def build_model(self, cfg):
        return LeNetScratch(pool=str(cfg.pool))

    def build_optimizer(self, params, cfg):
        return SGDScratch(params, lr=float(cfg.lr), momentum=float(cfg.momentum))


SPEC = LeNetSpec()