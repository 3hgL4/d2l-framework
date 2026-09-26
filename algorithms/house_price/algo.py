"""Kaggle 房价预测基线：从零线性回归（d2l 4.10）—— MiniSpec 声明式接入。

任务：House Prices - Advanced Regression Techniques，预测房价（连续值回归）。

流水线：
    预处理（d2l 4.10.4）：
      - 连续特征：标准化（均值/方差取 train+test 合并集，d2l 同款）→ 缺失填 0；
      - 离散特征：独热编码 pd.get_dummies(dummy_na=True)，
        "NA" 视为合法类别并生成指示符列（满足"缺失也是信息"）；
      - train/test 合并后统一编码，保证特征空间一致。
    模型：手写 w, b 的线性回归（ nn.Parameter 从零维护）。
    损失：训练用 MSE（原价空间，d2l 同款）。
    评估：log-RMSE = sqrt(MSE(log(clip(y_hat,1,∞)), log(y)))，对数空间衡量相对误差。
    优化器：手写 Adam（AdamScratch，Kingma & Ba 2015），lr=0.001,
    betas=(0.9,0.999), eps=1e-8, weight_decay=0（全部按题目给定固定）。
    验证：cfg.k=5 折交叉验证，--sweep fold=0,1,2,3,4 逐折训练；
    fold=-1 时全量训练并生成 Kaggle 提交文件到 addition 目录。

运行（交叉验证）: python run.py --algo house_price --sweep fold=0,1,2,3,4
运行（提交文件）: python run.py --algo house_price --override fold=-1
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]   # d2l 项目根
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infra.callbacks import Callback
from infra.minispec import MiniSpec

DATA_DIR = ROOT / "data" / "house-prices-advanced-regression-techniques"
ADDITION_DIR = ROOT.parent / "addition"      # 结果输出目录


# ---------------------------------------------------------------------------
# ① 预处理：返回合并编码后的 (特征张量, 训练标签, 测试 Id 列)
#    标签保持原价（训练 MSE 用原价，log 只在评估指标里做，d2l 4.10 同款）
# ---------------------------------------------------------------------------
def build_features():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    all_features = pd.concat((train.iloc[:, 1:-1], test.iloc[:, 1:]))  # 去 Id/SalePrice

    numeric = all_features.dtypes[all_features.dtypes != "object"].index
    # 连续：标准化后缺失填 0（填 0 = 填回均值，不引入偏移）
    all_features[numeric] = all_features[numeric].apply(
        lambda x: (x - x.mean()) / (x.std()))
    all_features[numeric] = all_features[numeric].fillna(0)
    # 离散：独热编码，dummy_na=True 把 "NA" 当合法类别生成指示符列
    all_features = pd.get_dummies(all_features, dummy_na=True)
    all_features = all_features.astype("float32")

    n_train = len(train)
    X = torch.tensor(all_features.values, dtype=torch.float32)
    y = torch.tensor(train["SalePrice"].values, dtype=torch.float32).reshape(-1, 1)
    return X, y, n_train, test["Id"].values


# ---------------------------------------------------------------------------
# ② 模型：手写参数的线性回归（d2l 3.2）。y 须为列向量 (n,1)，与标签形状一致
# ---------------------------------------------------------------------------
class LinearRegressionScratch(nn.Module):
    def __init__(self, num_inputs: int, lr: float = 0.001):
        super().__init__()
        self.w = nn.Parameter(torch.randn(num_inputs, 1) * 0.01)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        return X @ self.w + self.b


# ---------------------------------------------------------------------------
# ③ 数据：按 cfg.fold 做 k 折切分（fold=0..k-1）或全量训练（fold=-1）
#    切分用 rng(seed+fold) 置换，保证每折可复现
# ---------------------------------------------------------------------------
def load_data(cfg):
    X, y, n_train, _ = build_features()
    fold, k = int(cfg.fold), int(cfg.k)
    if fold == -1:                       # 提交模式：全量训练，val=训练集（仅报告拟合）
        ds = torch.utils.data.TensorDataset(X[:n_train], y)
        return ds, ds
    rng = np.random.default_rng(int(cfg.seed) + fold)
    perm = rng.permutation(n_train)
    val_idx = perm[fold::k]              # 第 fold 折做验证，其余做训练
    train_idx = np.setdiff1d(perm, val_idx)
    return (torch.utils.data.TensorDataset(X[train_idx], y[train_idx]),
            torch.utils.data.TensorDataset(X[val_idx], y[val_idx]))


# ---------------------------------------------------------------------------
# ④ 手写 Adam（Kingma & Ba, ICLR 2015）：一阶/二阶矩 + 偏差修正。
#    m̂ = m/(1-β1^t), v̂ = v/(1-β2^t), θ ← θ - lr·m̂/(√v̂+ε)
#    契约要求 step()/zero_grad()/state_dict()/load_state_dict()；param_groups
#    供 lr 日志读取。weight_decay 为经典 L2 形式（梯度加 wd·θ），题目给定 0
# ---------------------------------------------------------------------------
class AdamScratch:
    def __init__(self, params, lr: float = 0.001, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        self.params = list(params)
        self.lr, self.betas = float(lr), (float(betas[0]), float(betas[1]))
        self.eps, self.wd = float(eps), float(weight_decay)
        self.param_groups = [{"lr": self.lr, "params": self.params}]
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        self.t += 1
        b1, b2 = self.betas
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad + self.wd * p if self.wd else p.grad
            self.m[i].mul_(b1).add_(g, alpha=1 - b1)
            self.v[i].mul_(b2).addcmul_(g, g, value=1 - b2)
            m_hat = self.m[i] / (1 - b1 ** self.t)
            v_hat = self.v[i] / (1 - b2 ** self.t)
            p -= self.lr * m_hat / (v_hat.sqrt() + self.eps)

    def state_dict(self):
        return {"lr": self.lr, "t": self.t, "m": self.m, "v": self.v}

    def load_state_dict(self, state):
        self.lr = float(state["lr"])
        self.t = int(state["t"])
        self.m, self.v = state["m"], state["v"]
        self.param_groups[0]["lr"] = self.lr


# ---------------------------------------------------------------------------
# ⑤ 指标：log-RMSE（d2l 4.10）：预测截断到 ≥1 后与真实值同取对数再算 RMSE，
#    衡量相对误差（Kaggle 官方评测指标）
# ---------------------------------------------------------------------------
def log_rmse(y_hat, y):
    clipped = torch.clamp(y_hat, 1, float("inf"))
    return float(torch.sqrt(nn.functional.mse_loss(torch.log(clipped), torch.log(y))))


# ---------------------------------------------------------------------------
# ⑥ 提交回调：fold=-1 的全量训练结束后，预测测试集并写 Kaggle 提交文件
#    （模型输出即原价空间，无需逆变换）
# ---------------------------------------------------------------------------
class KaggleSubmit(Callback):
    def on_train_end(self, ctx):
        if int(ctx.cfg.fold) != -1:
            return
        X, _, n_train, test_ids = build_features()
        X_test = X[n_train:]
        device = ctx.device
        ctx.model.eval()
        with torch.no_grad():
            preds = ctx.model(X_test.to(device)).cpu().reshape(-1)
        assert len(preds) == len(test_ids), (
            f"预测数 {len(preds)} != 测试样本数 {len(test_ids)}，"
            f"请检查模型输出维度（常见 bug：输出层漏投影到 1 维）")
        ADDITION_DIR.mkdir(parents=True, exist_ok=True)
        out = ADDITION_DIR / "submission.csv"
        pd.DataFrame({"Id": test_ids,
                      "SalePrice": preds.numpy()}).to_csv(out, index=False)
        ctx.logger.info(f"[提交] Kaggle 提交文件已保存: {out}")


# ---------------------------------------------------------------------------
# ⑦ 声明注册：五组算法知识
#    优化器用 (类, 固定超参) 形式：betas/eps/weight_decay 按题目固定，
#    lr 仍走 cfg.lr（--override lr=... 可调）
# ---------------------------------------------------------------------------
class HousePriceSpec(MiniSpec):
    name = "house_price"
    model = LinearRegressionScratch
    loss = nn.MSELoss
    datasets = load_data
    metrics = {"log_rmse": log_rmse}
    optimizer = (AdamScratch, {"betas": (0.9, 0.999), "eps": 1e-8,
                               "weight_decay": 0.0})
    config = {"epochs": 100, "lr": 0.001, "batch_size": 64,
              "k": 5, "fold": 0,
              "monitor": "val_log_rmse", "patience": 0, "amp": False}

    def build_model(self, cfg):
        X, _, _, _ = build_features()
        return LinearRegressionScratch(num_inputs=X.shape[1], lr=float(cfg.lr))

    def get_callbacks(self, cfg):
        return [KaggleSubmit()]


SPEC = HousePriceSpec()
