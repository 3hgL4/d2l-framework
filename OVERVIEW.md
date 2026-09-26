# d2l-framework 全貌（AI 快速上下文）

> 本文件面向 AI 助手：读完即可获得本框架的完整心智模型，无需遍历源码。
> 文档分工：README.md（门面+理念+导航）· USAGE.md（**命令唯一出处**）· 说明书.md（使用者手册：心智模型/五组声明/FAQ）· CHANGELOG.md（冻结面变更账本）。

## 1. 一句话定位

《动手学深度学习》自学用的**极简训练框架**：算法与工程严格分离——每学一个新算法只新增 `algorithms/<名字>/algo.py`（约 30 行纯算法代码），训练循环/断点/早停/日志/曲线等工程由 `infra/` 一次性代劳，**不严格对齐 d2l 书的数字**，只对齐算法结构性事实（层结构/激活/损失形式）。

## 2. 目录地图

```
d2l/
├── run.py              唯一入口：--algo / --resume / --override / --sweep / --seeds
├── infra/              工程层（算法无关，绝不 import algorithms.*，红线有静态检查）
│   ├── contract.py     AlgoSpec 契约 = infra 与算法层唯一接口（duck-typing）
│   ├── minispec.py     声明式基类 = AlgoSpec 糖衣，五处填空覆盖 90% 场景（推荐）
│   ├── trainer.py      通用训练循环（只认契约，调度回调）
│   ├── config.py       三层配置合并：infra 默认 ← 算法 config ← --override
│   ├── data.py         DataLoader 工厂（pin_memory/Windows spawn 安全）
│   ├── datasets.py     数据文件格式解析（IDX 手写读取器；路径/归一化决策仍在算法层）
│   ├── evaluator.py    验证集评估（inference_mode）
│   ├── callbacks.py    早停/best 追踪/存档/曲线（可插拔）
│   ├── checkpoint.py   断点全状态：权重+优化器+RNG+extras（续训=状态等价）
│   └── seed/logger/history/viz
├── algorithms/         算法层（每算法一目录，核心文件 algo.py）
│   ├── _template/      接入模板：复制→五处填空→运行
│   ├── softmax/        从零实现范例（手写 W/b + 稳定 log-softmax + 预测图回调）
│   ├── mlp/            784→256→10，ReLU，可选暂退法 dropout_in/dropout_h（默认 0）
│   ├── linreg/         线性回归（学习者练习位）
│   ├── logreg_iris/    iris 上的对数几率回归（手写 sigmoid + 从零 BCE）
│   ├── lenet/          d2l 6.6 全组件从零：手写卷积(im2col+手写backward,可pad/stride)/最大+平均汇聚/全连接/稳定交叉熵/手写SGD/IDX解析
│   ├── house_price/    Kaggle 房价从零线性回归 + 手写 AdamScratch，k 折交叉验证
│   ├── house_mlp/      房价 MLP（复用 house_price 特征件与 mlp 的 dropout）
│   └── fake/           合成数据测试算法（冒烟用活样例）
├── tests/smoke_1..8    冒烟回归：改 infra 后必跑，任何一条 FAIL 即中止
└── runs/<algo>/<时间戳>/  自动产物（config.yaml/env.json/train.log/history.csv/curves.png/ckpt/）
```

## 3. 一次训练的生命周期

```
run.py --algo mlp --override lr=0.01
  → config 三层合并（infra 默认 ← 算法 config ← override；数字/布尔自动识别，嵌套用点号 viz.live=False）
  → 创建 runs/<algo>/<时间戳>/，落 config.yaml + env.json，日志双写
  → spec.build_model → build_dataloaders → 训练循环：
      每 batch: unpack → to(device) → forward → loss → backward → optimizer.step
      每 epoch: evaluator 算验证指标 → 回调（best 追踪/存档/早停）→ history.csv 增量写
  → 结束: curves.png + ckpt/{last,best}.pt
```

## 4. 算法接入：MiniSpec 五处填空（`infra/minispec.py`）

```python
class MyNet(nn.Module): ...                    # ① 模型（从零实现手写 nn.Parameter）
def load_data(cfg): ...                        # ② 数据：下载/归一化决策全在此，返回 (train_ds, val_ds)
class MyAlgo(MiniSpec):
    name      = "my_algo"                      # ③ 与目录名一致（--algo 用）
    model     = MyNet                          # 必填；需按 cfg 组网时覆写 build_model(cfg)
    loss      = cross_entropy_fn               # 必填 loss(y_hat, y) -> 已 mean 标量
    datasets  = load_data                      # 必填
    optimizer = torch.optim.SGD                # 可选（默认 SGD，lr 走 cfg.lr）；也支持 (Adam,{...})/"adam" 字符串/自定义工厂
    metrics   = {"acc": fn}                    # 可选 fn(y_hat,y)->float，自动入曲线/history
    config    = {"epochs": 20, "lr": 0.01}     # 可选算法默认超参
```

逃生舱（全部可选）：`unpack`（batch 非 (X,y)）、`scheduler`、`get_callbacks`（算法专属回调，如 softmax 的预测图）、`training_step(batch, ctx)`（全权接管单批优化——GAN/多优化器/梯度累积；infra 退化为循环+回调+聚合）。MiniSpec 仍不够时直接实现 AlgoSpec 完整契约（见 `contract.py`）。

## 5. 变更纪律（重要，改动前必读）

- **冻结面**：trainer 主循环、checkpoint 语义、契约必选方法、回调事件名 → 改动须在 CHANGELOG.md 记一行（日期|改了什么|为什么）+ smoke_1–8 全量回归。
- **边缘面**：INFRA_DEFAULTS 新键、新内置回调、viz 样式、MiniSpec 新增可选声明 → 增量演进，CHANGELOG 酌情记。
- **依赖红线**：infra 任何模块不得 import algorithms.*（smoke_3 静态检查）。
- `infra.__version__`（0.3.0）= contract_version，随 checkpoint 落盘，续训跨版本显式告警。
- 目标是"给学习者省工程时间"，勿为对齐书引入工程特性。

## 6. 常用命令

```bash
# 环境：一律用 conda ai 环境 C:\Users\Administrator\anaconda3\envs\ai\python.exe（torch 2.6.0+cu124）
cd "C:\Users\Administrator\Desktop\ai 环境\d2l"

# 训练（脚本/CI 必须关 viz，否则等待手动关窗）
python run.py --algo mlp --override epochs=20 lr=0.01 num_workers=0 viz.live=False viz.wait_close=False
# 无显示器先 export MPLBACKEND=Agg

# 扫描/多seed：--sweep lr=0.01,0.1 --seeds 3（索引落 runs/<algo>/sweeps/*.csv，与 --resume 互斥）
# 续训：--resume runs/mlp/<stamp>/ckpt/last.pt --override epochs=40
# 早停/监控：patience=5 monitor=val_loss mode=min（patience=0 关闭）
# 全量冒烟：for f in tests/smoke_*.py; do python "$f"; done
# 其余命令（可视化/早停/确定性等全部参数）→ USAGE.md
```

## 7. 数据与环境事实

- 数据根：`d2l/data`（2026-09-22 起，原 `ai 环境/data` 已并入），FashionMNIST 在 `d2l/data/FashionMNIST/raw`，house-prices 在 `d2l/data/house-prices-advanced-regression-techniques`（算法层 `download=False`）。
- 硬件：RTX 4080 Laptop 12GB；device=auto 默认走 GPU；MLP 规模实测 ~0.7s/epoch。
- AMP：nn 优化器可 `amp=True`；**手写优化器必须 amp=False**（小模型常实测更慢，mlp/softmax 均默认 False）。
- Windows：中文+空格路径全程 pathlib、文件 IO 强制 utf-8、num_workers 建议 0、matplotlib 中文字体已处理。
- torch.compile 在 Windows+cu124 下 Triton 不可用，勿投入。

## 8. 性能基线（2026-09-18 优化后，softmax 规模）

数据预转换 TensorDataset（算法层 load_data 一次性做，代替逐样本 ToTensor 5.9s→0.5s）、单批 loss 只 `.item()` 一次、`non_blocking` 拷贝、evaluator inference_mode、Checkpointer 每轮只组装一次 payload。softmax epoch 稳态 6.5s→1.39s（~4.7×）。未做：TF32（CNN/MLP 章节可加）、num_workers>0（大数据集再开）。

## 9. 当前算法清单与状态

| 算法 | 内容 | 状态 |
|---|---|---|
| softmax | d2l 3.4–3.6 从零实现 + 预测图回调 | 范例，冒烟全绿 |
| mlp | d2l 4.1–4.3+4.6，784→256→10 ReLU，dropout_in/dropout_h 可调（默认 0） | 2026-09-19 实测 20 轮 val_acc 0.8115，曲线健康无过拟合 |
| logreg_iris | iris 对数几率回归（手写 sigmoid + 从零 BCE） | 可用 |
| house_price | Kaggle 房价从零线性回归 + 手写 AdamScratch，k 折 CV，fold=-1 出提交文件 | 可用（python run.py --algo house_price --sweep fold=0,1,2,3,4） |
| house_mlp | 房价 MLP，复用 house_price 特征件 | 可用 |
| lenet | d2l 6.6 全组件从零实现：卷积（im2col + 手写 backward，支持 stride/padding）、最大/平均汇聚（`pool=max\|avg`）、全连接、稳定交叉熵、手写 SGD（动量法）、IDX 手写解析 + /255 预处理 | 2026-09-22 实测 4s/epoch，3 轮 val_acc 0.724，续训状态等价；数值对拍在 tests/smoke_8（gradcheck） |
| linreg | 线性回归 | 学习者练习位 |
| fake | 合成数据 | 冒烟专用 |
