# d2l-framework

《动手学深度学习》（d2l）自学用的**极简可复用训练框架**：算法与工程基础设施严格分离，每学一个新算法只新增一个算法目录，不触碰 infra 冻结面。

已通过 **softmax / linreg / fake** 三个算法接入验证，其中 linreg 接入期间 infra 冻结面**零改动**（全部变更均可查 [CHANGELOG.md](CHANGELOG.md)）。

健康度快照：softmax 在 FashionMNIST 上 3 轮即达 val_acc ≈ 0.82（d2l 教材水准）；linreg 合成数据收敛，学到的 w/b 逼近真值 `[2.0, -3.4] / 4.2`，与 d2l 3.2 节行为一致。

## 设计理念

1. **依赖倒置** —— infra 只依赖 `infra/contract.py` 定义的 `AlgoSpec` 契约，绝不 import 任何算法（有静态检查兜底）；
2. **决策归属分层** —— 归一化/指标/专属可视化是算法知识，走契约注入；训练循环/断点/早停/日志只写一次；
3. **断点续训 = 状态等价** —— checkpoint 含优化器/RNG/extras 全状态；无法等价时显式告警并写元数据，绝不静默退化；
4. **可复现是默认行为** —— 每次运行自动归档 config / 环境快照 / 日志 / 训练历史；
5. **核心冻结 + 边缘扩展** —— 冻结面改动须记 CHANGELOG 并全量回归，边缘面自由演进（"基础设施零改动"的可达版本）。

## 目录结构

```
d2l/
├── run.py              # 唯一入口：--algo / --resume / --override
├── CHANGELOG.md        # 冻结面变更审计账本
├── USAGE.md            # 用法速查（训练/续训/接入/测试全命令）
├── infra/              # 基础设施（核心冻结 + 边缘扩展，变更见 CHANGELOG.md）
│   ├── contract.py     #   AlgoSpec 契约：唯一注入点 + 变更纪律
│   ├── trainer.py      #   通用训练循环（只认契约，调度回调）
│   ├── config.py       #   两层配置：infra 默认 ← 算法默认 ← --override
│   ├── seed.py / logger.py / history.py / checkpoint.py
│   ├── callbacks.py    #   早停 / best 追踪 / 存档 / 曲线（全为可插拔回调）
│   └── data.py / evaluator.py / viz.py
├── algorithms/         # 算法层：每学一个算法新增一个目录
│   ├── _template/      #   接入模板：复制 → 替换 TODO → 运行
│   ├── softmax/        #   softmax 回归（FashionMNIST，从零实现）
│   ├── linreg/         #   线性回归（d2l 3.2 七步，合成数据）
│   └── fake/           #   合成数据测试算法（冒烟用）
├── tests/              # 冒烟测试：种子/断点/全流程/真实接入 + 红线检查
└── runs/<algo>/<stamp>/  # 自动生成，含 ckpt 与全部产物（不入库）
```

## 快速开始

```bash
# 依赖：Python 3.11+, PyTorch 2.x, torchvision, matplotlib, pyyaml
python run.py --algo softmax --override epochs=3 num_workers=0
python run.py --algo linreg
python run.py --algo softmax --resume runs/softmax/<stamp>/ckpt/last.pt

# 验证框架健康（4 个冒烟测试；无显示器/CI 环境先 export MPLBACKEND=Agg）
python tests/smoke_3_full_flow.py
```

每次运行自动落盘 `runs/<algo>/<时间戳>/`：config.yaml、env.json（机器可读环境快照）、train.log、history.csv、curves.png、predictions.png（算法层回调）、ckpt/{last,best}.pt。

## 接入新算法（三步）

1. 复制 `algorithms/_template/` 为 `algorithms/<算法名>/`；
2. 替换 TODO：模型 / 损失 / 优化器 / 数据加载 / 指标（约 60–120 行）；
3. `python run.py --algo <算法名>`。

契约方法：`default_config / build_model / build_loss / build_optimizer / build_dataloaders / unpack_batch / compute_metrics`（必选），`build_scheduler / get_callbacks`（可选）。

## Windows 兼容

多进程 spawn 保护、中文+空格路径全程 pathlib、文件 IO 强制 utf-8、matplotlib 中文字体——均已在冒烟测试中实测。
