# d2l-framework

《动手学深度学习》（d2l）自学用的**极简可复用训练框架**：算法与工程基础设施严格分离，让算法研究者**只写算法，不写工程**，每学一个新算法只新增一个算法目录，不触碰 infra 冻结面。

接入验证：**fake / softmax** 均以声明式 MiniSpec 在库并通过全量冒烟（1/2/3/4/5）；softmax 为从零实现范例（d2l 3.4-3.6，手写 W/b + 稳定 log-softmax + 从零交叉熵 + 预测图回调）。linreg 留给学习者按模板自行练习。

健康度快照：fake 线性可分问题收敛正常，断点续训历史无缝衔接（冒烟 3 实测）。

## 设计理念

1. **让算法研究者关注算法** —— 模型/损失/数据/指标是算法知识，必须写；DataLoader 装配/batch 解包/指标聚合/续训/早停是工程，一次写好后由框架代劳；
2. **双层契约** —— `infra/minispec.py` 声明式基类（五处填空，约 30 行）覆盖 90% 场景；`infra/contract.py` 完整契约（AlgoSpec）留给全控场景。MiniSpec 是 AlgoSpec 的糖衣，唯一注入点不变；
3. **依赖倒置** —— infra 只依赖 contract.py 定义的 AlgoSpec 契约，绝不 import 任何算法（有静态检查兜底）；
4. **决策归属分层** —— 归一化/指标/专属可视化是算法知识，走契约注入；训练循环/断点/早停/日志只写一次；
5. **断点续训 = 状态等价** —— checkpoint 含优化器/RNG/extras 全状态；无法等价时显式告警并写元数据，绝不静默退化；
6. **可复现是默认行为** —— 每次运行自动归档 config / 环境快照 / 日志 / 训练历史；
7. **核心冻结 + 边缘扩展** —— 冻结面改动须记 CHANGELOG 并全量回归，边缘面自由演进（"基础设施零改动"的可达版本）。

## 目录结构

```
d2l/
├── run.py              # 唯一入口：--algo / --resume / --override
├── CHANGELOG.md        # 冻结面变更审计账本
├── USAGE.md            # 用法速查（训练/续训/接入/测试全命令）
├── infra/              # 基础设施（核心冻结 + 边缘扩展，变更见 CHANGELOG.md）
│   ├── contract.py     #   AlgoSpec 契约：唯一注入点 + 变更纪律
│   ├── minispec.py     #   声明式算法基类：只声明算法知识，工程代劳（边缘面新增）
│   ├── trainer.py      #   通用训练循环（只认契约，调度回调）
│   ├── config.py       #   两层配置：infra 默认 ← 算法默认 ← --override
│   ├── seed.py / logger.py / history.py / checkpoint.py
│   ├── callbacks.py    #   早停 / best 追踪 / 存档 / 曲线（全为可插拔回调）
│   └── data.py / evaluator.py / viz.py
├── algorithms/         # 算法层：每学一个算法新增一个目录
│   ├── _template/      #   接入模板：复制 → 五处填空（纯算法）→ 运行
│   ├── softmax/        #   softmax 从零实现（d2l 3.4-3.6，声明式范例 + 预测图回调）
│   └── fake/           #   合成数据测试算法（声明式接入的活样例，冒烟用）
├── tests/              # 冒烟测试：种子/断点/全流程/声明式契约 + 红线检查
└── runs/<algo>/<stamp>/  # 自动生成，含 ckpt 与全部产物（不入库）
```

## 快速开始

```bash
# 依赖：Python 3.11+, PyTorch 2.x, torchvision, matplotlib, pyyaml
python run.py --algo softmax --override epochs=2 num_workers=0 viz.live=False
python run.py --algo softmax --resume runs/softmax/<stamp>/ckpt/last.pt --override epochs=5

# 验证框架健康（5 个冒烟测试；无显示器/CI 环境先 export MPLBACKEND=Agg）
python tests/smoke_3_full_flow.py
```

每次运行自动落盘 `runs/<algo>/<时间戳>/`：config.yaml、env.json（机器可读环境快照）、train.log、history.csv、curves.png、ckpt/{last,best}.pt，以及算法层回调的产物（如有）。

## 接入新算法（三步，只填算法）

1. 复制 `algorithms/_template/` 为 `algorithms/<算法名>/`；
2. 五处填空（全是算法知识，约 30 行）：模型 nn.Module / 数据集构造 / 损失 / 指标 / 默认超参；
3. `python run.py --algo <算法名>`。

声明式声明：`name / model / loss / datasets` 必填，`optimizer / metrics / config / unpack / scheduler / get_callbacks` 可选（见 `infra/minispec.py`）。需要全部控制权（多优化器/非常规 batch/自定义装配）时，直接实现 AlgoSpec 完整契约：`default_config / build_model / build_loss / build_optimizer / build_dataloaders / unpack_batch / compute_metrics`（必选），`build_scheduler / get_callbacks`（可选）。

## Windows 兼容

多进程 spawn 保护、中文+空格路径全程 pathlib、文件 IO 强制 utf-8、matplotlib 中文字体——均已在冒烟测试中实测。
