# d2l-framework

《动手学深度学习》（d2l）自学用的**极简可复用训练框架**：算法与工程严格分离——每学一个新算法只新增 `algorithms/<名字>/algo.py`（约 30 行纯算法代码），训练循环/断点/早停/日志/曲线等工程由 `infra/` 一次性代劳。不严格对齐书上的数字，只对齐算法结构性事实（层结构/激活/损失形式）。

健康度：fake / softmax 均以声明式 MiniSpec 接入并通过全量冒烟（smoke_1–8）；断点续训历史无缝衔接（smoke_3 实测）。

## 设计理念

1. **让算法研究者关注算法** —— 模型/损失/数据/指标是算法知识，必须写；DataLoader 装配/batch 解包/指标聚合/续训/早停是工程，一次写好后由框架代劳；
2. **双层契约** —— `infra/minispec.py` 声明式基类（五处填空，约 30 行）覆盖 90% 场景；`infra/contract.py` 完整契约（AlgoSpec）留给全控场景。MiniSpec 是 AlgoSpec 的糖衣，唯一注入点不变。逃生舱：可选 `training_step(batch, ctx)` 允许算法全权接管单批优化（GAN/多优化器/梯度累积），infra 退化为聚合与回调；
3. **依赖倒置** —— infra 只依赖 contract.py 定义的 AlgoSpec 契约，绝不 import 任何算法（有静态检查兜底）；
4. **决策归属分层** —— 归一化/指标/专属可视化是算法知识，走契约注入；训练循环/断点/早停/日志只写一次；
5. **断点续训 = 状态等价** —— checkpoint 含优化器/RNG/extras 全状态；无法等价时显式告警并写元数据，绝不静默退化；
6. **可复现是默认行为** —— 每次运行自动归档 config / 环境快照 / 日志 / 训练历史；
7. **核心冻结 + 边缘扩展** —— 冻结面改动须记 CHANGELOG 并全量回归，边缘面自由演进。

## 文档地图（各有分工，不重复）

| 文件 | 读者 | 内容 |
|---|---|---|
| OVERVIEW.md | AI 助手 / 想深入了解的人 | 全貌心智模型：目录地图、训练生命周期、接入签名、变更纪律、环境与性能事实 |
| USAGE.md | 所有人 | **命令唯一出处**：训练/扫描/续训/可视化/接入/冒烟全部命令 + 配置键速查 |
| 说明书.md | 使用者 | 怎么想、怎么用、错了看哪：心智模型 / 五组声明 / 学算法循环 / 排错 FAQ |
| CHANGELOG.md | 审计 | 冻结面变更账本（一行一改：日期 \| 改了什么 \| 为什么） |

## 快速开始

```bash
# 依赖：Python 3.11+, PyTorch 2.x, torchvision, matplotlib, pyyaml
# 本机用 conda ai 环境：C:\Users\Administrator\anaconda3\envs\ai\python.exe（torch 2.6.0+cu124）
python run.py --algo softmax --override epochs=2 num_workers=0 viz.live=False
```

全部命令（扫描/续训/早停/测试等）→ USAGE.md；每次运行自动落盘 `runs/<algo>/<时间戳>/`（config.yaml、env.json、train.log、history.csv、curves.png、ckpt/）。

## 当前算法

softmax（d2l 3.4–3.6 从零范例）· mlp · lenet（d2l 6.6 卷积/汇聚/全连接/损失/优化器/数据解析全组件从零）· linreg（练习位）· logreg_iris · house_price（Kaggle 房价从零线性回归 + 手写 Adam）· house_mlp（房价 MLP）· fake（冒烟专用）。逐个详情见 OVERVIEW.md §9。

**接入新算法**：复制 `algorithms/_template/` → 五处填空（约 30 行，全是算法知识）→ `python run.py --algo <算法名>`。填空细节见 说明书.md §2，命令见 USAGE.md §6–7。
