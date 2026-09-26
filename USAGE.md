# 用法速查（d2l-framework）

所有命令均在项目根 `d2l/` 下执行。产物统一落 `runs/<算法>/<时间戳>/`。

## 1. 训练

```bash
# 基本训练：跑 algorithms/softmax 的默认配置
python run.py --algo softmax

# 临时覆盖参数：空格分隔多个 key=value，数字/布尔自动识别
python run.py --algo softmax --override epochs=30 lr=0.3 num_workers=4

# 覆盖嵌套配置：用点号路径
python run.py --algo softmax --override viz.live=False ckpt.save_best=False

# 换算法同理
python run.py --algo linreg
```

## 1.5 参数扫描 / 多 seed（批量实验）

```bash
# 扫描 lr 两个取值（每值一次完整训练，各自独立 run 目录）
python run.py --algo mlp --sweep lr=0.01,0.1

# 多参数 = 笛卡尔积（2 lr × 2 batch_size = 4 次）
python run.py --algo mlp --sweep lr=0.01,0.1 batch_size=128,256

# 每个组合再连跑 3 个 seed（42/43/44），共 12 次
python run.py --algo mlp --sweep lr=0.01,0.1 --seeds 3

# 扫描可与 --override 叠加（扫描值优先）
python run.py --algo mlp --sweep lr=0.01,0.1 --override epochs=30 patience=5

# 只做多 seed 重复（同一配置跑 5 次，看方差）
python run.py --algo softmax --seeds 5
```

说明：
- 索引落 `runs/<算法>/sweeps/<时间戳>.csv`（组合、seed、run_dir、最优值、实际轮数、耗时），结束打印按最优值排名；
- 扫描模式强制 `viz.live=False / wait_close=False`（否则 N 个窗口弹出）；看曲线请对感兴趣的 run_dir 单跑或续训；
- `--sweep/--seeds` 与 `--resume` 互斥。

## 2. 断点续训

```bash
# 从上次中断处继续（历史/最优记录/RNG 无缝衔接）
python run.py --algo softmax --resume runs/softmax/20260918-154519/ckpt/last.pt

# 续训同时提高目标轮数
python run.py --algo softmax --resume runs/softmax/<stamp>/ckpt/last.pt --override epochs=60
```

## 3. 可视化控制

```bash
# 训练结束后曲线窗口保持打开，由你手动关闭（默认行为）
python run.py --algo softmax

# 脚本/批处理场景恢复"结束自动关窗退出"
python run.py --algo softmax --override viz.wait_close=False

# 关闭实时刷新窗口（只存 curves.png 到 run 目录）
python run.py --algo softmax --override viz.live=False

# 连存盘也关（更快，无任何图）
python run.py --algo softmax --override viz.live=False viz.save=False

# 关闭算法专属可视化回调（如自定义预测图，按算法自身配置键而定）
python run.py --algo mlp --override predict_demo=False
```

## 4. 可复现 / 设备 / 性能

```bash
# 完全可复现模式：牺牲速度换确定性（cudnn deterministic）
python run.py --algo softmax --override deterministic=True seed=42

# 强制设备（默认 auto：有 GPU 用 GPU，无则 CPU）
python run.py --algo linreg --override device=cpu

# 数据加载进程数（Windows spawn 模式，0 最稳；>0 需 dataset 可 pickle）
python run.py --algo softmax --override num_workers=0

# 混合精度（手写优化器必须保持 False，nn 优化器可 True 提速）
python run.py --algo mlp --override amp=True
```

## 5. 停止策略 / 训练细节

```bash
# 早停：监控 val_loss，8 轮无改善即停（0 = 关闭早停）
python run.py --algo softmax --override patience=8 min_delta=0.0001

# 换监控指标与方向（如监控 val_acc 越大越好）
python run.py --algo mlp --override monitor=val_acc mode=max patience=5

# 梯度裁剪（>0 生效，防梯度爆炸）
python run.py --algo mlp --override grad_clip=1.0

# batch 级细节写入日志文件（默认静默，只打 epoch 摘要行）
python run.py --algo softmax --override verbose=True

# 每 5 轮打一行摘要（默认 1，最后一轮必打）
python run.py --algo softmax --override log_every=5
```

## 6. 接入新算法（三步，infra 零改动，只填算法）

```bash
# 1. 复制模板目录并改名（目录名即 --algo 的值）
#    algorithms/_template/  ->  algorithms/mlp/
# 2. 五处填空（全是算法知识，约 30 行）：
#    模型 nn.Module / 数据集构造 load_data / 损失 loss / 指标 metrics / 默认超参 config
#    声明式基类见 infra/minispec.py；优化器 optimizer 可选（默认 SGD，lr 走 --override）
# 3. 运行（先小轮数验证，再正式训练）
python run.py --algo mlp --override epochs=3 num_workers=0
```

## 7. 进阶：完整契约接入（需要全部控制权时）

```bash
# MiniSpec 不够用时（多优化器 / 非常规 batch / 自定义装配逻辑），
# 直接实现 contract.AlgoSpec：default_config / build_model / build_loss /
# build_optimizer / build_dataloaders / unpack_batch / compute_metrics 必选，
# build_scheduler / get_callbacks 可选，duck-typing 无需继承（见 infra/contract.py）
```

## 7.5 算法实验报告（tools/algo_report.py，结论层留痕）

```bash
# 最新一次 run → 单次报告 reports/<算法>/<时间戳>.md
python tools/algo_report.py --algo mlp

# 指定某次 run
python tools/algo_report.py --algo mlp --run 20260922-200605

# 该算法全部 runs 汇成对比表 → reports/<算法>/compare.md
python tools/algo_report.py --algo mlp --compare
```

说明：
- **报告内容**：algo.py 头部"我的实验笔记"注释块 + 关键指标（最终/best val_loss·val_acc、train/val loss 差及过拟合提示）+ 超参（算法自定义与 infra 分列）+ 训练历史表 + 日志尾部 + 曲线图链接；
- **写笔记的位置**：`algorithms/<算法>/algo.py` 最顶部的连续 `#` 注释块（模板：目标/假设/结论），报告自动原样收录——假设在跑之前写，结论跑完回填；
- **compare 表为动态列**：只列出各 run 之间取值有差异的配置键（差异原因候选），完全相同的参数折成一行"所有 run 固定: ..."；早期 run 缺某键显示 `?`；
- 报告可随时再生成（纯读 runs/ 产物），故 `reports/` 不入库；不 import torch，任何 Python ≥3.10 可跑。

## 8. 冒烟测试（全量回归，改 infra 后必跑）

```bash
python tests/smoke_1_seed_config.py            # 种子复现 / 配置合并 / 目录 / 日志
python tests/smoke_2_history_ckpt_callbacks.py # 早停触发 / 断点状态等价
python tests/smoke_3_full_flow.py              # 全流程 + spawn 多进程 + infra 红线检查
python tests/smoke_4_softmax.py                # 真实算法接入 + 续训 + 预测图（softmax 从零实现）
python tests/smoke_5_minispec.py               # MiniSpec 声明式契约正确性（校验/配置/优化器/损失/装载）
python tests/smoke_6_custom_step.py            # training_step 接管 + 契约版本化告警
python tests/smoke_7_config_units.py           # config 纯函数边界 + 多源 RNG 恢复（单元级）
python tests/smoke_8_lenet.py                  # LeNet 全组件从零：gradcheck 数值对拍 + 训练/续训
```

## 9. 产物说明（每次运行自动生成）

```text
runs/<算法>/<时间戳>/
├── config.yaml     # 最终生效配置（三层合并结果，存档1）
├── env.json        # 环境/版本/GPU 快照（存档2，机器可读可 diff）
├── train.log       # 双写日志（控制台+文件，utf-8）
├── history.csv     # 逐 epoch 训练历史（曲线图的数据源）
├── curves.png      # 全指标曲线（轮数/损失/各准确率）
├── predictions.png # 预测展示图（算法层回调产物，如有）
└── ckpt/
    ├── last.pt     # 最新断点（--resume 用它）
    └── best.pt     # 最优轮断点（预测图默认用它）
```

## 10. 常用配置键速查

| 键 | 默认 | 说明 |
|---|---|---|
| seed / deterministic | 42 / False | 种子与 cudnn 确定性 |
| device / amp | auto / True | 设备与混合精度（手写优化器须 amp=False） |
| epochs / lr / batch_size | 10 / 0.1 / 128 | 轮数（infra 默认）；lr/batch_size 为 MiniSpec 声明默认，可在 config 声明中覆盖 |
| num_workers | 0 | DataLoader 进程数（Windows 建议 0 或 4） |
| monitor / mode / patience / min_delta | val_loss / min / 0 / 0 | 早停策略（patience=0 关闭） |
| grad_clip | 0 | 梯度范数裁剪阈值 |
| log_every / verbose | 1 / False | 摘要行频率 / batch 级细节 |
| viz.live / viz.save / viz.wait_close | True / True / True | 实时窗口 / 存盘 / 结束后等手动关窗（False=自动退出） |
| ckpt.save_last / ckpt.save_best | True / True | 断点保存 |
