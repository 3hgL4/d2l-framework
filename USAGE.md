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

## 2. 断点续训

```bash
# 从上次中断处继续（历史/最优记录/RNG 无缝衔接）
python run.py --algo softmax --resume runs/softmax/20260918-154519/ckpt/last.pt

# 续训同时提高目标轮数
python run.py --algo softmax --resume runs/softmax/<stamp>/ckpt/last.pt --override epochs=60
```

## 3. 可视化控制

```bash
# 训练结束后曲线窗口保持打开，由你手动关闭（默认自动关）
python run.py --algo softmax --override viz.wait_close=True

# 关闭实时刷新窗口（只存 curves.png 到 run 目录）
python run.py --algo softmax --override viz.live=False

# 连存盘也关（更快，无任何图）
python run.py --algo softmax --override viz.live=False viz.save=False

# 关闭算法专属预测图（softmax 的 predictions.png）
python run.py --algo softmax --override predict_demo=False
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

## 6. 接入新算法（三步，infra 零改动）

```bash
# 1. 复制模板目录并改名（目录名即 --algo 的值）
#    algorithms/_template/  ->  algorithms/mlp/
# 2. 替换 algo.py 里的 TODO（模型/损失/优化器/数据/指标），末尾保留 SPEC = XxxAlgo()
# 3. 运行（先小轮数验证，再正式训练）
python run.py --algo mlp --override epochs=3 num_workers=0
```

## 7. 单文件自检（不训练，不碰数据集）

```bash
# 在任意目录直接运行 algo.py：合成数据走一遍前向/反向，验证代码能跑
python algorithms/softmax/algo.py
```

## 8. 冒烟测试（全量回归，改 infra 后必跑）

```bash
python tests/smoke_1_seed_config.py            # 种子复现 / 配置合并 / 目录 / 日志
python tests/smoke_2_history_ckpt_callbacks.py # 早停触发 / 断点状态等价
python tests/smoke_3_full_flow.py              # 全流程 + spawn 多进程 + infra 红线检查
python tests/smoke_4_softmax.py                # 真实算法接入 + 续训 + 预测图
```

## 9. 产物说明（每次运行自动生成）

```text
runs/<算法>/<时间戳>/
├── config.yaml     # 最终生效配置（三层合并结果，存档1）
├── env.txt         # 环境/版本/GPU 快照（存档2）
├── train.log       # 双写日志（控制台+文件，utf-8）
├── history.csv     # 逐 epoch 训练历史（曲线图的数据源）
├── curves.png      # 全指标曲线（轮数/损失/各准确率）
├── predictions.png # 预测展示图（softmax 专属回调，绿对红错）
└── ckpt/
    ├── last.pt     # 最新断点（--resume 用它）
    └── best.pt     # 最优轮断点（预测图默认用它）
```

## 10. 常用配置键速查

| 键 | 默认 | 说明 |
|---|---|---|
| seed / deterministic | 42 / False | 种子与 cudnn 确定性 |
| device / amp | auto / True | 设备与混合精度（手写优化器须 amp=False） |
| epochs / lr / batch_size | 10 / 各算法定 | 训练轮数与算法超参 |
| num_workers | 0 | DataLoader 进程数（Windows 建议 0 或 4） |
| monitor / mode / patience / min_delta | val_loss / min / 0 / 0 | 早停策略（patience=0 关闭） |
| grad_clip | 0 | 梯度范数裁剪阈值 |
| log_every / verbose | 1 / False | 摘要行频率 / batch 级细节 |
| viz.live / viz.save / viz.wait_close | True / True / False | 实时窗口 / 存盘 / 结束后等手动关窗 |
| ckpt.save_last / ckpt.save_best | True / True | 断点保存 |
