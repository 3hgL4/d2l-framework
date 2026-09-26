# CHANGELOG（一行一改：日期 | 改了什么 | 为什么）

格式约定：冻结面改动（trainer 主循环 / checkpoint 语义 / 契约必选方法 / 回调事件名）
必须记一行；边缘面改动酌情合并记。放根目录而非 infra/ 内，因 run.py 与
算法层接口同属审计范围。

- 2026-09-18 | v1 交付：infra 11 模块 + run.py + algorithms/{softmax,fake} + tests 4 冒烟 | 建立"核心冻结"基线
- 2026-09-18 | history.py：构造时不再删旧文件 | 全量重写天然覆盖，构造不应有删除副作用
- 2026-09-18 | trainer.py：新增可选契约 get_callbacks(cfg)（边缘面） | 算法专属可视化注入，infra 不感知算法
- 2026-09-18 | algorithms/softmax/algo.py：sys.path 引导 + __main__ 自检 | 从任意位置可导入，直接运行给自检而非报错
- 2026-09-18 | config.py/run.py：砍掉 config.yaml 覆盖层，build_config 改双参 | 三层中该层是空壳，默认值与调参已有归属
- 2026-09-18 | trainer.py：优化器无 state_dict 的续训退化路径加 WARNING | 静默退化会让续训不等价，宁可吵不可错
- 2026-09-18 | contract.py：写入"核心冻结 + 边缘扩展"变更纪律 | 替代"infra 永零改动"的不可达目标
- 2026-09-18 | 新增本 CHANGELOG | 让冻结纪律可审计，不依赖记忆
- 2026-09-18 | trainer.py：续训退化时 extras 写入 resumed_degraded_optimizer，随 checkpoint 落盘 | 告警只进日志，元数据进状态才可机器审计
- 2026-09-19 | config.py：build_config 对未知 override 键（含 dict 覆盖标量）发 UserWarning（边缘面） | 拼错键（如 epohs=5）此前静默新增死键，对应功能默默失效，违反 fail-fast
- 2026-09-19 | run.py：新增 --list（扫描 algorithms/ 目录约定，下划线开头跳过）；缺 --algo 时报错并列出可用算法；load_spec 校验 spec.name 与目录名一致（不一致告警） | 可发现性 + 约定校验：checkpoint 元数据/日志与 run 目录对得上
- 2026-09-19 | run.py：新增 --sweep key=v1,v2,...（笛卡尔积）与 --seeds N；扫描模式强制关 viz 窗口，索引落 runs/<algo>/sweeps/<stamp>.csv 并按最优值排名；与 --resume 互斥 | 批量实验是当前最大功能缺口：调参/多 seed 此前需手动 N 次 + 肉眼比对 history.csv
- 2026-09-19 | run.py：单跑与扫描共用 run_once（配置合成/三件套/训练） | 两条路径产物资质一致，避免漂移
- 2026-09-19 | 新增 tests/smoke_7_config_units.py（单元级：deep_merge/parse_overrides/未知键告警/AttrDict/python+numpy RNG 恢复） | 流程冒烟之外补地基纯函数边界，改动 config 即回归
- 2026-09-18 | contract.py：注明 build_model 必须返回 nn.Module | infra 靠 parameters()/state_dict()/to(device) 枚举张量，显式成文防误用
- 2026-09-18 | viz/callbacks/trainer/config：新增 viz.wait_close（默认 False） | 训练结束不再强制关窗，用户可 --override viz.wait_close=True 手动关图
- 2026-09-18 | config.py：viz.wait_close 默认值 False -> True | 用户交互式使用为主，手动关窗应为默认；脚本场景显式传 False
- 2026-09-18 | config.py/run.py/tests：env.txt -> env.json（机器可读） | 与 config.yaml 同格式，两次实验环境可直接 diff
- 2026-09-18 | README.md：统一"核心冻结"措辞、补测试命令/runs 树项/健康度快照 | 消除"零改动"残留话术与变更纪律的自相矛盾（外部评审发现）
- 2026-09-18 | infra/minispec.py：新增 MiniSpec 声明式算法基类（边缘面，新增模块，冻结面零改动） | 让算法研究者只写算法（模型/损失/数据/指标/超参），DataLoader 装配/batch 解包/指标聚合等工程由适配器代劳
- 2026-09-18 | tests/fake_algo.py：FakeAlgo 由手写契约改为 MiniSpec 声明式（行为等价） | 冒烟 2/3 升级为声明式接入的回归验证；声明错误在类定义时报错
- 2026-09-18 | algorithms/_template/algo.py：模板改写为 MiniSpec 五处填空（约 100 行 -> 30 行算法代码） | 新算法接入的工程代码量降为零，工程知识不再出现在算法文件
- 2026-09-18 | contract.py：docstring 注明 MiniSpec 声明式入口 | 唯一注入点不变，文档与实际接入方式一致
- 2026-09-18 | infra/__init__.py：模块清单加 minispec，版本 0.1.0 -> 0.2.0 | 新增边缘面能力，向后兼容
- 2026-09-18 | tests/smoke_5_minispec.py：新增 MiniSpec 适配器冒烟（11 项） | 声明校验/配置合成/优化器三形态/损失三形态/装载/聚合/unpack/scheduler/callbacks 全覆盖
- 2026-09-18 | README/USAGE：双层契约说明 + 目录树同步磁盘实际（softmax/linreg 已删） | 修正文档漂移；smoke_4 标注需重建 softmax 后可用
- 2026-09-18 | algorithms/softmax/：以 MiniSpec 声明式重建 softmax 从零实现（d2l 3.4-3.6：稳定 log-softmax + 从零交叉熵 + 手写 W/b + PredictionPlotter 回调） | 用户要求的声明式接入工作流范例；smoke_4 恢复全绿（2 轮 val_acc 0.804，续训/预测图断言全过）
- 2026-09-18 | algorithms/_template/algo.py：升级为通用模板——固定六段骨架 + 算法家族差异速查表（分类/回归/手写损失）+ 回归/调度/解包/cfg组网变体注释 | 回答"任何 algo.py 结构是否一样"：一样，换算法只换 ②③④⑥ 内容
- 2026-09-18 | algorithms/logreg_iris/：新增 logistic 回归（iris 二分类，手写 sigmoid + 从零 BCE，同一 random_state=666 split） | 机器学习课程第二次作业的框架侧对照实现，MiniSpec 第四个算法接入
- 2026-09-18 | trainer.py：主循环新增可选契约 training_step(batch, ctx)（冻结面改动，已全量回归 smoke_1-6） | 算法全权接管单批优化（GAN/多优化器/梯度累积），infra 退化为聚合与回调；y_hat/y 为 None 时该批跳过指标
- 2026-09-18 | callbacks.py：TrainContext 新增 device 字段，fit() 传入 | 兑现 training_step 契约"自管 batch 搬运（ctx.device）"；ctx 已有 model/optimizer/scaler，训练态经 ctx 拿回
- 2026-09-18 | trainer.py：checkpoint 新增 contract_version 键，续训版本不一致显式告警（冻结面 checkpoint 语义扩展） | 契约可版本化：跨版本续训宁可吵不可静默错
- 2026-09-18 | contract.py/minispec.py/README：training_step 契约语义成文 | 文档与实现同步，防漂移
- 2026-09-18 | infra/__init__.py：版本 0.2.0 -> 0.3.0 | 上述冻结面扩展一并升版
- 2026-09-18 | tests/smoke_6_custom_step.py：新增自定义步冒烟（10 项） | 手写 SGD 绕开 optimizer、跳指标分支、contract_version 落盘、续训保留自定义步、跨版本告警
- 2026-09-18 | trainer.py：单批 float(loss) 只算一次并复用；batch 搬运加 non_blocking=True（冻结面，已全量回归 smoke_1-6） | 重复 .item() 即重复 D2H 同步；pin_memory 的异步拷贝此前从未兑现（实测 softmax 负载 2.27 -> 2.10 ms/batch）
- 2026-09-18 | evaluator.py：@torch.no_grad() -> @torch.inference_mode()；autocast 提到循环外；batch 搬运加 non_blocking=True（已全量回归） | 验证输出从不进 autograd，语义等价；省每批上下文进出与同步拷贝
- 2026-09-18 | callbacks.py：Checkpointer 每轮只组装一次 checkpoint payload | is_best 轮此前 capture_rng + torch.save 各执行两次；同轮 last/best 内容严格一致反而更正确
- 2026-09-18 | data.py：make_loader 新增可选 pin_memory 参数，make_dataloaders 按目标设备判断（device=cpu 不再分配锁页内存） | 锁页内存在 cpu 路径只有开销；参数缺省保持旧行为
- 2026-09-18 | algorithms/softmax/algo.py：load_data 一次性预转换 TensorDataset（等价 ToTensor，含 (N,1,28,28) 形状）；config 增加 amp=False（均算法层，infra 零改动） | ToTensor 逐样本转换每 epoch 重复执行，实测 5.9s -> 0.5s（12x）；784x10 小模型 AMP 为负收益（实测 -23%），大模型章节可删
- 2026-09-22 | algorithms/lenet/algo.py：build_model 移除 _check_against_torch/_print_shapes 调用，自检与形状推演改挂 __main__（算法层修正） | 自检内 torch.manual_seed(0) 在 trainer 的 set_seed(cfg.seed) 之后执行，抹掉 cfg.seed 并破坏 --seeds 扫描/复现语义；且每次训练重复跑对拍+打印
- 2026-09-22 | algorithms/{mlp,softmax,_template,house_price}/algo.py：DATA_ROOT/DATA_DIR 由 ROOT.parent/data 改为 ROOT/data；OVERVIEW.md §7 数据根同步 | 数据目录已迁至 d2l/data（原 ai 环境/data 不存在），旧路径下 FashionMNIST/house-prices 加载必失败；lenet 的 ROOT/data 为正确写法，据此统一
- 2026-09-22 | 文档去重分工：README.md 重写为门面+理念+文档导航（删过时目录树/命令/接入细节），OVERVIEW.md 补齐算法清单（house_price/house_mlp）与文档分工句，说明书.md §3 命令块改为指路 USAGE.md | 目录树/接入方法/常用命令此前在 4 份文档重复维护且 README 已漂移（冒烟数、算法目录过时），单一出处防漂移
- 2026-09-22 | algorithms/lenet/algo.py：删除 __main__ 自检块，数值对拍整体迁至 tests/smoke_8_lenet.py（标量对拍升级为双精度 gradcheck，覆盖 conv/maxpool/avgpool 解析梯度，另含形状推演与真实训练+续训冒烟） | 用户指出自检属工程代码，算法层按纪律只含算法知识，测试归 tests/；gradcheck 比标量对拍更强（逐张量有限差分验证 dW/db/dx）
- 2026-09-22 | algorithms/lenet/：新增 LeNet 全组件从零实现（d2l 6.6）——Conv2dScratch（im2col+手写 backward/col2im，支持 kernel/stride/padding）、MaxPool/AvgPool2dScratch（argmax 掩码/均分梯度 backward）、LinearScratch、手写稳定 sigmoid 与 log-softmax 交叉熵、SGDScratch（动量法，含 state_dict/load_state_dict 续训等价）、IDX 手写解析器（struct+gzip，不经 torchvision）；`__main__` 形状推演 + conv 标量对拍自检（算法层，infra 零改动） | 用户要求卷积/汇聚/全连接/损失/优化器/数据导入/预处理/padding/stride 全部手写；实测 3 轮 val_acc 0.724（书值量级）、续训无退化告警、pool=max 可切换
- 2026-09-22 | algorithms/lenet/algo.py：修复 _im2col as_strided 维序步长错配（(kh,kw,oh,ow) 视图误配 (oh,ow,kh,kw) 步长，窗口内容错位）；_pool_windows 同型问题改为 6 维视图再 reshape | conv 标量对拍失败暴露；教训：as_strided 的 shape 维序必须与 strides 配对逐一对应，滑窗维度拆成独立两维（窗口内偏移步长不变 × 窗口滑动步长乘 s）
- 2026-09-22 | algorithms/lenet/algo.py：算法本体缺失后重写（数据流顺序：IDX 手写解析+/255 预处理 → im2col 卷积 _Conv2dFunc(pad/stride+手写 backward) → 最大/平均汇聚 → 全连接 → LeNetScratch → 稳定交叉熵 → 手写 SGD 动量法 → MiniSpec 注册，无注释）；修 backward 尺寸裁剪用错 padding 后 H/W 的 bug；smoke_8 1 轮 val_acc 带宽 0.6→0.85（实测 0.616，与文档 3 轮 0.724 轨迹一致）；load_data 改 np.array 拷贝消除只读 NumPy 警告 | 文档/CHANGELOG/smoke_8 在而 algo.py 不在（此前版本丢失）；gradcheck 全过、10 轮 best val_acc 0.8654
- 2026-09-22 | 新增 infra/datasets.py：IDX 二进制格式手写读取器（read_idx_images/read_idx_labels，含 magic 校验）自 algorithms/lenet/algo.py 迁入，OVERVIEW 目录地图同步 | IDX 解析是纯数据格式转换（工程件，torchvision.datasets 的自带替代），无算法决策，softmax/lenet 等数据路径可复用；算法层保留路径选择与 /255 归一化等数据决策；边缘面增量演进，契约零改动，全量冒烟回归
- 2026-09-23 | infra/datasets.py 新增 load_fashion_mnist(cfg)，algorithms/lenet/algo.py 删除 load_data/DATA_ROOT 改为声明 datasets=load_fashion_mnist | 用户决策：数据装载属可复用工程件下沉 infra（手写算子/损失/优化器保留算法层，教学价值优先；算子库待全部网络学完再统一收编）| 边缘面增量，契约零改动
- 2026-09-26 | 新增 tools/algo_report.py（边缘工具，不属 infra/算法层）：读取 algorithms/<algo>/algo.py 头部"我的实验笔记"注释块 + runs/<algo>/<stamp> 既有产物（config/log/history/env），生成 reports/<algo>/<stamp>.md 单次报告与 compare.md 多 run 对比表；--algo/--run/--compare 三个子命令；不 import torch、契约零改动 | 用户需要"笔记 + 关键数据 → 指定算法报告"的结论层留痕；报告可再生成故 reports/ 不入库；已用 mlp 全部 6 个 run 实测
- 2026-09-26 | .gitignore 补 reports/；algorithms/mlp/algo.py 头部加注释模板（目标/假设/结论占位，内容待用户填） | 报告为可再生工件；模板示范 header 注释块的收录机制
- 2026-09-26 | infra/trainer.py：fit() 组装 optimizer/dataloader 后新增 [优化器]（类名+lr）与 [数据]（train/val 样本数、batch_size、batches/epoch）两行 INFO，双写 train.log | 报告与审计需要优化器型号与数据规模溯源——此前 config 只记 lr 无优化器类名，数据规模完全未落盘；契约零改动，全量冒烟 1–8 回归通过（conda ai 环境，1m39s）
- 2026-09-26 | tools/algo_report.py：单次报告收录数据规模与优化器行；compare 表新增 opt 列（2026-09-26 15:00 前的旧 run 无该日志行，显示 ?） | 与 infra 新日志行配套
