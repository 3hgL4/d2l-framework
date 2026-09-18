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
