"""d2l 工程基础设施（算法无关层）。

依赖红线：本包任何模块不得 import algorithms.*（冒烟测试静态检查）。

模块职责：
  seed        随机种子（全链路可复现）
  config      配置合成 / 运行目录 / 环境快照
  logger      控制台+文件双写日志
  history     训练历史 CSV（逐 epoch 增量落盘）
  checkpoint  断点全状态（权重/优化器/RNG/extras）
  callbacks   生命周期钩子与内置回调（早停/最佳追踪/存档/曲线）
  contract    算法契约（AlgoSpec Protocol，唯一注入点）
  minispec    声明式算法基类（AlgoSpec 糖衣：只声明算法知识，工程代劳）
  data        DataLoader 工厂（Windows 多进程安全）
  evaluator   验证集评估执行器
  viz         训练曲线实时刷新与存盘
  trainer     通用训练循环
"""
__version__ = "0.2.0"
