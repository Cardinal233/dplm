# 简化版 PyTorch Lightning 训练流水线

本文档整理自 `byprot.training_pipeline` 的实现，旨在抽离出一套在日常研究
和快速迭代中更易上手的训练流程。新的 `byprot.simple_lightning_pipeline`
模块保留了原有流水线的核心能力，并针对常见的使用场景做了如下精简：

- **统一封装组件**：复用 `utils.common_pipeline` 的注册机制，自动实例化
  `LightningDataModule`、`LightningModule`、日志与回调。
- **精简启动入口**：暴露单一的 `train` 方法，默认负责超参数记录、
  checkpoint 解析以及可选的测试阶段。
- **保留可复用能力**：通过数据类 `PipelineComponents` 将关键对象打包，
  便于在 Notebook 或脚本中继续操作。

## 与原流水线的差异

| 功能点 | 原 `training_pipeline` | 简化版 `simple_lightning_pipeline` |
| --- | --- | --- |
| 入口函数 | `train(config)` | `train(config, ckpt_path=None, test_after_fit=False)` |
| 自动日志 | 支持 | 支持 |
| 自定义组件 | 基于 registry | 基于 registry |
| 断点续训 | 自动解析 `config.train` | 通过显式 `ckpt_path` 参数 |
| 结构复杂度 | 包含超参搜索、FSDP、测试入口等 | 聚焦训练主流程 |

## 快速上手

1. **准备配置**：保持与现有 Hydra/OmegaConf 配置一致。例如：

    ```yaml
    # configs/train.yaml
    datamodule:
      _target_: protein.datamodules.demo.DemoDataModule
      batch_size: 2

    model:
      _target_: protein.models.demo.DemoModel

    task:
      _target_: protein.tasks.demo.DemoTask
      model: ${model}

    trainer:
      _target_: pytorch_lightning.Trainer
      max_epochs: 5
      accelerator: gpu
      devices: 1
    ```

2. **在脚本或 Notebook 中调用**：

    ```python
    from omegaconf import OmegaConf
    from byprot.simple_lightning_pipeline import train

    cfg = OmegaConf.load("configs/train.yaml")
    trainer = train(cfg, ckpt_path=None, test_after_fit=True)
    ```

3. **复用组件**：如需进一步处理，可结合 `build_components` 获取原始对象：

    ```python
    from byprot.simple_lightning_pipeline import build_components

    components = build_components(cfg)
    components.trainer.validate(components.module, components.datamodule)
    ```

## 推荐的实验流程

1. **设定随机种子**：在配置中加入 `seed: 42`，可保证多卡/多进程环境下的可复现性。
2. **控制回调与日志**：在配置中保留需要的回调（如 EarlyStopping）和日志记录器
   定义，简化版流水线会原样实例化这些对象。
3. **断点续训**：当希望从历史 checkpoint 恢复时，直接传入相对路径：

    ```python
    train(cfg, ckpt_path="checkpoints/last.ckpt")
    ```

4. **测试阶段**：`test_after_fit=True` 可在训练完成后立即执行测试，避免重复
   编写测试脚本。

通过上述简化，研究人员可在保留原有工程生态的同时，以更少的代码复用
Lightning 训练能力，实现更加顺畅的实验迭代。
