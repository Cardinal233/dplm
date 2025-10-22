# PyTorch Lightning 训练流水线总览

本文结合 `byprot.training_pipeline` 与 `byprot.simple_lightning_pipeline`
两个实现，对现有的 Lightning 训练流水线进行系统性梳理，总结各模块的
职责、配置入口与常见扩展方式。通过本文档，读者可以快速理解 dplm 项目
中的训练脚手架，并在需要时选用完整或精简的实现。

## 1. 核心流程概览

### 1.1 全量流水线

`byprot.training_pipeline.train` 是生产环境使用的主流程，主要步骤如下：

1. **基础设置**：根据配置调用 `utils.extras` 应用日志、配置树打印等功能，
   并通过 `seed_everything` 设置随机种子。
2. **组件构建**：复用 `utils.common_pipeline` 实例化数据模块、任务模块、
   日志记录器与回调，内部依赖注册表保证类名解析稳定。
3. **Trainer 初始化**：通过 `hydra.utils.instantiate(config.trainer, …)`
   构造 `pytorch_lightning.Trainer`，可以在配置中指定 FSDP 等策略。
4. **训练与恢复**：自动解析 `config.train.ckpt_path`，支持断点续训；随后
   调用 `trainer.fit` 执行训练，必要时根据 `config.test` 运行测试阶段。
5. **收尾处理**：通过 `utils.finish` 关闭日志资源，并在成功训练后输出最
   优 checkpoint 路径，供后续评估或部署使用。

### 1.2 精简流水线

`byprot.simple_lightning_pipeline.train` 面向研究与原型场景，保留上述
流程的关键节点，同时做出如下裁剪：

- 以显式函数参数 `ckpt_path`、`test_after_fit` 控制断点续训与测试阶段；
- 将 Trainer、数据模块、任务模块等通过 `PipelineComponents` 一次性返回，
  方便在 Notebook 或脚本中继续复用；
- 默认调用 `utils.extras` 与 `utils.log_hyperparameters`，确保实验记录
  与正式环境一致。

## 2. 配置体系与注册机制

### 2.1 Hydra 配置约定

训练脚手架遵循 Hydra/OmegaConf 规范，核心配置键包括：

- `datamodule`：声明 `LightningDataModule` 子类及其超参数；
- `model`：声明具体的神经网络结构；
- `task`：包装模型、定义优化逻辑，最终实例化为 `LightningModule`；
- `trainer`：配置 `pytorch_lightning.Trainer` 的加速、设备、回调等；
- `callbacks` 与 `logger`：列出需要启用的回调与日志后端；
- `train`、`paths`、`seed` 等附加字段，用于控制 checkpoint、输出路径和可复现性。

对于已有实验目录，可使用 `utils.config.resolve_experiment_config` 直接
加载 `.hydra/config.yaml` 并合并 CLI 覆盖项，实现实验复现与继续训练。

### 2.2 注册表解析

`utils.config.instantiate_from_config` 会根据 `group` 参数，从
`utils.registry` 中读取预注册的类：

| Group        | 注册表                 | 说明 |
| ------------ | ---------------------- | ---- |
| `datamodule` | `byprot.datamodules.DATAMODULE_REGISTRY` | 管理各类数据加载脚本，如 `cath_datamodule`、`tokenized_protein_datamodule` 等 |
| `model`      | `byprot.models.MODEL_REGISTRY` | 收录模型结构，例如 Transformer、语言模型骨架 |
| `task`       | `byprot.tasks.TASK_REGISTRY` | 定义训练任务与损失、指标逻辑 |

注册表保证了配置中 `_target_` 字段的稳定解析，若类不存在会给出明确的 KeyError。

## 3. 模块文档

### 3.1 数据模块 (`byprot.datamodules`)

- **主要职责**：实现 `LightningDataModule` 接口（`prepare_data`、
  `setup`、`train_dataloader` 等），封装数据下载、预处理与采样策略。
- **典型实现**：
  - `cath_datamodule.CATHDataModule`：面向 CATH 蛋白结构数据集。
  - `tokenized_protein_datamodule.TokenizedProteinDataModule`：加载已
    分词的蛋白序列。
- **配置要点**：控制 batch size、随机种子、分布式采样器等，可通过
  Hydra 的 `${...}` 语法复用通用参数。

### 3.2 模型模块 (`byprot.models`)

- **主要职责**：提供具体的神经网络骨架，返回 `torch.nn.Module`。
- **典型实现**：Transformers、语言模型、结构预测模型等，均注册在
  `MODEL_REGISTRY` 中。
- **与任务模块的关系**：模型不直接处理损失计算，而是交由任务模块组合。

### 3.3 任务模块 (`byprot.tasks`)

- **主要职责**：继承 `LightningModule`，在 `training_step`、`validation_step`
  中调用模型并计算损失，定义优化器、学习率调度器等。
- **典型实现**：
  - `lm.dplm.DPLMTrainingTask`：蛋白语言模型训练任务；
  - `lm.mlm.MLMTrainingTask` 与 `lm.dplm2.DPLM2TrainingTask`：覆盖不同掩码或目标策略；
  - `struct_tokenizer.structok.StrucTok`：结合结构标注的训练示例。
- **配置要点**：通过 `model: ${model}` 引用模型配置，实现任务与模型的解耦。

### 3.4 工具模块 (`byprot.utils`)

- `extras`：统一打开日志、打印配置树、注册 OmegaConf resolver。
- `log_hyperparameters`：在所有日志后端记录任务、模型、数据模块与参数量。
- `common_pipeline`：实例化数据模块、任务模块、日志与回调，是两个训练流程
  共享的核心装配逻辑。
- `finish`：统一关闭日志、回调资源（例如安全结束 WandB 会话）。
- `resolve_ckpt_path`：处理相对路径 checkpoint，支持当前工作目录与预设目录。

### 3.5 日志与回调

- **日志**：配置项 `logger` 支持 TensorBoard、WandB 等常见后端；对于
  TensorBoard，`common_pipeline` 会自动生成 hparams 文件以避免首次运行报错。
- **回调**：通过 `callbacks` 配置 EarlyStopping、ModelCheckpoint、
  学习率监控等；当 `trainer.enable_progress_bar` 为 `True` 时，默认添加
  自定义的 `BetterRichProgressBar` 以提供更友好的命令行进度条。

### 3.6 训练器 (`pytorch_lightning.Trainer`)

Trainer 完全由配置驱动，可选择：

- 分布式策略，例如 FSDP (`lightning.pytorch.strategies.FSDPStrategy`)；
- 设备与精度控制，如 `accelerator=gpu`、`precision=bf16`；
- 自动化功能，包含梯度累积、最大步数、检查点周期等。

## 4. 简化流水线使用指南

`byprot.simple_lightning_pipeline` 保留与全量流程一致的配置接口，
提供以下 API：

- `build_components(config)`：仅实例化组件，便于在自定义脚本中灵活调用；
- `train(config, ckpt_path=None, test_after_fit=False)`：执行训练并返回 Trainer。

### 4.1 快速上手示例

```yaml
# configs/train.yaml
datamodule:
  _target_: tokenized_protein
  data_dir: ${paths.data_dir}
  max_tokens: 4000
  max_len: 512

model:
  _target_: dplm2
  net:
    arch_type: esm
    name: airkingbd/dplm_650m

task:
  _target_: lm/dplm2
  model: ${model}

trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 5
  accelerator: gpu
  devices: 1
```

```python
from omegaconf import OmegaConf
from byprot.simple_lightning_pipeline import train

cfg = OmegaConf.load("configs/train.yaml")
trainer = train(cfg, ckpt_path=None, test_after_fit=True)
```

如需访问底层组件，可调用：

```python
from byprot.simple_lightning_pipeline import build_components

components = build_components(cfg)
components.trainer.validate(components.module, components.datamodule)
```

## 5. 实践建议

1. **设定随机种子**：在配置中添加 `seed: 42`，确保单机多卡与多进程环境的
   可复现性。
2. **管理回调与日志**：仅保留必要的回调定义，避免重复保存大量 checkpoint；
   日志后端建议统一命名 `save_dir` 与 `name`，便于排查实验。
3. **断点续训**：正式流程可通过 `config.train.ckpt_path` 自动恢复，简化流程
   支持直接传入相对路径 `train(cfg, ckpt_path="checkpoints/last.ckpt")`。
4. **测试与评估**：生产流程默认在 `config.test=True` 时加载 `best.ckpt` 测试，
   简化流程可通过 `test_after_fit=True` 即刻评估。

通过上述文档，团队成员可以快速定位训练流水线中的关键模块，理解配置项
与代码实现的映射关系，并在需要时选择合适的训练入口进行扩展。
