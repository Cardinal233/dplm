"""简化版的 PyTorch Lightning 训练流水线。

该模块基于 ``byprot.training_pipeline`` 中的生产级实现，保留了最核心的
组件组装逻辑，并删除了与超大规模训练紧密耦合的细节，旨在提供一个更加
易读、易复用的最小训练脚手架。核心目标包括：

* 统一地构建 ``LightningDataModule``、``LightningModule``、日志记录器
  和回调。
* 提供一套更直接的 ``train`` 接口，便于在研究或原型阶段快速启动训练。
* 兼容现有 Hydra/OmegaConf 配置，保证与现有工程的衔接。

使用示例见 ``docs/pylightning_simple_pipeline.md``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import Logger as LightningLoggerBase

from . import utils

log = utils.get_logger(__name__)


@dataclass
class PipelineComponents:
    """保存训练过程中常见的 Lightning 组件集合。"""

    datamodule: LightningDataModule
    module: LightningModule
    trainer: Trainer
    callbacks: List[Callback]
    loggers: List[LightningLoggerBase]


def build_components(config: DictConfig) -> PipelineComponents:
    """根据配置构建最小化的 Lightning 训练组件。

    Args:
        config: Hydra/OmegaConf 风格的配置对象。期望包含 ``datamodule``、
            ``model``、``task``、``trainer`` 以及可选的 ``callbacks``、
            ``logger`` 等字段。

    Returns:
        PipelineComponents: 封装好的数据模块、任务模块、训练器、回调和日志对象。
    """

    # 复用原工程中的通用构建逻辑，确保与 registry 和路径解析保持一致。
    datamodule, pl_module, loggers, callbacks = utils.common_pipeline(
        config, training=True
    )

    # 初始化 Trainer，默认保持 partial 转换策略以便延后解析复杂对象。
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=loggers, _convert_="partial"
    )

    return PipelineComponents(
        datamodule=datamodule,
        module=pl_module,
        trainer=trainer,
        callbacks=callbacks,
        loggers=loggers,
    )


def _resolve_ckpt(config: DictConfig, ckpt_path: Optional[str]) -> Optional[str]:
    """解析相对或绝对的 checkpoint 路径。"""

    if not ckpt_path:
        return None

    resolved = utils.resolve_ckpt_path(config.paths.ckpt_dir, ckpt_path)
    if resolved and resolved != ckpt_path:
        log.info(f"解析 checkpoint 路径 {ckpt_path} -> {resolved}")
    return resolved


def train(
    config: DictConfig,
    *,
    ckpt_path: Optional[str] = None,
    test_after_fit: bool = False,
) -> Trainer:
    """执行一次完整的简化训练流程。

    主要步骤如下：

    1. 调用 :func:`byprot.utils.extras` 以应用额外的通用设置；
    2. 可选地设置随机种子，保证实验可复现；
    3. 构建数据模块、任务模块、回调、日志和 Trainer；
    4. 记录超参数并启动 ``trainer.fit``；
    5. 根据需要执行 ``trainer.test``，并在最后调用 :func:`byprot.utils.finish`。

    Args:
        config: Hydra/OmegaConf 配置对象。
        ckpt_path: 需要恢复的 checkpoint 路径，支持相对路径。
        test_after_fit: 是否在训练结束后立即执行测试。

    Returns:
        Trainer: 训练结束后的 Trainer 对象，可用于进一步分析。
    """

    utils.extras(config)

    if config.get("seed") is not None:
        log.info(f"设置随机种子为 {config.seed}")
        seed_everything(config.seed, workers=True)

    components = build_components(config)

    # 记录超参数，方便在日志后端对比实验。
    utils.log_hyperparameters(
        config=config,
        model=components.module,
        datamodule=components.datamodule,
        trainer=components.trainer,
        callbacks=components.callbacks,
        logger=components.loggers,
    )

    resume_ckpt = _resolve_ckpt(config, ckpt_path)
    if resume_ckpt:
        log.info(f"从 checkpoint <{resume_ckpt}> 恢复训练")

    components.trainer.fit(
        model=components.module,
        datamodule=components.datamodule,
        ckpt_path=resume_ckpt,
    )

    if test_after_fit:
        log.info("训练完成，开始测试阶段")
        components.trainer.test(
            model=components.module,
            datamodule=components.datamodule,
        )

    utils.finish(
        config=config,
        model=components.module,
        datamodule=components.datamodule,
        trainer=components.trainer,
        callbacks=components.callbacks,
        logger=components.loggers,
    )

    return components.trainer


__all__ = ["PipelineComponents", "build_components", "train"]
