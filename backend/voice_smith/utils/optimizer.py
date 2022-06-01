import torch
import numpy as np
from typing import Iterable, Dict, Any, Union
from voice_smith.config.configs import (
    AcousticFinetuningConfig,
    AcousticPretrainingConfig,
)


class ScheduledOptimPretraining:
    def __init__(
        self,
        parameters: Iterable,
        train_config: AcousticPretrainingConfig,
        current_step: int,
    ):
        self._optimizer = torch.optim.Adam(
            parameters,
            betas=train_config.optimizer_config.betas,
            eps=train_config.optimizer_config.eps,
        )
        self.n_warmup_steps = train_config.optimizer_config.warm_up_step
        self.anneal_steps = train_config.optimizer_config.anneal_steps
        self.anneal_rate = train_config.optimizer_config.anneal_rate
        self.current_step = current_step
        self.init_lr = train_config.optimizer_config.learning_rate

    def step_and_update_lr(self, step: int) -> None:
        self._update_learning_rate(step)
        self._optimizer.step()

    def zero_grad(self) -> None:
        self._optimizer.zero_grad()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._optimizer.load_state_dict(state_dict)

    def _get_lr_scale(self) -> float:
        lr_scale = np.min(
            [
                np.power(1 if self.current_step == 0 else self.current_step, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.current_step,
            ]
        )
        for s in self.anneal_steps:
            if self.current_step > s:
                lr_scale = lr_scale * self.anneal_rate
        return lr_scale

    def _update_learning_rate(self, step: int) -> None:
        """Learning rate scheduling per step"""
        self.current_step = step
        lr = self.init_lr * self._get_lr_scale()
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr


class ScheduledOptimFinetuning:
    def __init__(
        self,
        parameters: Iterable,
        train_config: AcousticFinetuningConfig,
        current_step: int,
    ):
        self._optimizer = torch.optim.AdamW(
            parameters,
            betas=train_config.optimizer_config.betas,
            eps=train_config.optimizer_config.eps,
        )
        self.current_step = current_step
        self.init_lr = train_config.optimizer_config.learning_rate
        self.lr_decay = train_config.optimizer_config.lr_decay

    def step_and_update_lr(self, step: int) -> None:
        self._update_learning_rate(step)
        self._optimizer.step()

    def zero_grad(self) -> None:
        self._optimizer.zero_grad()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._optimizer.load_state_dict(state_dict)

    def _get_lr_scale(self) -> float:
        lr_scale = self.lr_decay**self.current_step
        return lr_scale

    def _update_learning_rate(self, step: int) -> None:
        """Learning rate scheduling per step"""
        self.current_step = step
        lr = self.init_lr * self._get_lr_scale()
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
