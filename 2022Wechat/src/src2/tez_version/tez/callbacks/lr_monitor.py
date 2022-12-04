from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple, Type

from torch.optim.optimizer import Optimizer

import numpy as np
from tez import enums
from tez.callbacks import Callback

class LearningRateMonitor(Callback):

    def on_train_step_end(self, model, **kwargs):
        optimizer = model.optimizer
        lr_momentum_stat = self._get_lr_momentum_stat(optimizer)

    def _get_lr_momentum_stat(self, optimizer: Optimizer, names: List[str]) -> Dict[str, float]:
        lr_momentum_stat = {}
        param_groups = optimizer.param_groups
        use_betas = "betas" in optimizer.defaults

        for pg, name in zip(param_groups, names):
            lr = self._extract_lr(pg, name)
            lr_momentum_stat.update(lr)
            momentum = self._extract_momentum(
                param_group=pg, name=name.replace(name, f"{name}-momentum"), use_betas=use_betas
            )
            lr_momentum_stat.update(momentum)

        return lr_momentum_stat

    def _extract_lr(self, param_group: Dict[str, Any], name: str) -> Dict[str, Any]:
        lr = param_group["lr"]
        self.lrs[name].append(lr)
        return {name: lr}

    def _extract_momentum(self, param_group: Dict[str, List], name: str, use_betas: bool) -> Dict[str, float]:
        if not self.log_momentum:
            return {}

        momentum = param_group["betas"][0] if use_betas else param_group.get("momentum", 0)
        self.last_momentum_values[name] = momentum
        return {name: momentum}

    def _find_names_from_schedulers(
        self, lr_scheduler_configs: List[LRSchedulerConfig], add_lr_sch_names: bool = True
    ) -> Tuple[List[List[str]], List[Optimizer], DefaultDict[Type[Optimizer], int]]:
        # Create unique names in the case we have multiple of the same learning
        # rate scheduler + multiple parameter groups
        names = []
        seen_optimizers: List[Optimizer] = []
        seen_optimizer_types: DefaultDict[Type[Optimizer], int] = defaultdict(int)
        for config in lr_scheduler_configs:
            sch = config.scheduler
            if config.name is not None:
                name = config.name
            else:
                name = "lr-" + sch.optimizer.__class__.__name__

            updated_names = self._check_duplicates_and_update_name(
                sch.optimizer, name, seen_optimizers, seen_optimizer_types, config, add_lr_sch_names
            )
            names.append(updated_names)

        return names, seen_optimizers, seen_optimizer_types


    def _find_names_from_optimizers(
        self,
        optimizers: List[Any],
        seen_optimizers: List[Optimizer],
        seen_optimizer_types: DefaultDict[Type[Optimizer], int],
        add_lr_sch_names: bool = True,
    ) -> Tuple[List[List[str]], List[Optimizer]]:
        names = []
        optimizers_without_scheduler = []

        for optimizer in optimizers:
            # Deepspeed optimizer wraps the native optimizer
            optimizer = optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer
            if optimizer in seen_optimizers:
                continue

            name = "lr-" + optimizer.__class__.__name__
            updated_names = self._check_duplicates_and_update_name(
                optimizer, name, seen_optimizers, seen_optimizer_types, None, add_lr_sch_names
            )
            names.append(updated_names)
            optimizers_without_scheduler.append(optimizer)

        return names, optimizers_without_scheduler

    def _extract_stats(self, trainer: "pl.Trainer", interval: str) -> Dict[str, float]:
        (
            scheduler_hparam_keys,
            optimizers_with_scheduler,
            optimizers_with_scheduler_types,
        ) = self._find_names_from_schedulers(trainer.lr_scheduler_configs, add_lr_sch_names=False)
