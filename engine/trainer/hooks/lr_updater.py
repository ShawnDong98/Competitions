from __future__ import division
import numbers
from mathim import cos, pi

from .hook import Hook


class LrUpdaterHook(Hook):
    def __init__(
        self,
        by_epoch = True,
        warmup = None,
        warmup_iters = 0,
        warmup_ratio = 0.1,
        **kwargs
    ):
        # validate the "warmup" argument
        if warmup is not None:
            if warmup not in ['constant', 'linear', 'exp']:
                raise ValueError(
                    '"{}" is not a supported type for warming up, valid types'
                    ' are "constant" and "linear"'.format(warmup))

        if warmup is not None:
            assert warmup_iters > 0, \
                '"warmup_iters" must be a positive integer'
            assert 0 < warmup_ratio <= 1.0, \
                '"warmup_ratio" must be in range (0,1]'

        self.by_epoch = by_epoch
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio

        self.base_lr = []  # initial lr for all param groups
        self.regular_lr = [] # expected lr if no warming up is performed

    def _set_lr(self, trainer, lr_groups):
        for param_group, lr in zip(trainer.optimizer.param_groups, lr_groups):
            param_group['lr'] = lr

    def get_lr(self, trainer, base_lr):
        raise NotImplemented

    def get_regular_lr(self, trainer):
        return [self.get_lr(trainer, _base_lr) for _base_lr in self.base_lr]

    def get_warmup_lr(self, cur_iters):
        if self.warmup == 'constant':
            warmup_lr = [_lr * self.warmup_ratio for _lr in self.regular_lr]
        elif self.warmup == 'linear':
            k = (1 - cur_iters / self.warmup_iters) * (1 - self.warmup_ratio)
            warmup_lr = [_lr * (1 - k) for _lr in self.regular_lr]
        elif self.warmup == 'exp':
            k = self.warmup_ratio ** (1 - cur_iters / self.warmup_iters)
            warmup_lr = [_lr * k for _lr in self.regular_lr]
        return warmup_lr

    def before_run(self, trainer):
        # NOTE: when resuming from a checkpoint, if 'initial_lr' is not saved,
        # it will be set according to the optimizer params
        for group in trainer.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.base_lr = [
            group['initial_lr'] for group in trainer.optimizer.param_groups
        ]

    def before_train_epoch(self, trainer):
        if not self.by_epoch:
            return
        self.regular_lr = self.get_regular_lr(trainer)
        self._set_lr(trainer, self.regular_lr)

    def before_train_iter(self, trainer):
        cur_iter = trainer.iter
        if not self.by_epoch:
            self.regular_lr = self.get_regular_lr(trainer)
            if self.warmup is None or cur_iter >= self.warmup_iters:
                self._set_lr(trainer, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self.set_lr(trainer, warmup_lr)
        elif self.by_epoch:
            if self.warmup is None and cur_iter > self.warmup_iters:
                return
            elif cur_iter == self.warmup_iters:
                self._set_lr(trainer, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(trainer, warmup_lr)


class FixedLrUpdaterHook(LrUpdaterHook):
    def __init__(self, **kwargs):
        super(FixedLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, trainer, base_lr):
        return base_lr

class StepLrUpdaterHook(LrUpdaterHook):
    def __init__(self, step, gamma=0.1, **kwargs):
        assert isinstance(step, (list, int))
        if isinstance(step, list):
            for s in step:
                assert isinstance(s, int) and s > 0
        elif isinstance(step, int):
            assert step > 0
        else:
            raise TypeError('"step" must be a list or integer')
        self.step = step
        self.gamma = gamma
        super(StepLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, trainer, base_lr):
        progress = trainer.epoch if self.by_epoch else trainer.step

        if isinstance(self.step, int):
            return base_lr * (self.gamma ** (progress // self.step))

        exp = len(self.step)
        for i, s in enumerate(self.step):
            if progress < s:
                exp = i
                break
        return base_lr * self.gamma ** exp


class ExpLrUpdaterHook(LrUpdaterHook):
    def __init__(self, gamma, **kwargs):
        self.gamma = gamma
        super(ExpLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, trainer, base_lr):
        progress = trainer.epoch if self.by_epoch else trainer.iter
        return base_lr * self.gamma ** progress


class PolyLrUpdaterHook(LrUpdaterHook):
    def __init__(self, power=1., **kwargs):
        self.power = power
        super(PolyLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, trainer, base_lr):
        if self.by_epoch:
            progress = trainer.epoch
            max_progress = trainer.max_epoch
        else:
            progres = trainer.iter
            max_progress = trainer.max_iters

        return base_lr * (1 - progress / max_progress) ** self.power



class InvLrUpdaterHook(LrUpdaterHook):
    def __init__(self, gamma, power=1., **kwargs):
        self.gamma = gamma
        self.power = power
        super(InvLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, trainer, base_lr):
        progress = trainer.epoch if self.by_epoch else trainer.iter
        return base_lr * (1 + self.gamma * progress) ** (-self.power)


class CosineAnnealingLrUpdaterHook(LrUpdaterHook):
    def __init__(self, min_lr=None, min_lr_ratio=None, **kwargs):
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        super(CosineAnnealingLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, trainer, base_lr):
        if self.by_epoch:
            progress = trainer.epoch
            max_progress = trainer.max_epochs
        else:
            progress = trainer.iter
            max_progress = trainer.max_iters
        if self.min_lr_ratio is not None:                                                                     target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr

        return annealing_cos(base_lr, target_lr, progress / max_progress)


class CosineRestartLrUpdaterHook(LrUpdaterHook):
    """Cosine annealing with restarts learning rate scheme.
    Args:
        periods (list[int]): Periods for each cosine anneling cycle.
        restart_weights (list[float], optional): Restart weights at each
            restart iteration. Default: [1].
        min_lr (float, optional): The minimum lr. Default: None.
        min_lr_ratio (float, optional): The ratio of minimum lr to the base lr.
            Either `min_lr` or `min_lr_ratio` should be specified.
            Default: None.
    """

    def __init__(self,
                 periods,
                 restart_weights=[1],
                 min_lr=None,
                 min_lr_ratio=None,
                 **kwargs):
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.periods = periods
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        self.restart_weights = restart_weights
        assert (len(self.periods) == len(self.restart_weights)
                ), 'periods and restart_weights should have the same length.'
        super(CosineRestartLrUpdaterHook, self).__init__(**kwargs)

        self.cumulative_periods = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]

    def get_lr(self, trainer, base_lr):
        if self.by_epoch:
            progress = trainer.epoch
        else:
            progress = trainer.iter

        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr

        idx = get_position_from_periods(progress, self.cumulative_periods)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_periods[idx - 1]
        current_periods = self.periods[idx]

        alpha = min((progress - nearest_restart) / current_periods, 1)
        return annealing_cos(base_lr, target_lr, alpha, current_weight)

class CyclicLrUpdaterHook(LrUpdaterHook):
    """Cyclic LR Scheduler.
    Implement the cyclical learning rate policy (CLR) described in
    https://arxiv.org/pdf/1506.01186.pdf
    Different from the original paper, we use cosine annealing rather than
    triangular policy inside a cycle. This improves the performance in the
    3D detection area.
    Args:
        by_epoch (bool): Whether to update LR by epoch.
        target_ratio (tuple[float]): Relative ratio of the highest LR and the
            lowest LR to the initial LR.
        cyclic_times (int): Number of cycles during training
        step_ratio_up (float): The ratio of the increasing process of LR in
            the total cycle.
        anneal_strategy (str): {'cos', 'linear'}
            Specifies the annealing strategy: 'cos' for cosine annealing,
            'linear' for linear annealing. Default: 'cos'.
    """

    def __init__(self,
                 by_epoch=False,
                 target_ratio=(10, 1e-4),
                 cyclic_times=1,
                 step_ratio_up=0.4,
                 anneal_strategy='cos',
                 **kwargs):
        if isinstance(target_ratio, float):
            target_ratio = (target_ratio, target_ratio / 1e5)
        elif isinstance(target_ratio, tuple):
            target_ratio = (target_ratio[0], target_ratio[0] / 1e5) \
                if len(target_ratio) == 1 else target_ratio
        else:
            raise ValueError('target_ratio should be either float '
                             f'or tuple, got {type(target_ratio)}')

        assert len(target_ratio) == 2, \
            '"target_ratio" must be list or tuple of two floats'
        assert 0 <= step_ratio_up < 1.0, \
            '"step_ratio_up" must be in range [0,1)'

        self.target_ratio = target_ratio
        self.cyclic_times = cyclic_times
        self.step_ratio_up = step_ratio_up
        self.lr_phases = []  # init lr_phases
        # validate anneal_strategy
        if anneal_strategy not in ['cos', 'linear']:
            raise ValueError('anneal_strategy must be one of "cos" or '
                             f'"linear", instead got {anneal_strategy}')
        elif anneal_strategy == 'cos':
            self.anneal_func = annealing_cos
        elif anneal_strategy == 'linear':
            self.anneal_func = annealing_linear

        assert not by_epoch, \
            'currently only support "by_epoch" = False'
        super(CyclicLrUpdaterHook, self).__init__(by_epoch, **kwargs)

    def before_run(self, trainer):
        super(CyclicLrUpdaterHook, self).before_run(trainer)
        # initiate lr_phases
        # total lr_phases are separated as up and down
        max_iter_per_phase = trainer.max_iters // self.cyclic_times
        iter_up_phase = int(self.step_ratio_up * max_iter_per_phase)
        self.lr_phases.append(
            [0, iter_up_phase, max_iter_per_phase, 1, self.target_ratio[0]])
        self.lr_phases.append([
            iter_up_phase, max_iter_per_phase, max_iter_per_phase,
            self.target_ratio[0], self.target_ratio[1]
        ])

    def get_lr(self, trainer, base_lr):
        curr_iter = trainer.iter
        for (start_iter, end_iter, max_iter_per_phase, start_ratio,
             end_ratio) in self.lr_phases:
            curr_iter %= max_iter_per_phase
            if start_iter <= curr_iter < end_iter:
                progress = curr_iter - start_iter
                return self.anneal_func(base_lr * start_ratio,
                                        base_lr * end_ratio,
                                        progress / (end_iter - start_iter))


def annealing_cos(start, end, factor, weight=1):
    """Calculate annealing cos learning rate.
    Cosine anneal from `weight * start + (1 - weight) * end` to `end` as
    percentage goes from 0.0 to 1.0.
    Args:
        start (float): The starting learning rate of the cosine annealing.
        end (float): The ending learing rate of the cosine annealing.
        factor (float): The coefficient of `pi` when calculating the current
            percentage. Range from 0.0 to 1.0.
        weight (float, optional): The combination factor of `start` and `end`
            when calculating the actual starting learning rate. Default to 1.
    """
    cos_out = cos(pi * factor) + 1
    return end + 0.5 * weight * (start - end) * cos_out


def annealing_linear(start, end, factor):
    """Calculate annealing linear learning rate.
    Linear anneal from `start` to `end` as percentage goes from 0.0 to 1.0.
    Args:
        start (float): The starting learning rate of the linear annealing.
        end (float): The ending learing rate of the linear annealing.
        factor (float): The coefficient of `pi` when calculating the current
            percentage. Range from 0.0 to 1.0.
    """
    return start + (end - start) * factor

def format_param(name, optim, param):
    if isinstance(param, numbers.Number):
        return [param] * len(optim.param_groups)
    elif isinstance(param, (list, tuple)):  # multi param groups
        if len(param) != len(optim.param_groups):
            raise ValueError(f'expected {len(optim.param_groups)} '
                             f'values for {name}, got {len(param)}')
        return param
    else:  # multi optimizers
        if name not in param:
            raise KeyError(f'{name} is not found in {param.keys()}')
        return param[name]
