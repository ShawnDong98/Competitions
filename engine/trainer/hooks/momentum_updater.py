from .hook import HOOKS, Hook
from .lr_updater import annealing_cos, annealing_linear, format_param

class MomentumUpdaterHook(Hook):
    def __init__(
        self, 
        by_epoch=True, 
        warmup=None, 
        warmup_iters=0, 
        warmup_ratio=0.9
    ):
        if warmup is not None:
            if warmup not in ['constant', 'linear', 'exp']:
                raise ValueError(
                    'Warmup type {} not supported'.format(warmup)
                )
        if warmup is not None:
            assert warmup_iters > 0, \
                '"warmup_iters" must be greater than 0'
            assert 0 < warmup_ratio <= 1.0, \
                '"warmup_ratio" must be in range (0, 1]'

        self.by_epoch = by_epoch
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio

        self.base_momentum = []
        self.regular_momentum = []

    def _set_momentum(self, trainer, momentum_groups):
        for param_group, mom in zip(trainer.optimizer.param_groups, momentum_groups):
            if 'momentum' in param_group.keys():
                param_group['momentum'] = mom
            elif 'betas' in param_group.keys():
                param_group['betas'] = (mom, param_group['betas'][1])

    def get_momentum(self, trainer, base_momentum):
        raise NotImplementedError

    def get_regular_momentum(self, trainer):
        return [
            self.get_momentum(trainer, _base_momentum)
            for _base_momentum in self.base_momentum
        ]

    def get_warmup_momentum(self, cur_iters):
        if self.warmup == 'constant':
            warmup_momentum = [
                _momentum / self.warmup_ratio
                for _momentum in self.regular_momentum
            ]
        elif self.warmup == 'linear':
            k = (1 - cur_iters / self.warmup_iters) * (1 - self.warmup_ratio)
            warmup_momentum = [
                _momentum / (1 - k) for _momentum in self.regular_mom
            ]
        elif self.warmup == 'exp':
            k = self.warmup_ratio ** (1 - cur_iters / self.warmup_iters)
            warmup_momentum = [
                _momentum / k for _momentum in self.regular_mom
            ]
        return warmup_momentum

    def before_run(self, trainer):
        # NOTE: when resuming from a checkpoint,
        # if 'initial_momentum' is not saved,
        # it will be set according to the optimizer params
        for group in trainer.optimizer.param_groups:
            if 'momentum' in group.keys():
                group.setdefault('initial_momentum', group['momentum'])
            else:
                group.setdefault('initial_momentum', group['betas'][0])
        self.base_momentum = [
            group['initial_momentum']
            for group in trainer.optimizer.param_groups
        ]

    def before_train_epoch(self, trainer):
        if not self.by_epoch:
            return
        self.regular_mom = self.get_regular_momentum(trainer)
        self._set_momentum(trainer, self.regular_mom)

    def before_train_iter(self, trainer):
        cur_iter = trainer.iter
        if not self.by_epoch:
            self.regular_mom = self.get_regular_momentum(trainer)
            if self.warmup is None or cur_iter >= self.warmup_iters:
                self._set_momentum(trainer, self.regular_mom)
            else:
                warmup_momentum = self.get_warmup_momentum(cur_iter)
                self._set_momentum(tariner, warmup_momentum)
        elif self.by_epoch:
            if self.warmup is None or cur_iter > self.warmup_iters:
                return
            elif cur_iter == self.warmup_iters:
                self._set_momentum(trainer, self.regular_mom)
            else:
                warmup_momentum = self.get_warmup_momentum(cur_iter)
                self._set_momentum(trainer, warmup_momentum)



@HOOKS.register_module()
class OneCycleMomentumUpdaterHook(MomentumUpdaterHook):
    """OneCycle momentum Scheduler.
    This momentum scheduler usually used together with the OneCycleLrUpdater
    to improve the performance.
    Args:
        base_momentum (float or list): Lower momentum boundaries in the cycle
            for each parameter group. Note that momentum is cycled inversely
            to learning rate; at the peak of a cycle, momentum is
            'base_momentum' and learning rate is 'max_lr'.
            Default: 0.85
        max_momentum (float or list): Upper momentum boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_momentum - base_momentum).
            Note that momentum is cycled inversely
            to learning rate; at the start of a cycle, momentum is
            'max_momentum' and learning rate is 'base_lr'
            Default: 0.95
        pct_start (float): The percentage of the cycle (in number of steps)
            spent increasing the learning rate.
            Default: 0.3
        anneal_strategy (str): {'cos', 'linear'}
            Specifies the annealing strategy: 'cos' for cosine annealing,
            'linear' for linear annealing.
            Default: 'cos'
        three_phase (bool): If three_phase is True, use a third phase of the
            schedule to annihilate the learning rate according to
            final_div_factor instead of modifying the second phase (the first
            two phases will be symmetrical about the step indicated by
            pct_start).
            Default: False
    """
    def __init__(
        self,
        base_momentum = 0.85,
        max_momentum = 0.95,
        pct_start = 0.3,
        anneal_strategy = 'cos',
        three_phase = False,
        **kwargs
    ):
        if 'by_epoch' not in kwargs:
            kwargs['by_epoch'] = False
        else:
            assert not kwargs['by_epoch'], 'currently only support "by_epoch" = False'
        if not isinstance(base_momentum, (float, list, dict)):
            raise ValueError('base_momentum must be the type among of float, list or dict.')
        self._base_momentum = base_momentum
        if not isinstance(max_momentum, (float, list, dict)):
            raise ValueError('max_momentum must be the type among of float, list or dict.')
        self._max_momentum = max_momentum

        if pct_start < 0 or pct_start > 1 or not isinstance(pct_start, float):
            raise ValueError(f'Expected float between 0 and 1 pct_start, but got {pct_start}')
        self.pct_start = pct_start

        if anneal_strategy not in ['cos', 'linear']:
            raise ValueError(f'anneal_strategy must by one of "cos" or "linear", instead of {anneal_strategy}')
        elif anneal_strategy == 'cos':
            self.anneal_func = annealing_cos
        elif anneal_strategy == 'linear':
            self.anneal_func = anneal_linear
        self.three_phase = three_phase
        self.momentum_phases = []
        super(OneCycleMomentumUpdaterHook, self).__init__(**kwargs)

    def before_run(self, trainer):
        optim = trainer.optimizer
        if ('momentum' not in optim.defaults and 'betas' not in optim.defaults):
            raise ValueError('optimizer must support momentum with option enabled')
        self.use_beta1 = 'betas' in optim.defaults
        k = type(optim).__name__
        _base_momentum = format_param(k, optim, self._base_momentum)
        _max_momentum = format_param(k, optim, self._max_momentum)
        for group, b_momentum, m_momentum in zip(optim.param_groups, _base_momentum, _max_momentum):
            if self.use_beta1:
                _, beta2 = group['betas']
                group['betas'] = (m_momentum, beta2)
            else:
                group['momentum'] = m_momentum
            group['base_momentum'] = b_momentum
            group['max_momentum'] = m_momentum

        if self.three_phase:
            self.momentum_phases.append({
                'end_iter':
                float(self.pct_start * trainer.max_iters) - 1,
                'start_momentum':
                'max_momentum',
                'end_momentum':
                'base_momentum'
            })
            self.momentum_phases.append({
                'end_iter':
                float(2 * self.pct_start * trainer.max_iters) - 2,
                'start_momentum':
                'base_momentum',
                'end_momentum':
                'max_momentum'
            })
            self.momentum_phases.append({
                'end_iter': trainer.max_iters - 1,
                'start_momentum': 'max_momentum',
                'end_momentum': 'max_momentum'
            })
        else:
            self.momentum_phases.append({
                'end_iter':
                float(self.pct_start * trainer.max_iters) - 1,
                'start_momentum':
                'max_momentum',
                'end_momentum':
                'base_momentum'
            })
            self.momentum_phases.append({
                'end_iter': trainer.max_iters - 1,
                'start_momentum': 'base_momentum',
                'end_momentum': 'max_momentum'
            })
    def _set_momentum(self, trainer, momentum_groups):
        for param_group, mom in zip(trainer.optimizer.param_groups, momentum_groups):
            if 'momentum' in param_group.keys():
                param_group['momentum'] = mom
            elif 'betas' in param_group.keys():
                param_group['betas'] = (mom, param_group['betas'][1])

    def get_momentum(self, trainer, param_group):
        cur_iter = trainer.iter
        start_iter = 0
        for i, phase in enumerate(self.momentum_phases):
            end_iter = phase['end_iter']
            if cur_iter <= end_iter or i == len(self.momentum_phases) - 1:
                pct = (cur_iter - start_iter) / (end_iter - start_iter)
                momentum = self.anneal_func(
                    param_group[phase['start_momentum']],
                    param_group[phase['end_momentum']],
                    pct
                )
                break
            start_iter = end_iter

        return momentum

    def get_regular_momentum(self, trainer):
        momentum_groups = []
        for param_group in trainer.optimizer.param_groups:
            momentum_groups.append(self.get_momentum(trainer, param_group))
        return momentum_groups

