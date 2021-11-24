import numpy as np
from .hook import Hook

class EarlyStoppingHook(Hook):
    """Stop training when a monitored quantity has stopped improving.
    # Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
    """

    def __init__(
        self, 
        monitor='val_loss',
        min_delta=0.0, 
        patience=0, 
        verbose=0, 
        mode='auto'
    ):
        super(EarlyStoppingHook, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0

        if mode not in ['auto', 'min', 'max']:
            print('EarlyStopping mode %s is unknown, fallback to auto mode.' % mode)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def before_run(self, trainer):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def after_val_epoch(self, trainer):
        epoch = trainer.epoch
        current = trainer.log_buffer.output.get(self.monitor)
        stop_training = False
        if current is None:
            print('Early stopping conditioned on metric `%s` ''which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(trainer.log_buffer.output.keys()))), RuntimeWarning
            )
            exit(-1)

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                stop_training = True

        trainer.stop_training = stop_training

    def after_run(self, trainer):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))