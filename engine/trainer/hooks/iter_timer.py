import time

from .hook import Hook

class IterTimerHook(Hook):
    def before_epoch(self, trainer):
        self.t = time.time()

    def before_iter(self, trainer):
        trainer.log_buffer.update({'data_time': time.time() - self.t})

    def after_iter(self, trainer):
        trainer.log_buffer.update({'time': time.time() - self.t})
        self.t = time.time()