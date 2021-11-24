from .hook import Hook
from ..utils import master_only


class CheckpointHook(Hook):
    def __init__(
        self,
        interval = -1,
        save_optimizer = True,
        out_dir = None,
        **kwargs
    ):
        self.interval = interval
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.args = kwargs

    @master_only
    def after_train_epoch(self, trainer):
        if not self.every_n_epcohs(trainer, self.interval):
            return

        if not self.out_dir:
            self.out_dir = trainer.work_dir

        trainer.save_checkpoint(
            out_dir = self.out_dir, 
            filename_tmpl = f'{trainer.config.model.name}_epoch_{trainer.epoch}',
            save_optimizer=self.save_optimizer, 
            **self.args
        )
