import pytorch_lightning as pl


class LinearWarmupLRCallback(pl.Callback):
    """Simple linear warmup that works with existing optimizer setup.

    It only changes optimizer param_group['lr'] during the first warmup_steps.
    After warmup, lr is kept at base_lr.
    """
    def __init__(self, base_lr: float, warmup_steps: int = 0, min_lr_scale: float = 0.0, verbose: bool = True):
        super().__init__()
        self.base_lr = float(base_lr)
        self.warmup_steps = int(warmup_steps)
        self.min_lr_scale = float(min_lr_scale)
        self.verbose = bool(verbose)
        self._printed_done = False

    def _scale(self, step: int) -> float:
        if self.warmup_steps <= 0:
            return 1.0
        if step >= self.warmup_steps:
            return 1.0
        progress = float(step + 1) / float(self.warmup_steps)
        return self.min_lr_scale + (1.0 - self.min_lr_scale) * progress

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.warmup_steps <= 0:
            return
        step = int(trainer.global_step)
        lr = self.base_lr * self._scale(step)
        for opt in trainer.optimizers:
            for group in opt.param_groups:
                group["lr"] = lr
        if self.verbose and step == 0 and trainer.is_global_zero:
            print(f"[WarmupLR] enabled: base_lr={self.base_lr}, warmup_steps={self.warmup_steps}, min_lr_scale={self.min_lr_scale}", flush=True)
        if self.verbose and (not self._printed_done) and step >= self.warmup_steps and trainer.is_global_zero:
            print(f"[WarmupLR] finished at global_step={step}, lr={self.base_lr}", flush=True)
            self._printed_done = True
