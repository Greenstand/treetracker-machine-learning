from torch.optim.lr_scheduler import _LRScheduler, StepLR

class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters  # avoid zero lr
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [ max( base_lr * ( 1 - self.last_epoch/self.max_iters )**self.power, self.min_lr)
                for base_lr in self.base_lrs]

class GradualWarmupLR(_LRScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler, last_epoch=-1):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.total_epoch:
            if not self.finished:
                self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                self.finished = True
            return self.after_scheduler.get_lr()
        return [base_lr * ((self.multiplier - 1.0) / self.total_epoch * self.last_epoch + 1.0) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if self.last_epoch < self.total_epoch:
            super(GradualWarmupLR, self).step(epoch)
        else:
            self.after_scheduler.step(epoch - self.total_epoch)