import numpy as np
import torch

class CheckPoint(object):
    def __init__(self, verbose=False, trace_func=print):
        self.verbose = verbose
        self.trace_func = trace_func
        self.val_loss_min = np.Inf
        self.best_score = None

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            self.trace_func(f'\nValidation loss decreased ({self.val_loss_min:.3f} --> {val_loss:.3f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss