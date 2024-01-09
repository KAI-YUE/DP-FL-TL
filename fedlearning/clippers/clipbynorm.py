import torch

from opacus.optimizers.optimizer import _check_processed_flag, _mark_as_processed
from fedlearning.clippers.initialize import Clipper
from opt_einsum.contract import contract

class ClipByNorm(Clipper):
    """Clipper that clips values to a specified range."""
    def __init__(self, bound):
        super().__init__()
        self.bound = bound

    def _clip(self, tensor):
        pass
    
    def clip_then_accumulate(self, optimizer):
        """
        Performs gradient clipping.
        Stores clipped and aggregated gradients into `p.summed_grad```
        """

        if len(optimizer.grad_samples[0]) == 0:
            # Empty batch
            per_sample_clip_factor = torch.zeros((0,))
        else:
            per_param_norms = [
                g.reshape(len(g), -1).norm(2, dim=-1) for g in optimizer.grad_samples
            ]
            per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
            per_sample_clip_factor = (
                optimizer.max_grad_norm / (per_sample_norms + 1e-6)
            ).clamp(max=1.0)

        for p in optimizer.params:
            _check_processed_flag(p.grad_sample)
            grad_sample = optimizer._get_flat_grad_sample(p)
            grad = contract("i,i...", per_sample_clip_factor, grad_sample)
            
            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            # p.summed_grad.zero_()

            _mark_as_processed(p.grad_sample)