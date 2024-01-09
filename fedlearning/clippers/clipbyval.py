from opacus.optimizers.optimizer import _check_processed_flag, _mark_as_processed
from fedlearning.clippers.initialize import Clipper

class ClipByValue(Clipper):
    """Clipper that clips values to a specified range."""
    def __init__(self, bound):
        super().__init__()
        self.bound = bound

    def _clip(self, tensor):
        """Clips the given tensor."""
        return tensor.clamp(-self.bound, self.bound)
    
    def clip_then_accumulate(self, optimizer):
        """
        Performs personalized clipping given an optimizer.
        Stores clipped and aggregated gradients into `p.summed_grad```
        """

        for p in optimizer.params:
            _check_processed_flag(p.grad_sample)
            grad_sample = optimizer._get_flat_grad_sample(p)
            for grad in grad_sample:
                #print(grad)
                grad = self._clip(grad)

                if p.summed_grad is not None:
                    p.summed_grad += grad
                else:
                    p.summed_grad = grad

            _mark_as_processed(p.grad_sample)