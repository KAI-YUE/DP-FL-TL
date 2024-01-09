from typing import Callable, List, Optional, Union

from opacus.optimizers.optimizer import DPOptimizer, _check_processed_flag, _mark_as_processed

from fedlearning.clippers import clipper_registry

class MyDPOptimizer(DPOptimizer):
    def __init__(self, config, dp_optimizer):
        self.add_noise = config.add_noise
        self.dp_optimizer = dp_optimizer
        self.clipper = clipper_registry[config.clipping_scheme](config.clipping_bound)

    def zero_grad(self, set_to_none: bool = False):
        self.dp_optimizer.zero_grad(set_to_none)

    def pre_step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        """
        Perform actions specific to ``DPOptimizer`` before calling
        underlying  ``optimizer.step()``

        Args:
            closure: A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        self.clipper.clip_then_accumulate(self.dp_optimizer)
        if self.dp_optimizer._check_skip_next_step():
            self.dp_optimizer._is_last_step_skipped = True
            return False

        if self.add_noise:
            self.dp_optimizer.add_noise()
        else:
            # break
            self.set_grad()
            # pass

        self.dp_optimizer.scale_grad()

        if self.dp_optimizer.step_hook:
            self.dp_optimizer.step_hook(self.dp_optimizer)

        self.dp_optimizer._is_last_step_skipped = False
        return True
    
    def set_grad(self):
        for p in self.dp_optimizer.params:
            _check_processed_flag(p.summed_grad)
            p.grad = p.summed_grad.view_as(p)

            _mark_as_processed(p.summed_grad)

    def step(self):
        if self.pre_step():
            return self.dp_optimizer.original_optimizer.step()
        else:
            return None

def prepare_mydp_optimizer(config, optimizer):
    dp_optimizer = MyDPOptimizer(config, optimizer)
    return dp_optimizer