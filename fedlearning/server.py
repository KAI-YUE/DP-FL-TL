# My libraries
from fedlearning.buffer import WeightBuffer
from fedlearning.aggregators import aggregator_registry

class GlobalUpdater(object):
    def __init__(self, config, **kwargs):
        """Construct a global updater for a server.

        Args:
            config (class):              global configuration containing info listed as follows:
        """
        self.num_users = int(config.user_num)
        self.device = config.device
        self.global_lr = config.global_lr

        self.aggregator = aggregator_registry[config.aggregator](config)

    def global_step(self, model, packages, **kwargs):
        # merge benign and attacker packages, as we assume the server does not know which client is attacker
        accumulated_delta = self.aggregator(packages)
        accumulated_delta *= (self.global_lr)
        global_weight = WeightBuffer(model) - accumulated_delta
        model.load_state_dict(global_weight.state_dict())
