import torch

from fedlearning.buffer import WeightBuffer
from fedlearning.aggregators.initialize import _BaseAggregator

class Vote(_BaseAggregator):
    r"""Computes the vote results of binary signs."""
    def __init__(self, config):
        self.num_users = config.user_num

    def __call__(self, local_packages):
        accumulated_delta = local_packages[0]
        for user_id, package in local_packages.items():
            if user_id == 0:
                continue
            accumulated_delta = accumulated_delta + package

        # take the sign and multiply by the learning rate
        for w_weight, delta in accumulated_delta._weight_dict.items():
            accumulated_delta._weight_dict[w_weight] = delta.sign()

        return accumulated_delta


class PluralityVote(_BaseAggregator):
    r"""Computes the vote results of ternary results."""
    def __init__(self, config):
        self.num_users = config.total_users

    def __call__(self, local_packages):
        first_idx = list(local_packages.keys())[0]
        user_state_dict = local_packages[first_idx].state_dict()

        num_ones, num_zeros, voted_results = {}, {}, {}
        for w_name, w_value in user_state_dict.items():
            num_ones[w_name] = torch.zeros_like(w_value)
            num_zeros[w_name] = torch.zeros_like(w_value)
            voted_results[w_name] = -torch.ones_like(w_value)

        for user_id, package in local_packages.items():
            for w_name, w_value in package._weight_dict.items():
                num_ones[w_name] += (w_value == 1) 
                num_zeros[w_name] += (w_value == 0)

        for w_name, w_value in user_state_dict.items():
            voted_results[w_name] = torch.where(num_ones[w_name] > 1/3*self.num_users, torch.ones_like(w_value), voted_results[w_name]) 
            voted_results[w_name] = torch.where(num_zeros[w_name] > 1/3*self.num_users, torch.zeros_like(w_value), voted_results[w_name])

        delta = WeightBuffer(voted_results)

        return delta