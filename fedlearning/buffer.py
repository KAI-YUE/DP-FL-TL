import copy

# PyTorch Libraries
import torch
import torch.nn as nn

class WeightBuffer(object):
    def __init__(self, model=None, mode="copy", **kwargs):
        self.freeze_weight = {}
        if model is None:
            try:
                self._weight_dict = copy.deepcopy(kwargs["weight_dict"])
                self.freeze_weight = copy.deepcopy(kwargs["freeze_weight"])
            except:
                raise NotImplementedError("WeightBuffer must be initialized with a model or weight_dict")
        else:
            self._weight_dict = copy.deepcopy(model.state_dict())
            for ((w_name, w_value), w) in zip(self._weight_dict.items(), model.parameters()):
                self.freeze_weight[w_name] = not w.requires_grad

            if mode == "zeros":
                for w_name, w_value in self._weight_dict.items():
                    self._weight_dict[w_name].data = torch.zeros_like(w_value)
        
        
    def __add__(self, weight_buffer):
        weight_dict = copy.deepcopy(self._weight_dict)
        key = parse_dp(weight_buffer)
        for w_name, w_value in weight_dict.items():
            if self.freeze_weight[w_name]: continue
            weight_dict[w_name].data = self._weight_dict[w_name].data + weight_buffer._weight_dict[key+w_name].data

        return WeightBuffer(weight_dict=weight_dict, freeze_weight=self.freeze_weight)

    def __sub__(self, weight_buffer):
        weight_dict = copy.deepcopy(self._weight_dict)
        key = parse_dp(weight_buffer)
        for w_name, w_value in weight_dict.items():
            if self.freeze_weight[w_name]: continue
            weight_dict[w_name].data = self._weight_dict[w_name].data - weight_buffer._weight_dict[key+w_name].data

        return WeightBuffer(weight_dict=weight_dict, freeze_weight=self.freeze_weight)

    def __mul__(self,rhs):
        weight_dict = copy.deepcopy(self._weight_dict)
        for w_name, w_value in weight_dict.items():
            if self.freeze_weight[w_name]: continue
            weight_dict[w_name].data = rhs*self._weight_dict[w_name].data

        return WeightBuffer(weight_dict=weight_dict, freeze_weight=self.freeze_weight)

    def push(self, weight_dict):
        self._weight_dict = copy.deepcopy(weight_dict)

    def state_dict(self):
        return self._weight_dict

# designed for opacus model weight
def parse_dp(weight_buffer):
    if "_module." in list(weight_buffer._weight_dict.keys())[0]:
        key = "_module."
    else:
        key = ""
    return key

def _get_para(state_dict) -> torch.Tensor:
    # flat concatenation of all parameters
    layer_parameters = []

    for name, param in state_dict.items():      
        layer_parameters.append(param.data.view(-1))

    return torch.cat(layer_parameters)