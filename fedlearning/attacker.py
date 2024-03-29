import copy
import numpy as np
import torch 
from fedlearning.buffer import WeightBuffer
from fedlearning.compressors import compressor_registry
from fedlearning.datasets import fetch_dataloader
from fedlearning.dp.utils import prepare_dp
from deeplearning.utils import init_optimizer, AverageMeter
from fedlearning.dp.myattackerdp_optimizer import prepare_myattackerdp_optimizer

class Attacker(object):
    def __init__(self, config, model, **kwargs):
        """Construct a local updater for a user.
        """
        self.local_model = copy.deepcopy(model)
        self.w0 =  WeightBuffer(model.state_dict())
        self.optimizer = init_optimizer(config, self.local_model)
        self.local_epochs = config.local_epochs
        self.device = config.device
        self.config = config
        self.complete_attack = False
        self.init_compressor(config)

    def init_local_dataset(self, *args):
        pass

    def init_compressor(self, config):
        if config.attacker_compressor in compressor_registry.keys():
            self.compressor = compressor_registry[config.attacker_compressor](config)
        else:
            self.compressor = None

    def local_step(self):
        pass

    def uplink_transmit(self):
        delta = self.compute_delta()
        self.postprocessing(delta)
        return delta

    def postprocessing(self, delta):
        """Compress the local gradients.
        """
        if self.compressor is None:
            return
        gradient = delta.state_dict()
        for w_name, grad in gradient.items():
            gradient[w_name] = self.compressor.compress(grad)


class AttackerLocalUpdater(Attacker):
    def __init__(self, config, model, **kwargs):
        """Construct a local updater for a user.
        """
        super(AttackerLocalUpdater, self).__init__(config, model, **kwargs)

    def init_local_dataset(self, dataset, data_idx):
        subset = {"images":dataset.dst_train['images'][data_idx], "labels":dataset.dst_train['labels'][data_idx]}
        self.data_loader = fetch_dataloader(self.config, subset, shuffle=True)
        self.prepare_dp()

    def prepare_dp(self):
        self.local_model, self.optimizer, self.data_loader = prepare_dp(self.config, self.local_model, self.optimizer, self.data_loader)
        self.optimizer = prepare_myattackerdp_optimizer(self.config, self.optimizer)

    def local_step(self, criterion, **kwargs):
        """Perform local update tau times.

        Args,
            model(nn.module):       the global model
        """
       
        # for e in range(self.local_epochs):
        #     for _, contents in enumerate(self.data_loader):
        #         self.optimizer.zero_grad()
        #         target = contents[1].to(self.device)
        #         input = contents[0].to(self.device)

        #         # Compute output
        #         output = self.local_model(input)
        #         loss = criterion(output, target).mean()

        #         # Compute gradient and do SGD step
        #         loss.backward()
        #         self.optimizer.step()

        for e in range(self.local_epochs):
            images, labels = next(iter(self.data_loader))
            self.optimizer.zero_grad()

            # Compute output
            output = self.local_model(images.to(self.device))
            loss = criterion(output, labels.to(self.device)).mean()

            # Compute gradient and do SGD step
            loss.backward()
            self.optimizer.step()   
            self.optimizer.zero_grad()

    def compute_delta(self):
        """Simulate the transmission of local gradients to the central server.
        """ 
        w_tau = WeightBuffer(self.local_model.state_dict())
        delta = self.w0 - w_tau

        return delta
    
