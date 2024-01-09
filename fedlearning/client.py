import copy
import numpy as np
import torch 
from fedlearning.buffer import WeightBuffer
from fedlearning.compressors import compressor_registry
from fedlearning.datasets import fetch_dataloader
from fedlearning.dp.utils import prepare_dp
from deeplearning.utils import init_optimizer, AverageMeter, accuracy
from fedlearning.dp.my_dp_optimizer import prepare_mydp_optimizer


class Client(object):
    def __init__(self, config, model, **kwargs):
        """Construct a local updater for a user.
        """
        self.local_model = copy.deepcopy(model)
        self.w0 =  WeightBuffer(model)
        self.optimizer = init_optimizer(config, self.local_model)
        self.local_epochs = config.local_epochs
        self.device = config.device
        self.config = config
        self.complete_attack = False
        self.init_compressor(config)

    def init_local_dataset(self, *args):
        pass

    def init_compressor(self, config):
        if config.compressor in compressor_registry.keys():
            self.compressor = compressor_registry[config.compressor](config)
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
            if delta.freeze_weight[w_name]: continue
            gradient[w_name] = self.compressor.compress(grad)


class LocalUpdater(Client):
    def __init__(self, config, model, **kwargs):
        """Construct a local updater for a user.
        """
        super(LocalUpdater, self).__init__(config, model, **kwargs)

    def init_local_dataset(self, dataset, data_idx):
        
        if self.config.pytorch_builtin:
            subset = torch.utils.data.Subset(dataset.dst_train, data_idx)
            self.data_loader = fetch_dataloader(self.config, subset, shuffle=True)
        else:
            subset = {"images":dataset.dst_train['images'][data_idx], "labels":dataset.dst_train['labels'][data_idx]}
            self.data_loader = fetch_dataloader(self.config, subset, shuffle=True)
        
        if self.config.dp_on:
            self.prepare_dp()

    def prepare_dp(self):
        self.local_model, self.optimizer, self.data_loader = prepare_dp(self.config, self.local_model, self.optimizer, self.data_loader)
        self.optimizer = prepare_mydp_optimizer(self.config, self.optimizer)

    def local_step(self, criterion, **kwargs):
        """Perform local update tau times.

        Args,
            model(nn.module):       the global model
        """
        accuracy_meter = AverageMeter('Accs', ':6.2f')
        # for e in range(self.local_epochs):
        #     for i, contents in enumerate(self.data_loader):
        #         self.optimizer.zero_grad()
        #         labels = contents[1].to(self.device)
        #         images = contents[0].to(self.device)

        #         # Compute output
        #         output = self.local_model(images)
        #         loss = criterion(output, labels).mean()

        #         # Compute gradient and do SGD step
        #         loss.backward()
        #         self.optimizer.step()

        #         # Measure accuracy and record loss
        #         acc = accuracy(output.data, labels.to(self.device), topk=(1,))[0]
        #         accuracy_meter.update(acc.item(), images.size(0))

        #         if i % 10 == 0:
        #             print(
        #         'Acc {accuracy.val:.3f} ({accuracy.avg:.3f}) [sens]'.format(
        #             accuracy=accuracy_meter))


        iter_idx = 0

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

            # Measure accuracy and record loss
            acc = accuracy(output.data, labels.to(self.device), topk=(1,))[0]
            accuracy_meter.update(acc.item(), images.size(0))

            if iter_idx % 10 == 0:
                print(
            'Acc {accuracy.val:.3f} ({accuracy.avg:.3f}) [sens]'.format(
                accuracy=accuracy_meter))
            iter_idx += 1

    def compute_delta(self):
        """Simulate the transmission of local gradients to the central server.
        """ 
        w_tau = WeightBuffer(self.local_model)
        delta = self.w0 - w_tau

        return delta
    
