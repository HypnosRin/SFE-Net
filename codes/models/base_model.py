import torch
import torch.nn as nn
from collections import OrderedDict

from networks import get_network
from optimizers import get_optimizer
from schedulers import get_scheduler
from loss_functions import get_loss_function


class BaseModel(object):
    """
    base class
    """

    def __init__(self, opt):
        """
        set device, network, optimizer, scheduler, loss function
        restore checkpoint
        set initial learning rate
        set DataParallel if needed
        print network description
        """
        self.opt = opt
        self.device = None

        self.network = None
        self.optimizer = None
        self.scheduler = None
        self.loss_function = None

        self.loss = None
        self.accum_loss = None

        self.device = torch.device('cuda' if opt['gpu'] else 'cpu')

        self.network = get_network(opt['network'])
        if opt['checkpoint'] is not None and opt['checkpoint']['network'] is not None:
            self.load_network(load_path=opt['checkpoint']['network'])
        self.network = self.network.to(self.device)

        if opt['checkpoint'] is not None and opt['checkpoint']['training_state'] is not None:
            opt['optimizer']['lr'] = self.restore_lr(opt['checkpoint']['training_state'])
        self.optimizer = get_optimizer(filter(lambda p: p.requires_grad, self.network.parameters()), opt['optimizer'])
        self.scheduler = get_scheduler(self.optimizer, opt['scheduler'])
        if opt['checkpoint'] is not None and opt['checkpoint']['training_state'] is not None:
            self.restored_epoch, self.restored_step = self.restore_epoch_step(opt['checkpoint']['training_state'])
            self.load_training_state(load_path=opt['checkpoint']['training_state'])
        else:
            self.restored_epoch, self.restored_step = None, None

        self.loss_function = get_loss_function(opt['loss_function'])

        if opt['gpu'] and opt['is_data_parallel']:
            self.network = torch.nn.DataParallel(self.network)

        desc = self.get_network_description()
        print(f'network contains {desc[1]} parameters, among which {desc[2]} parameters require gradient')

    def feed_data(self, data):
        pass

    def test(self):
        pass

    def optimize_parameters(self):
        pass

    def set_learning_rate(self, learning_rate):
        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate

    def get_current_learning_rate(self):
        if self.optimizer is not None:
            return self.optimizer.param_groups[0]['lr']
        else:
            return None

    def update_learning_rate(self, step):
        if self.scheduler is None:
            pass
        elif self.opt['scheduler']['name'] == 'CosineAnnealingLR':
            self.scheduler.step()
        elif self.opt['scheduler']['name'] == 'CosineAnnealingWarmRestarts':
            self.scheduler.step()
        elif self.opt['scheduler']['name'] == 'CosineAnnealingLR_Restart':
            self.scheduler.step()
        elif self.opt['scheduler']['name'] == 'ReduceLROnPlateau':
            if step is None:
                self.scheduler.step(metrics=float('inf'))
            else:
                if self.accum_loss is None:
                    self.accum_loss = []
                self.accum_loss.append(self.loss.item())
                if step % self.opt['scheduler']['step_interval'] == 0:
                    metric = sum(self.accum_loss) / len(self.accum_loss)
                    self.scheduler.step(metrics=metric)
                    print(f'ReduceLROnPlateau step with metric = {metric}')
                    self.accum_loss = []
        else:
            raise NotImplementedError

    def unpack_network(self):
        """
        remove unnecessary '.module'
        """
        if isinstance(self.network, nn.DataParallel) or isinstance(self.network, nn.parallel.DistributedDataParallel):
            return self.network.module
        else:
            return self.network

    def get_network_description(self):
        """get the string and total parameters of the network"""
        network = self.unpack_network()
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        m = sum(map(lambda x: x.numel(), filter(lambda p: p.requires_grad, network.parameters())))
        return s, n, m

    def save_network(self, save_path):
        network = self.unpack_network()
        torch.save(network.state_dict(), save_path)
        print(f'save network to {save_path}')

    def load_network(self, load_path):
        network = self.unpack_network()
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean)
        print(f'restore network from {load_path}')

    def save_training_state(self, save_path, epoch, step):
        state = {'epoch': epoch, 'step': step, 'lr': self.get_current_learning_rate(),
                 'optimizer': self.optimizer.state_dict() if self.optimizer is not None else None,
                 'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None}
        torch.save(state, save_path)
        print(f'save training state to {save_path}')

    def restore_lr(self, load_path):
        return torch.load(load_path)['lr']

    def restore_epoch_step(self, load_path):
        state = torch.load(load_path)
        return state['epoch'], state['step']

    def load_training_state(self, load_path):
        state = torch.load(load_path)
        if state['optimizer'] is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(state['optimizer'])
            for optim_state in self.optimizer.state.values():
                for k, v in optim_state.items():
                    if torch.is_tensor(v):
                        optim_state[k] = v.to(self.device)
        if state['scheduler'] is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(state['scheduler'])
        print(f'restore training state from {load_path}')
