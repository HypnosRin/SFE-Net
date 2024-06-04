import torch

from models.base_model import BaseModel
from utils.universal_util import normalization


class UNetBased_Model(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.norm_lr = opt['norm_lr']
        self.norm_k = opt['norm_k']
        self.lr = torch.rand(size=(16, 1, 224, 224))
        self.gt_kernel = torch.rand(size=(16, 33, 33))
        self.pred_kernel = torch.rand(size=(16, 1, 33, 33))

    def feed_data(self, data):
        self.lr = data['lr'].to(self.device)
        if 'kernel' in data.keys():
            self.gt_kernel = data['kernel'].to(self.device)

    def test(self):
        self.network.eval()
        with torch.no_grad():
            self.pred_kernel = self.network(normalization(self.lr, batch=True) if self.norm_lr else self.lr)
            self.loss = self.loss_function(self.pred_kernel.squeeze(1),
                                           normalization(self.gt_kernel, batch=True) if self.norm_k else self.gt_kernel) \
                if self.loss_function is not None else None
            if self.norm_k:
                self.pred_kernel = normalization(self.pred_kernel, batch=True)
                self.pred_kernel /= torch.sum(self.pred_kernel, dim=(-2, -1), keepdim=True)
        self.network.train()

    def optimize_parameters(self):
        self.network.train()
        if self.opt['optimizer']['name'] in ('Adam', 'SGD'):
            self.optimizer.zero_grad()
            self.pred_kernel = self.network(normalization(self.lr, batch=True) if self.norm_lr else self.lr)
            self.loss = self.loss_function(self.pred_kernel.squeeze(1),
                                           normalization(self.gt_kernel, batch=True) if self.norm_k else self.gt_kernel)
            self.loss.backward()
            self.optimizer.step()
        else:
            raise NotImplementedError
