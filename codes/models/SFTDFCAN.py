import torch

from models.base_model import BaseModel
from utils.universal_util import pickle_load, PCA_Encoder, normalization


class SFTDFCAN(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.norm_lr = opt['norm_lr']
        self.norm_hr = opt['norm_hr']
        self.hr = torch.rand(size=(16, 1, 224, 224))
        self.lr = torch.rand(size=(16, 1, 224, 224))
        self.sr = torch.rand(size=(16, 1, 224, 224))
        self.kernel = torch.rand(size=(16, 33, 33))  # (B, l, l)
        self.pca_encoder = PCA_Encoder(weight=pickle_load(opt['pca_matrix']).to(self.device),
                                       mean=pickle_load(opt['pca_mean']).to(self.device))
        self.kernel_code = torch.rand(size=(16, 25))  # (B, h)

    def feed_data(self, data):
        self.lr = data['lr'].to(self.device)
        if 'hr' in data.keys():
            self.hr = data['hr'].to(self.device)
        if 'kernel_code' in data.keys():
            self.kernel = None
            self.kernel_code = data['kernel_code'].to(self.device)
        elif 'kernel' in data.keys():
            self.kernel = data['kernel'].to(self.device)
            self.kernel_code = self.pca_encoder(self.kernel)

    def test(self):
        self.network.eval()
        with torch.no_grad():
            self.sr = self.network(normalization(self.lr, batch=True) if self.norm_lr else self.lr, self.kernel_code)
            self.loss = self.loss_function(normalization(self.hr, batch=True) if self.norm_hr else self.hr, self.sr) \
                if self.loss_function is not None else None
        self.network.train()

    def optimize_parameters(self):
        self.network.train()
        if self.opt['optimizer']['name'] in ('Adam', 'SGD'):
            self.optimizer.zero_grad()
            self.sr = self.network(normalization(self.lr, batch=True) if self.norm_lr else self.lr, self.kernel_code)
            self.loss = self.loss_function(normalization(self.hr, batch=True) if self.norm_hr else self.hr, self.sr)
            self.loss.backward()
            self.optimizer.step()
        else:
            raise NotImplementedError
