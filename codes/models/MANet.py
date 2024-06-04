import torch

from models.base_model import BaseModel
from utils.universal_util import normalization


class MANet_Model(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.norm_lr = opt['norm_lr']
        self.times_1e4 = opt['times_1e4']
        self.lr = torch.rand(size=(16, 1, 224, 224))
        self.gt_kernel = torch.rand(size=(16, 33, 33))
        self.pred_kernel = torch.rand(size=(16, 224 * 224, 33, 33))

    def feed_data(self, data):
        self.lr = data['lr'].to(self.device)
        if 'kernel' in data.keys():
            self.gt_kernel = data['kernel'].to(self.device)

    def test(self):
        self.network.eval()
        with torch.no_grad():
            self.pred_kernel = self.network(normalization(self.lr, batch=True) if self.norm_lr else self.lr)
            # times 1e4 since kernel pixel values are very small
            weight = 1e4 if self.times_1e4 else 1.0
            self.loss = self.loss_function(self.pred_kernel * weight,
                                           self.gt_kernel.unsqueeze(1).expand(-1, self.pred_kernel.shape[1],
                                                                              -1, -1) * weight) \
                if self.loss_function is not None else None
        self.network.train()

    def optimize_parameters(self):
        self.network.train()
        if self.opt['optimizer']['name'] in ('Adam', 'SGD'):
            self.optimizer.zero_grad()
            self.pred_kernel = self.network(normalization(self.lr, batch=True) if self.norm_lr else self.lr)
            # times 1e4 since kernel pixel values are very small
            weight = 1e4 if self.times_1e4 else 1.0
            self.loss = self.loss_function(self.pred_kernel * weight,
                                           self.gt_kernel.unsqueeze(1).expand(-1, self.pred_kernel.shape[1],
                                                                              -1, -1) * weight)
            self.loss.backward()
            self.optimizer.step()
        else:
            raise NotImplementedError

    def psnr_heat_map(self, is_norm=False):
        gt_kernel = self.gt_kernel.squeeze(0)
        pred_kernel = self.pred_kernel.squeeze(0).view((self.lr.shape[-2] * self.network.scale,
                                                        self.lr.shape[-1] * self.network.scale)
                                                       + self.pred_kernel.shape[-2:])
        # gt_kernel: (h, w)
        # pred_kernel: (H, W, h, w)
        if is_norm:
            gt_kernel = normalization(gt_kernel)
            max_val = torch.max(torch.max(pred_kernel, dim=-2, keepdim=True).values, dim=-1, keepdim=True).values
            min_val = torch.min(torch.min(pred_kernel, dim=-2, keepdim=True).values, dim=-1, keepdim=True).values
            pred_kernel = (pred_kernel - min_val) / (max_val - min_val)
        gt_kernel = gt_kernel.expand((pred_kernel.shape[0], pred_kernel.shape[1], -1, -1))
        mse = torch.mean((gt_kernel - pred_kernel) ** 2, dim=(-2, -1))
        max_v = torch.maximum(torch.max(torch.max(gt_kernel, dim=-1).values, dim=-1).values,
                              torch.max(torch.max(pred_kernel, dim=-1).values, dim=-1).values)
        return 20 * torch.log10(max_v / torch.sqrt(mse))  # heat map of shape (H, W)
