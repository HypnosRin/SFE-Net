import torch

from models.base_model import BaseModel


class DFCAN(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.hr = torch.rand(size=(16, 1, 224, 224))
        self.lr = torch.rand(size=(16, 1, 112, 112))
        self.sr = torch.rand(size=(16, 1, 224, 224))

    def feed_data(self, data):
        self.lr = data['lr'].to(self.device)
        if 'hr' in data.keys():
            self.hr = data['hr'].to(self.device)

    def test(self):
        self.network.eval()
        with torch.no_grad():
            self.sr = self.network(self.lr)
            self.loss = self.loss_function(self.hr, self.sr) if self.loss_function is not None else None
        self.network.train()

    def optimize_parameters(self):
        self.network.train()
        if self.opt['optimizer']['name'] in ('Adam', 'SGD'):
            self.optimizer.zero_grad()
            self.sr = self.network(self.lr)
            self.loss = self.loss_function(self.hr, self.sr)
            self.loss.backward()
            self.optimizer.step()
        else:
            raise NotImplementedError

