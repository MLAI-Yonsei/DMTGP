import torch
from torch import nn

# RMSE Loss 
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()
        self.mse = nn.MSELoss()
        self.eps = 1e-12

    def forward(self, target, pred):
        return torch.sqrt(self.mse(target, pred) + self.eps)

# MAPE Loss 
class MAPE(nn.Module):
    def __init__(self, eps=1e-12):
        super(MAPE,self).__init__()
        self.eps = eps

    def forward(self, target, pred):
        diff = torch.abs((target - pred) / (target + self.eps))
        mape = torch.mean(diff)
        return mape

# NME Loss 
class NME(nn.Module):
    def __init__(self, eps=1e-12):
        super(NME,self).__init__()
        self.eps = eps
        self.l1_loss = nn.L1Loss()

    def forward(self, target, pred):
        l1_loss = self.l1_loss(target, pred)
        nme = l1_loss * 100 / torch.sum(target)
        return nme

def get_all_metrics(metrics_list, target, pred):
    loss_list = []
    target = torch.Tensor(target)
    pred = torch.Tensor(pred)
    for metrics in metrics_list:
        loss = metrics(target, pred)
        loss_list.append(loss)
    return tuple(loss_list)