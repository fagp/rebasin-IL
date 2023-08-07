import torch
from copy import deepcopy
import numpy as np


class ILLambda05(torch.nn.Module):
    def __init__(
        self,
        criterion=None,
        gamma=0.05,
        beta=1e-2,
        gradient_clip=0.001,
        modela=None,
    ):
        super(ILLambda05, self).__init__()
        # residual optimization parameters
        self.lr = gamma
        self.wd = beta
        self.gclip = gradient_clip

        # base criterion for the task
        self.criterion = criterion if criterion is not None else torch.nn.MSELoss()

        # unchanged model
        self.modela = self.set_model(modela)

        # residual model
        self.delta = deepcopy(self.modela)
        for p in self.delta.parameters():
            p.data = torch.randn(p.shape).to(p.data.device) * 0.0001
            p.requires_grad = True

        # optimizer of the residual
        self.optim = torch.optim.SGD(
            self.delta.parameters(), lr=self.lr, weight_decay=self.wd
        )

    def set_model(self, modela):
        self.modela = deepcopy(modela)
        for p in self.modela.parameters():
            p.requires_grad = False
        return self.modela

    def optim_step(self):
        # apply gradient clipping
        for p in self.delta.parameters():
            if p.grad is not None:
                p.grad.data = torch.clip(p.grad.data, -self.gclip, self.gclip)

        # residual's optimizer step
        self.optim.step()

    def forward(self, modelb, input, target, task_id=None):
        # zero_grad residual
        self.optim.zero_grad()

        for p1, p2, p3 in zip(
            modelb.parameters(), self.modela.parameters(), self.delta.parameters()
        ):
            p1.add_(p2.data)
            p1.mul_(0.5)
            p1.add_(p3)

        loss = 0
        uniques_ids = np.unique(task_id.cpu().numpy()).tolist()
        for task in uniques_ids:
            (idx,) = torch.where(task_id == task)

            z = modelb(input[idx, ...])
            loss += self.criterion(z, target[idx, ...])

        return loss
