import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR


class BetaNet(nn.Module):
    """
    Parameters
    ----------
    n_groups : int
        Number of sensitive groups.

    sigma : float, default=15
        Scale parameter of the plug-in model.
    """

    def __init__(self, n_groups, sigma=15):
        super(BetaNet, self).__init__()
        self.sigma = sigma
        self.beta = nn.parameter.Parameter(torch.zeros(n_groups, 1))

    def forward(self, prior, group):
        """
        Parameters
        ----------
        prior : array_like, shape (n_samples, 1)
            Predict probability outputed by the pretrained model.

        group : array_like, shape (n_samples, n_groups)
            One-hot encoding of the sensitive attribute.

        Return
        ------
        shift_proba : array_like, shape (n_samples, 1)
            The calibrated probability.
        """
        logit = torch.logit(prior)
        shift_logit = logit + group @ self.beta
        shift_proba = torch.sigmoid(self.sigma * shift_logit)
        return shift_proba


class GroupWiseMetric(nn.Module):
    def __init__(self):
        super(GroupWiseMetric, self).__init__()

    def forward(self, pred, group, cond):
        l = []
        for i in range(cond.size(1)):
            cond_i = cond[:, [i]]
            value = torch.div(
                group.T @ (cond_i * pred),
                group.T @ cond_i,
            )
            diff = value.max() - value.min()
            l.append(diff * cond_i.sum())
        return torch.stack(l).sum() / cond.sum()


class Thresholding(object):
    """
    Parameters
    ----------
    alpha : float, default=1.0
        Number of sensitive groups.

    epoch : int, default=100
        Number of iteration. Must be a positive value.

    lr : float, default=1.0
        Learning rate.

    device : str, default='cpu'
        Set torch device.
    """

    def __init__(self, alpha=1.0, epoch=100, lr=1.0, device="cpu"):
        self.alpha = alpha
        self.epoch = epoch
        self.lr = lr
        self.device = device

    def fit(self, prior, group, cond):
        """
        Parameters
        ----------
        prior : array_like, shape (n_samples, n_groups)
            Predict probability outputed by the pretrained model.

        group : array_like, shape (n_samples, n_groups) or (n_samples, 1)
            Sensitive attribute or its one-hot encoding.

        cond : array_like, shape (n_samples, n_conds)
            One-hot encoding of fairness condition.

        Returns
        -------
        self : returns an instance of self.
        """

        if group.shape[1] == 1:
            self.n_groups = len(np.unique(group))
            group = np.eye(self.n_groups)[group.flatten()]
        else:
            self.n_groups = group.shape[1]

        weight = torch.tensor(group.mean(0)).to(self.device)
        prior = torch.FloatTensor(prior).to(self.device)
        group = torch.FloatTensor(group).to(self.device)
        cond = torch.FloatTensor(cond).to(self.device)

        self.net = BetaNet(self.n_groups).to(self.device)
        optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
        scheduler = ExponentialLR(optimizer, gamma=0.9)
        fair_loss = GroupWiseMetric()

        bayes_pred = torch.where(prior > 0.5, 1.0, 0.0)
        bayes_acc = self.accuracy(prior, bayes_pred)
        lmbda = torch.zeros(1).to(self.device)

        self.net.train()
        for epoch in range(self.epoch):

            optimizer.zero_grad()

            pred = self.net(prior, group)
            fair = fair_loss(pred, group, cond)
            reg = (torch.pow(self.net.beta, 2) * weight).sum()
            loss = fair + lmbda * reg

            loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % 5 == 4:
                with torch.no_grad():
                    pred = self.net(prior, group)
                    cur_pred = torch.where(pred > 0.5, 1.0, 0.0)
                    cur_acc = self.accuracy(prior, cur_pred)
                    lmbda += max(0, bayes_acc * 0.99 - cur_acc) * self.alpha

        return self

    def accuracy(self, prior, pred):
        acc = (prior * pred + (1 - prior) * (1 - pred)).mean()
        return acc

    def predict(self, prior, group):
        if group.shape[1] == 1:
            group = np.eye(self.n_groups)[group.flatten()]
        prior = torch.FloatTensor(prior).to(self.device)
        group = torch.FloatTensor(group).to(self.device)

        self.net.eval()
        with torch.no_grad():
            pred = self.net(prior, group)
            y_hat = torch.where(pred > 0.5, 1.0, 0.0).cpu().numpy()
        return y_hat
