# Max-Heinrich Laves
# Institute of Mechatronic Systems
# Leibniz Universität Hannover, Germany
# 2019

import torch
import torchvision
from utils import Identity


def kld_loss(mean, log_var):
    """
    see Appendix B from VAE paper:
    Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    https://arxiv.org/abs/1312.6114
    0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    :param mean: vector of latent space mean values
    :param log_var: vector of latent space log variances
    :return: loss value, normalized by batch size
    """

    kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return kld / mean.size(0)  # norm by batch size


class BaselineResNet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self._resnet = torchvision.models.resnet18(pretrained=True)
        self._resnet.fc = torch.nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x, dropout=False, p=0.5):
        y = self._resnet(x)
        return y


class BayesianResNet1(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self._resnet = torchvision.models.resnet18(pretrained=True)
        self._resnet.fc = Identity()
        self._fc = torch.nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x, dropout=False, p=0.5):

        x = self._resnet(x)

        # apply dropout at test time
        if dropout:
            x = torch.nn.functional.dropout(x, p=p, training=True, inplace=False)

        y = self._fc(x)

        return y


class BayesianResNet2(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self._resnet = torchvision.models.resnet18(pretrained=True)
        self._resnet.fc = torch.nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x, dropout=False, p=0.5):
        x = self._resnet.conv1(x)
        x = self._resnet.bn1(x)
        x = self._resnet.relu(x)
        x = self._resnet.maxpool(x)

        if dropout:
            x = torch.nn.functional.dropout(x, p=p, training=True, inplace=False)
        x = self._resnet.layer1(x)

        if dropout:
            x = torch.nn.functional.dropout(x, p=p, training=True, inplace=False)
        x = self._resnet.layer2(x)

        if dropout:
            x = torch.nn.functional.dropout(x, p=p, training=True, inplace=False)
        x = self._resnet.layer3(x)

        if dropout:
            x = torch.nn.functional.dropout(x, p=p, training=True, inplace=False)
        x = self._resnet.layer4(x)

        x = self._resnet.avgpool(x)
        x = x.view(x.size(0), -1)

        if dropout:
            x = torch.nn.functional.dropout(x, p=p, training=True, inplace=False)
        y = self._resnet.fc(x)

        return y


class ProbabilisticResNet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self._resnet = torchvision.models.resnet18(pretrained=True)
        self._resnet.fc = Identity()

        self._linear_means = torch.nn.Linear(512, num_classes)
        self._linear_log_vars = torch.nn.Linear(512, num_classes)

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        x = self._resnet(x)

        means = self._linear_means(x)
        log_vars = self._linear_log_vars(x)

        if self.training:
            y = self.reparameterize(means, log_vars)
        else:
            y = means

        # y = self.reparameterize(means, log_vars)

        return y, means, log_vars


class UncertNet(torch.nn.Module):
    def __init__(self, in_classes, out_classes=2, hidden_size=32):
        super().__init__()

        self._fc1 = torch.nn.Linear(in_features=in_classes, out_features=hidden_size)
        self._fc2 = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self._fc3 = torch.nn.Linear(in_features=hidden_size, out_features=out_classes)
        self._bn1 = torch.nn.BatchNorm1d(hidden_size)
        self._bn2 = torch.nn.BatchNorm1d(hidden_size)
        self._relu = torch.nn.ReLU()

    def forward(self, x):
        x = self._fc1(x)
        x = self._bn1(x)
        x = self._relu(x)

        x = self._fc2(x)
        x = self._bn2(x)
        x = self._relu(x)

        y = self._fc3(x)

        return y
