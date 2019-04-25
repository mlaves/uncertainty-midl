#!/usr/bin/env python3

# Max-Heinrich Laves
# Institute of Mechatronic Systems
# Leibniz Universität Hannover, Germany
# 2019

import torch
import numpy as np
from models import BayesianResNet1, BayesianResNet2, ProbabilisticResNet, UncertNet
from torch.utils.data import DataLoader
import argparse
import tqdm
from data_generator import KermanyDataset
from utils import accuracy
from sklearn import metrics


# enable cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Train UncertNet.')
parser.add_argument('--model', metavar='M', type=str, help='Specify the model M',
                    choices=['bayesian1', 'bayesian2', 'probabilistic'])
parser.add_argument('--snapshot_resnet', metavar='SR', type=str, help='Specify the model snapshot for ResNet')
parser.add_argument('--snapshot_uncert', metavar='SU', type=str, help='Specify the model snapshot for UncertNet')
parser.add_argument('--bs', metavar='N', type=int, help='Specify the batch size N', default=1,
                    choices=list(range(1, 65)))
args = parser.parse_args()

print("Test model:", args.model)
print("Test with batch_size:", str(args.bs))

# properties
batch_size = args.bs
num_classes = 4
bayesian_dropout_p = 0.5
num_workers = 8 if batch_size > 8 else batch_size
num_mc = 100

color = True
resize_to = (224, 224)

dataset_test = KermanyDataset("/home/laves/Downloads/OCT2017_3/train",
                              crop_to=(384, 384), resize_to=resize_to, color=color)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=num_workers)

assert len(dataset_test) > 0

print("Test dataset length:", len(dataset_test))
print('')

# create a model
resnet = torch.nn.Module()
if args.model == 'bayesian1':
    resnet = BayesianResNet1(num_classes=num_classes).to(device)
elif args.model == 'bayesian2':
    resnet = BayesianResNet2(num_classes=num_classes).to(device)
elif args.model == 'probabilistic':
    resnet = ProbabilisticResNet(num_classes=num_classes).to(device)
else:
    assert False

# load weights for resnet
checkpoint = torch.load(args.snapshot_resnet, map_location=device)
print("Loading previous weights at epoch " + str(checkpoint['epoch']) + " from " + args.snapshot_resnet)
resnet.load_state_dict(checkpoint['state_dict'])

# create uncertnet model here
model = UncertNet(in_classes=num_classes).to(device)

# load weights for uncertnet
checkpoint = torch.load(args.snapshot_uncert, map_location=device)
print("Loading previous weights at epoch " + str(checkpoint['epoch']) + " from " + args.snapshot_uncert)
model.load_state_dict(checkpoint['state_dict'])

print('')

# save accuracies and losses during training
y_true = []
y_pred = []
test_acc = []

resnet.eval()
model.eval()

with torch.no_grad():
    batches = tqdm.tqdm(dataloader_test)

    for x_resnet, y_resnet in batches:
        with torch.no_grad():
            x_resnet, y_resnet = x_resnet.to(device), y_resnet.to(device)

            if args.model in ['bayesian1', 'bayesian2']:
                y_resnet_pred = resnet(x_resnet, dropout=True, p=bayesian_dropout_p)
                mc_output = y_resnet_pred.softmax(1).unsqueeze(1)

                for mc in range(num_mc - 1):
                    y_resnet_pred = resnet(x_resnet, dropout=True, p=bayesian_dropout_p).softmax(1).unsqueeze(1)
                    mc_output = torch.cat((mc_output, y_resnet_pred), dim=1)

                mean = mc_output.mean(dim=1)
                var = mc_output.var(dim=1)
            elif args.model == 'probabilistic':
                y_resnet_pred, pred_mean, log_var = resnet(x_resnet)
                mc_output = y_resnet_pred.softmax(1).unsqueeze(1)

                for mc in range(num_mc - 1):
                    y_resnet_pred = resnet.reparameterize(pred_mean, log_var).softmax(1).unsqueeze(1)
                    mc_output = torch.cat((mc_output, y_resnet_pred), dim=1)

                mean = mc_output.mean(dim=1)
                var = mc_output.var(dim=1)
            else:
                assert False

        x_uncert = var.detach()
        y_uncert = torch.zeros(x_resnet.size(0)).long().to(device)
        for i in range(x_resnet.size(0)):
            if mean[i].argmax() == y_resnet[i]:
                y_uncert[i] = 0
            else:
                y_uncert[i] = 1

        # train uncertnet here
        y_uncert_pred = model(x_uncert)

        y_true.append(y_uncert.data.cpu().numpy())
        y_pred.append(y_uncert_pred.argmax(1).data.cpu().numpy())

        # calculate batch train accuracy
        batch_acc = accuracy(y_uncert_pred, y_uncert)
        test_acc.append(batch_acc)

        # print current loss
        batches.set_description("a: {:4f}".format(batch_acc))

    print('')

print('acc', np.mean(test_acc))
print(metrics.classification_report(np.array(y_true), np.array(y_pred)))
print(metrics.confusion_matrix(np.array(y_true), np.array(y_pred)))

