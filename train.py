#!/usr/bin/env python3

# Max-Heinrich Laves
# Institute of Mechatronic Systems
# Leibniz Universität Hannover, Germany
# 2019

import matplotlib
matplotlib.use('Agg')  # headless plotting
import matplotlib.pyplot as plt
import torch
from models import BaselineResNet, BayesianResNet1, BayesianResNet2, ProbabilisticResNet, kld_loss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import tqdm
import numpy as np
from data_generator import KermanyDataset
from utils import accuracy
from tensorboardX import SummaryWriter
import os


# enable cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Train solitary classifier on small OCT dataset for comparison.')
parser.add_argument('--model', metavar='M', type=str, help='Specify the model M',
                    choices=['baseline', 'bayesian1', 'bayesian2', 'probabilistic'])
parser.add_argument('--bs', metavar='N', type=int, help='Specify the batch size N', default=1,
                    choices=list(range(1, 129)))
parser.add_argument('--epochs', metavar='E', type=int, help='Specify the number of epochs E', default=100,
                    choices=list(range(1, 201)))
args = parser.parse_args()

print("Train model:", args.model)
print("Train with batch_size: " + str(args.bs))
print("Training epochs: " + str(args.epochs))
print('')

# setup tensorboardx
writer = SummaryWriter()

if not os.path.exists("./snapshots"):
    os.makedirs("./snapshots")

# dimension properties
batch_size = args.bs
val_batch_size = batch_size
num_workers = 8 if batch_size > 8 else batch_size
num_classes = 4
bayesian_dropout_p = 0.5
lambda_prop = 0.01

color = True
resize_to = (224, 224)

dataset_train = KermanyDataset("/home/laves/Downloads/OCT2017_3/test",
                               crop_to=(384, 384), resize_to=resize_to, color=color)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

dataset_valid = KermanyDataset("/home/laves/Downloads/OCT2017_3/val",
                               crop_to=(384, 384), resize_to=resize_to, color=color)
dataloader_valid = DataLoader(dataset_valid, batch_size=val_batch_size, num_workers=num_workers)

assert len(dataset_train) > 0
assert len(dataset_valid) > 0

print("Train dataset length:", len(dataset_train))
print("Valid dataset length:", len(dataset_valid))
print('')

# create a model
model = torch.nn.Module()
if args.model == 'baseline':
    model = BaselineResNet(num_classes=num_classes).to(device)
elif args.model == 'bayesian1':
    model = BayesianResNet1(num_classes=num_classes).to(device)
elif args.model == 'bayesian2':
    model = BayesianResNet2(num_classes=num_classes).to(device)
elif args.model == 'probabilistic':
    model = ProbabilisticResNet(num_classes=num_classes).to(device)
else:
    assert False

# calculate number of trainable parameters
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Total trainable parameters: {:,}".format(params))

# create your optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,
                             betas=(0.9, 0.999),
                             weight_decay=1e-8)
lr_scheduler = ReduceLROnPlateau(optimizer, patience=5)

# create loss function
criterion = torch.nn.CrossEntropyLoss()

print('')  # print empty line before training output

# save accuracies and losses during training
train_losses = []
train_accuracies = []
valid_losses = []
valid_accuracies = []

start_epoch = 0
epochs = args.epochs
e = 0
batch_counter = 0
batch_counter_valid = 0

for e in range(start_epoch, epochs):

    # go through training set
    model.train()
    print("lr =", optimizer.param_groups[0]['lr'])

    epoch_train_loss = []
    epoch_train_acc = []
    is_best = False

    x, y, y_pred = None, None, None
    batches = tqdm.tqdm(dataloader_train)
    for x, y in batches:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        if args.model == 'baseline':
            y_pred = model(x)
            train_loss = criterion(y_pred, y)
        elif args.model in ['bayesian1', 'bayesian2']:
            y_pred = model(x, dropout=True, p=bayesian_dropout_p)
            train_loss = criterion(y_pred, y)
        elif args.model == 'probabilistic':
            y_pred, means, log_vars = model(x)
            train_loss = criterion(y_pred, y) + lambda_prop * kld_loss(means, log_vars)
        else:
            assert False

        train_loss.backward()
        optimizer.step()

        # print current loss
        batches.set_description("loss: {:4f}".format(train_loss.item()))

        # sum epoch loss
        epoch_train_loss.append(train_loss.item())

        # calculate batch train accuracy
        batch_acc = accuracy(y_pred, y)
        epoch_train_acc.append(batch_acc)

        writer.add_scalar('data/train_loss', train_loss.item(), batch_counter)
        writer.add_scalar('data/train_acc', batch_acc, batch_counter)
        batch_counter += 1

    epoch_train_loss = np.mean(epoch_train_loss)
    epoch_train_acc = np.mean(epoch_train_acc)
    lr_scheduler.step(epoch_train_loss)

    # go through validation set
    model.eval()
    with torch.no_grad():

        epoch_valid_loss = []
        epoch_valid_acc = []

        batches = tqdm.tqdm(dataloader_valid)
        for x, y in batches:
            x, y = x.to(device), y.to(device)

            if args.model == 'baseline':
                y_pred = model(x)
                valid_loss = criterion(y_pred, y)
            elif args.model in ['bayesian1', 'bayesian2']:
                y_pred = model(x, dropout=True, p=bayesian_dropout_p)
                valid_loss = criterion(y_pred, y)
            elif args.model == 'probabilistic':
                y_pred, means, log_vars = model(x)
                valid_loss = criterion(y_pred, y) + lambda_prop * kld_loss(means, log_vars)
            else:
                assert False

            # print current loss
            batches.set_description("loss: {:4f}".format(valid_loss.item()))

            # sum epoch loss
            epoch_valid_loss.append(valid_loss.item())

            # calculate batch train accuracy
            batch_acc = accuracy(y_pred, y)
            epoch_valid_acc.append(batch_acc)

            writer.add_scalar('data/valid_classifier_loss', valid_loss.item(), batch_counter_valid)
            writer.add_scalar('data/valid_acc', batch_acc, batch_counter_valid)
            batch_counter_valid += 1

    epoch_valid_loss = np.mean(epoch_valid_loss)
    epoch_valid_acc = np.mean(epoch_valid_acc)

    print("Epoch {:d}: loss: {:4f}, acc: {:4f}, val_loss: {:4f}, val_acc: {:4f}"
          .format(e,
                  epoch_train_loss,
                  epoch_train_acc,
                  epoch_valid_loss,
                  epoch_valid_acc,
                  ))

    # save epoch losses
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_acc)
    valid_losses.append(epoch_valid_loss)
    valid_accuracies.append(epoch_valid_acc)

    if valid_losses[-1] <= np.min(valid_losses):
        is_best = True

    if is_best:
        filename = "./snapshots/" + args.model + "_best.pth.tar"
        print("Saving best weights so far with val_loss: {:4f}".format(valid_losses[-1]))
        torch.save({
            'epoch': e,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_losses': train_losses,
            'train_accs': train_accuracies,
            'val_losses': valid_losses,
            'val_accs': valid_accuracies,
        }, filename)

    if e == epochs-1:
        filename = "./snapshots/" + args.model + "_" + str(e) + ".pth.tar"
        print("Saving weights at epoch {:d}".format(e))
        torch.save({
            'epoch': e,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_losses': train_losses,
            'train_accs': train_accuracies,
            'val_losses': valid_losses,
            'val_accs': valid_accuracies,
            }, filename)

    print('')

    # plot losses
    plt.figure()
    plt.plot(range(len(train_losses)), train_losses, marker='x')
    plt.plot(range(len(valid_losses)), valid_losses, marker='x')
    plt.title(args.model+" loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(args.model+"_loss.pdf", dpi=300)
    plt.figure()
    plt.plot(range(len(train_accuracies)), train_accuracies, marker='x')
    plt.plot(range(len(valid_accuracies)), valid_accuracies, marker='x')
    plt.title(args.model+" accuracy")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.savefig(args.model+"_acc.pdf", dpi=300)
    plt.close('all')
