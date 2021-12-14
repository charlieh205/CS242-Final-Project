from copy import deepcopy
from torch import nn, optim
import torch
from utils.data import dataloader


class Client(object):

    def __init__(self, net, trainset, testset, lr=0.01, scheduler_step=13, batch_size=100):
        self.net = deepcopy(net)
        self.trainloader = dataloader(trainset, batch_size=batch_size)
        self.testloader = dataloader(testset, batch_size=batch_size)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step, gamma=0.2)
        self.train_loss_tracker = []
        self.train_acc_tracker = []
        self.test_loss_tracker = []
        self.test_acc_tracker = []

    def train(self):
        self.net.train()
        correct, total = 0, 0
        for inputs, targets in self.trainloader:
            # calculate loss on batch and backprop
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            loss.backward()
            self.optimizer.step()
            # track loss and accuracy
            self.train_loss_tracker.append(loss.item())
            _, pred = outputs.max(1)
            total += len(targets)
            correct += (pred == targets).sum().item()
        acc = correct / total
        self.train_acc_tracker.append(acc)
    
    def test(self):
        self.net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in self.testloader:
                # calculate loss on batch
                outputs = self.net(inputs)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                # track loss and accuracy
                self.test_loss_tracker.append(loss.item())
                _, pred = outputs.max(1)
                total += len(targets)
                correct += (pred == targets).sum().item()
        acc = correct / total
        self.test_acc_tracker.append(acc)
