import matplotlib.pyplot as plt
import torch
from copy import deepcopy
from tqdm import tqdm
from utils.data import dataloader, idxs_of_classes, sample_dataset, dataset_split
from utils.model import vectorize_model
from src.client import Client


class Server(object):

    def __init__(self, net, trainset, testset, num_clients, classes=None, client_data_pct=0.1):
        self.net = deepcopy(net)
        self.clients = []
        self.iid_setting = isinstance(num_clients, int)
        if self.iid_setting:
            for _ in range(num_clients):
                client_trainset = sample_dataset(trainset, pct=client_data_pct)
                self.clients.append(Client(net, client_trainset, testset))
        else:
            self.groups = []
            for group_size, group_classes in zip(num_clients, classes):
                clients = []
                for _ in range(group_size):
                    client_trainset = sample_dataset(trainset, n=2500, classes=group_classes)
                    clients.append(Client(net, client_trainset, testset))
                group_testset = dataset_split(testset, idxs_of_classes(testset, group_classes))
                group_testset = sample_dataset(group_testset, n=2500)
                clients[0].testloader = dataloader(group_testset)
                self.groups.append(clients)
            self.clients = sum(self.groups, [])
    
    @property
    def test_loss_trackers(self):
        if self.iid_setting:
            return [self.clients[0].test_loss_tracker]
        return [group[0].test_loss_tracker for group in self.groups]
    
    @property
    def test_acc_trackers(self):
        if self.iid_setting:
            return [self.clients[0].test_acc_tracker]
        return [group[0].test_acc_tracker for group in self.groups]
    
    def _plot(self):
        # colors = ['#336699', '#86bbd8', '#2f4858']
        # colors = ['#295827', '#7bbf64', '#cbf6bb']
        colors = ['#291b85', '#6653e0', '#c6befa']
        # colors = ['#841a1a', '#e15454', '#fabdbd']
        fig, axs = plt.subplots(1, 2, figsize=(9, 3))
        fig.tight_layout()
        for t, c in zip(self.test_loss_trackers, colors):
            axs[0].plot(t, color=c)
        axs[0].set_title("Test loss")
        for t, c in zip(self.test_acc_trackers, colors):
            axs[1].plot(t, color=c)
        axs[1].set_title("Test accuracy")
        plt.show()
    
    def _sample_clients(self, pct):
        sample = lambda x, n: [x[i] for i in torch.randperm(len(x))[:n]]
        if self.iid_setting:
            n = int(pct * len(self.clients))
            return sample(self.clients, n)
        sampled = []
        for group in self.groups:
            n = int(pct * len(group))
            sampled += sample(group, n)
        return sampled
    
    def _average_model(self, clients, weights):
        state_dicts = [c.net.state_dict() for c in clients]
        avg = deepcopy(self.net.state_dict())
        for param in avg:
            values = torch.stack([s[param].data * w for s, w in zip(state_dicts, weights)])
            avg[param] = values.sum(0) / weights.sum()
        return avg
    
    def _calc_weights(self, clients):
        updates = torch.stack([vectorize_model(c.net) for c in clients])
        weights = torch.zeros(len(updates))
        for i, x in enumerate(updates):
            partition = updates[torch.arange(len(updates)) != i]
            weights[i] = ((partition - x)**4).sum()
        weights *= len(updates) / weights.sum()
        return weights
    
    def train(self, rounds=50, local_epochs=3, participation=0.1, weighted=False):
        weights = torch.ones((1, len(self._sample_clients(participation))))
        weight_decay = 0.9
        with tqdm(range(rounds)) as pbar:
            for i in pbar:
                pbar_desc = f"Round {i + 1}/{rounds}"
                round_clients = self._sample_clients(participation)
                for j, c in enumerate(round_clients):
                    pbar.set_description(f"{pbar_desc} (Training client {j + 1}/{len(round_clients)})\t")
                    for _ in range(local_epochs):
                        c.train()
                pbar.set_description(f"{pbar_desc} (Consolidating updates)\t")
                if weighted:
                    weights = torch.cat((weights, self._calc_weights(round_clients).unsqueeze(0)))
                decay = weight_decay ** torch.arange(len(weights)).flip(0)
                avg_weights = torch.sum(decay.unsqueeze(1) * weights, 0) / decay.sum()
                new_global_model = self._average_model(round_clients, weights=avg_weights)
                for c in self.clients:
                    c.net.load_state_dict(new_global_model)
                    c.optimizer.zero_grad()
                    c.optimizer.step()
                    c.scheduler.step()
                if self.iid_setting:
                    pbar.set_description(f"{pbar_desc} (Testing)\t\t\t")
                    self.clients[0].test()
                else:
                    for j, group in enumerate(self.groups):
                        pbar.set_description(f"{pbar_desc} (Testing group {j + 1}/{len(self.groups)})\t\t")
                        group[0].test()
        print(f"Avg:\n{avg_weights}")
        self._plot()
