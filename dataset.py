import matplotlib.pyplot as plt
import math
import numpy as np

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import Dataset


class Dataset2D(Dataset):
    def __init__(self, class_size=100):
        self.size = class_size
        self.data = self.generate_spiral_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def transform_matrix_to_torch_tensor(data):
        instances = len(data[0])
        torch_data = torch.tensor(data[0])
        torch_data = torch_data.reshape(instances, 1)
        for i in range(1, len(data)):
            torch_data = torch.hstack((torch_data, torch.tensor(data[i]).reshape(instances, 1)))
        return torch_data

    def balance_scale_data(self):
        f = open('Data/balance-scale.data', 'r')
        data = [[], [], [], [], []]  # left_distance, left_weight, right_distance, right_weight, class
        vis_data = [[], [], []]
        for line in f:
            line = line.split(',')
            line[-1] = line[-1][:-1]  # remove \n
            if line[0] == 'L':
                data[-1].append(0)
            elif line[0] == 'B':
                data[-1].append(1)
            else:
                data[-1].append(2)
            for i in range(1, len(line)):
                data[i - 1].append(int(line[i]))
        for i in range(len(data[0])):
            vis_data[0].append(data[0][i] * data[1][i])
            vis_data[1].append(data[2][i] * data[3][i])
            vis_data[2].append(data[4][i])
        torch_data = self.transform_matrix_to_torch_tensor(vis_data)
        lengths = [int(len(torch_data) * 0.8), int(len(torch_data) * 0.2)]
        torch_data = torch.utils.data.random_split(torch_data, lengths)
        vis_torch_data = self.transform_matrix_to_torch_tensor(vis_data)
        return torch_data, vis_torch_data

    def populate_cluster(self, mean, i, final_data, k):
        dist = MultivariateNormal(torch.tensor(mean[i]), torch.eye(2))
        a = dist.sample([1, self.size])[0]
        a = torch.hstack((a, torch.full([self.size, 1], k)))
        final_data = torch.cat((final_data, a), 0)
        return final_data

    def generate_spiral_data(self):
        np.random.seed(0)
        N = self.size  # number of points per class
        K = 2  # number of classes
        X = np.zeros((N * K, 2))
        y = np.zeros(N * K, dtype='uint8')
        for j in range(K):
            ix = range(N * j, N * (j + 1))
            r = np.linspace(0.0, 1, N)  # radius
            t = np.linspace(j * 4, (j + 2) * 4, N) + np.random.randn(N) * 0.2  # theta
            X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            y[ix] = j
        final_data = torch.hstack((torch.tensor(X), torch.tensor(y).reshape(len(y), 1)))
        return final_data

    def generate_data(self):
        # cluster_means_a = [[0.0, 0.0], [0.0, 15.0], [15.0, -15.0], [-15.0, -15.0]]
        # cluster_means_b = [[0.0, 7.5], [7.5, -7.5], [-7.5, -7.5]]
        cluster_means_a = [[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0]]
        cluster_means_b = [[7.5, 7.5], [-7.5, -7.5], [-7.5, 7.5], [7.5, -7.5],
                           [-7.5, 0], [0, -7.5], [7.5, 0], [0, 7.5]]
        for i in range(len(cluster_means_a)):
            cluster_means_a[i][0] += 0
            cluster_means_a[i][1] += 0
        for i in range(len(cluster_means_b)):
            cluster_means_b[i][0] += 0
            cluster_means_b[i][1] += 0
        dist = MultivariateNormal(torch.tensor([0.0, 0.0]), torch.eye(2))
        final_data = torch.tensor(dist.sample([1, self.size]))[0]
        final_data = torch.hstack((final_data, torch.zeros(self.size, 1)))
        for i in range(len(cluster_means_a)):
            final_data = self.populate_cluster(cluster_means_a, i, final_data, 0)
        for i in range(len(cluster_means_b)):
            final_data = self.populate_cluster(cluster_means_b, i, final_data, 1)
        print(final_data.shape)
        return final_data

    @staticmethod
    def plot(data, hyperrectangle, ax=None):
        color = ['b', 'm', 'c', 'r', 'g', 'y', 'k', 'w', 'darkgrey', 'brown']
        if not ax:
            ax = plt.figure('Dataset', figsize=(7, 7)).add_subplot()
        ax.set_xlim(hyperrectangle.x)
        ax.set_ylim(hyperrectangle.y)
        ax.set_xlabel(r'$x_1$', labelpad=2)
        ax.set_ylabel(r'$x_2$', labelpad=2)
        for i in range(len(data)):
            # ax.scatter(data[i][0], data[i][1], s=5, c=color[int(data[i][2])])
            for j in range(len(data[i])):
                ax.scatter(data[i][j][0], data[i][j][1], s=25, marker=['o', 'x'][i],
                           c=color[int(data[i][j][2])])
