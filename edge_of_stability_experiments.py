import random
import statistics

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import Dataset, DataLoader
from scipy.sparse.linalg import LinearOperator, eigsh
import tqdm
from statistics import mean, stdev
import torch.nn.functional as F

from skelex import SkelEx
from hyperrectangle import Hyperrectangle
from visualisation import Visualizer
from boundex import BoundEx


class SpiralDataset(Dataset):
    def __init__(self, num_samples, num_classes):
        np.random.seed(0)
        n = num_samples // num_classes
        X = np.zeros((n * num_classes, 2))
        y = np.zeros(n * num_classes, dtype='int64')
        for j in range(num_classes):
            ix = range(n * j, n * (j + 1))
            r = np.linspace(0.0, 1, n)
            t = np.linspace(j * 4, (j + 2.5) * 4, n) + np.random.randn(n) * 0.2
            X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            y[ix] = j
        self.data = torch.Tensor(X)
        self.targets = torch.LongTensor(y)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def plot_decision_boundary(ax, X, y, model, title):
    data_colors = {0: 'red', 1: 'orange', 2: 'gold', 3: 'blue'}
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    X_mesh = np.c_[xx.ravel(), yy.ravel()]
    X_mesh_tensor = torch.Tensor(X_mesh)

    with torch.no_grad():
        Z = model(X_mesh_tensor).numpy()

    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)

    unique_labels = np.unique(y)
    for label in unique_labels:
        ax.scatter(X[y == label, 0], X[y == label, 1], c=data_colors[label], label=f'Class {label}')

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(title)
    ax.legend()


class SpiralNet(nn.Module):
    def __init__(self, layers_width, initialization=None):
        """

        @param layers_width: list with each entry specifying the width of the given layer
        @param initialization: set 'xavier' for Xavier uniform or leave None for default initialization
        """
        super(SpiralNet, self).__init__()
        self.flatten = nn.Flatten()
        layers = []
        for index in range(len(layers_width)-1):
            layer = nn.Linear(layers_width[index], layers_width[index+1])
            if initialization == 'xavier':
                nn.init.xavier_uniform_(layer.weight)
            layers.append(layer)
            layers.append(nn.ReLU())
        # Remove ReLU from the output layer
        layers.pop()
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x.float())
        logits = self.linear_relu_stack(x)
        return logits


def iterate_dataset(dataset: Dataset, batch_size: int):
    """Iterate through a dataset, yielding batches of data."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for (batch_X, batch_y) in loader:
        yield batch_X, batch_y


def compute_hvp(network: nn.Module, loss_fn: nn.Module,
                dataset: Dataset, vector: Tensor, physical_batch_size=1000):
    """Compute a Hessian-vector product."""
    p = len(parameters_to_vector(network.parameters()))
    n = len(dataset)
    hvp = torch.zeros(p, dtype=torch.float)
    vector = vector
    for (X, y) in iterate_dataset(dataset, physical_batch_size):
        loss = loss_fn(network(X), y) / n
        grads = torch.autograd.grad(loss, inputs=network.parameters(), create_graph=True)
        dot = parameters_to_vector(grads).mul(vector).sum()
        grads = [g.contiguous() for g in torch.autograd.grad(dot, network.parameters(), retain_graph=True)]
        hvp += parameters_to_vector(grads)
    return hvp


def lanczos(matrix_vector, dim: int, neigs: int):
    """ Invoke the Lanczos algorithm to compute the leading eigenvalues and eigenvectors of a matrix / linear operator
    (which we can access via matrix-vector products). """

    def mv(vec: np.ndarray):
        gpu_vec = torch.tensor(vec, dtype=torch.float)
        return matrix_vector(gpu_vec)

    operator = LinearOperator((dim, dim), matvec=mv)
    evals, evecs = eigsh(operator, neigs)
    return torch.from_numpy(np.ascontiguousarray(evals[::-1]).copy()).float(), \
           torch.from_numpy(np.ascontiguousarray(np.flip(evecs, -1)).copy()).float()


num_samples = 1000
num_classes = 4
num_epochs = 5000
dataset = SpiralDataset(num_samples, num_classes)
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.targets, test_size=0.2, random_state=42)
eig_freq = 10
lr = 0.005
neigs = 6
eigs = torch.zeros(num_epochs // eig_freq if eig_freq >= 0 else 0, neigs)
criterion = nn.CrossEntropyLoss()
X_train_tensor = torch.Tensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
maxs, calm_maxs, sharpness_store, accuracies, gradient_store = [], [], [], [], []
layers_width = [2, 10, 10, 10, 10, num_classes]
model = SpiralNet(layers_width)
optimizer = optim.Adam(model.parameters(), lr=lr)
evec = None
for epoch in tqdm.tqdm(range(num_epochs), desc='Epoch'):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    hvp_delta = lambda delta: compute_hvp(model, criterion, dataset, delta).detach().cpu()
    grad = torch.cat([param.grad.view(-1) for param in model.parameters() if param.grad is not None])
    gradient_store.append(grad.norm().item())
    nparams = len(parameters_to_vector((model.parameters())))
    evals, evecs = lanczos(hvp_delta, nparams, neigs=neigs)
    evec = evecs[:, 0]
    eigs[epoch // eig_freq, :] = evals
    sharpness_store.append(eigs[epoch // eig_freq, 0].item())
    optimizer.step()
    with torch.no_grad():
        X_test_tensor = torch.Tensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test_tensor).sum().item() / y_test.shape[0]
        accuracies.append(accuracy)
        if epoch > 50:
            condition = True
            for i in range(-1, -5, -1):
                if accuracies[i] != accuracy:
                    condition = False
            if condition and accuracy > 85:
                break
    maxs.append(max(sharpness_store))
    calm_sharpness_store = [statistics.median(sharpness_store[i:i+50]) for i in range(0, len(sharpness_store), 50)]
    calm_maxs.append(max(calm_sharpness_store))
    with torch.no_grad():
        X_test_tensor = torch.Tensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test_tensor).sum().item() / y_test.shape[0]

percentages = []
for hidden_layer_index in range(len(layers_width) - 2):
    number_of_neurons = layers_width[hidden_layer_index+1]
    hidden_layer_percentages = [(x+1) / number_of_neurons for x in range(number_of_neurons)]
    random.shuffle(hidden_layer_percentages)
    percentages.append(hidden_layer_percentages)

hyperrectangle = Hyperrectangle((X_train, X_test))
skelex1 = SkelEx(model.parameters(), {}, hyperrectangle)


"""steps = 4
linear_regions = []
gradients = []
fig, axs = plt.subplots(2, steps, figsize=(24, 8))
for i in range(steps):
    skeletons1 = skelex1.main(percentages)
    linear_regions.append(len(skeletons1[0].linear_regions))
    for r in range(2):
        axs[r, i].set_aspect(1)
        axs[r, i].set_xlim(hyperrectangle.x)
        axs[r, i].set_ylim(hyperrectangle.y)
        if r == 3:
            axs[r, i].set_xlabel(r'$x_1$', labelpad=2)
        if i == 0:
            axs[r, i].set_ylabel(r'$x_2$', labelpad=2)
    boundary_extractor1 = BoundEx(skeletons1, hyperrectangle)
    classification_polygons1, lines_used1 = \
        boundary_extractor1.extract_decision_boundary_from_skeletons_of_decision_functions()
    visualizer1 = Visualizer(skeletons1, model, hyperrectangle, num_classes)
    visualizer1.plot_skeleton(None, axs[0, i])
    visualizer1.plot_decision_boundary(boundary_extractor1, classification_polygons1, lines_used1, axs[0, i],
                                       save=True)
    plot_decision_boundary(axs[1, i], X_train, y_train, model, "Training Data Decision Boundary")
    grads1 = torch.cat([param.grad.view(-1) for param in model.parameters() if param.grad is not None])
    vector_to_parameters(parameters_to_vector(model.parameters()) + evec * 5 * lr, model.parameters())
    skelex1 = SkelEx(model.parameters(), {}, hyperrectangle)
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    grads2 = torch.cat([param.grad.view(-1) for param in model.parameters() if param.grad is not None])
    for grad in [torch.norm(grads1), torch.norm(grads2)]:
        if grad not in gradients:
            gradients.append(grad)
print(linear_regions)
print(gradients)"""
sharpness1 = [statistics.median(sharpness_store[i:i+1]) for i in range(0, len(sharpness_store), 1)]
gradients1 = [statistics.median(gradient_store[i:i+1]) for i in range(0, len(gradient_store), 1)]
accuracies1 = [statistics.median(accuracies[i:i+1]) for i in range(0, len(accuracies), 1)]
sharpness2 = [statistics.mean(sharpness_store[i:i+50]) for i in range(0, len(sharpness_store), 50)]
gradients2 = [statistics.mean(gradient_store[i:i+50]) for i in range(0, len(gradient_store), 50)]
accuracies2 = [statistics.mean(accuracies[i:i+50]) for i in range(0, len(accuracies), 50)]
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.plot(sharpness1, label='Sharpness')
plt.plot(gradients1, label='Gradient norm')
plt.xlabel('epoch')
plt.ylabel('norm')
plt.legend()
plt.title('Figure 1')
plt.subplot(2, 2, 3)
plt.plot(accuracies1)
plt.title('Figure 2')
plt.subplot(2, 2, 2)
plt.plot(sharpness2, label='Sharpness')
plt.plot(gradients2, label='Gradient norm')
plt.xlabel('epoch')
plt.ylabel('norm')
plt.legend()
plt.title('Figure 3')
plt.subplot(2, 2, 4)
plt.plot(accuracies2)
plt.title('Figure 4')
plt.tight_layout()


fig1, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
plot_decision_boundary(ax3, X_train, y_train, model, "Training Data Decision Boundary")
plot_decision_boundary(ax4, X_test, y_test, model, "Test Data Decision Boundary")
plt.tight_layout()
plt.show()
