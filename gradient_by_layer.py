import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import tqdm
import statistics
import math


# Define a simple feedforward neural network with ReLU activations for CIFAR-10
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 98)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(98, 98)
        self.fc3 = nn.Linear(98, 98)
        self.fc5 = nn.Linear(98, 10)  # 10 output classes for CIFAR-10

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc5(x)
        return x


class SimpleDropoutNN(nn.Module):
    def __init__(self):
        super(SimpleDropoutNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 98)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Add dropout with 50% probability
        self.fc2 = nn.Linear(98, 98)
        self.fc3 = nn.Linear(98, 98)
        self.fc5 = nn.Linear(98, 10)  # 10 output classes for CIFAR-10

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc5(x)
        return x


def train_and_test(net, optimizer, scheduler, criterion):
    for epoch in tqdm.tqdm(range(200), desc='Epoch'):
        if epoch % 20 == 0:
            estimate_number_of_activation_regions(net)
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        # Extract and print gradients grouped by layer
        norm = 0
        i1 = 0
        for name, param in net.named_parameters():
            if param.grad is not None:
                if 'weight' in name:  # Combine gradients for weights and biases
                    norm += param.grad.norm() ** 2
                else:
                    norm += param.grad.norm() ** 2
                    norm = norm ** (1 / 2)
                    norms[i1].append(norm.item())
                    norm = 0
                    i1 += 1
        # Calculate accuracy on the test dataset
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        # Append loss and accuracy values to the lists
        loss_values.append(running_loss / (i + 1))
        accuracy_values.append(accuracy)


def estimate_number_of_activation_regions(network):
    activation_regions = {}
    network.to('cpu')

    def hook_fn(module, input, output):
        activations.append(output)
    # Register a hook for each hidden layer (modify as needed)
    for layer in network.children():
        if isinstance(layer, torch.nn.Linear):
            layer.register_forward_hook(hook_fn)
    for _ in tqdm.tqdm(range(int(math.pow(10, 7))), desc='Data samples'):
        data_sample = torch.randn(1, 28*28)
        # Extract intermediate activations from hidden layers
        activations = []
        # Forward pass to trigger the hooks
        output = network(data_sample)
        # Apply threshold to determine active neurons for each hidden layer
        threshold = 0
        hidden_activations = activations[:-1]  # Exclude the output layer
        key = tuple([act > threshold for act in hidden_activations][0].tolist()[0])
        activation_regions[key] = activation_regions.get(key, 0) + 1
    print(len(activation_regions.keys()))
    network.to('mps')
    print(len(123))
    return len(activation_regions.keys())


# Training loop with learning rate schedule
if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps")
    # Define data transformations and load the Fashion MNIST dataset
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform_train,
                                                      download=True)
    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform_test)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)
    norms = [[], [], [], [], []]
    loss_values = []
    accuracy_values = []
    # Initialize the network, optimizer (with learning rate schedule), and loss function
    net1 = SimpleDropoutNN().to(device)
    optimizer1 = optim.SGD(net1.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler1 = optim.lr_scheduler.MultiStepLR(optimizer1, milestones=[50], gamma=0.1)
    criterion1 = nn.CrossEntropyLoss()
    train_and_test(net1, optimizer1, scheduler1, criterion1)
    """# Transform the learned parameters to the network with dropout
    net2 = SimpleDropoutNN().to(device)
    net2.load_state_dict(net1.state_dict())
    optimizer2 = optim.SGD(net2.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer2, milestones=[200], gamma=0.1)
    criterion2 = nn.CrossEntropyLoss()
    train_and_test(net2, optimizer2, scheduler2, criterion2)"""
    print("Finished Training")
    s1, s2 = 5, 10
    loss_values1 = [statistics.mean(loss_values[i:i + s1]) for i in range(0, len(loss_values), s1)]
    loss_values2 = [statistics.mean(loss_values[i:i + s2]) for i in range(0, len(loss_values), s2)]
    accuracy_values1 = [statistics.mean(accuracy_values[i:i + s1]) for i in range(0, len(accuracy_values), s1)]
    accuracy_values2 = [statistics.mean(accuracy_values[i:i + s2]) for i in range(0, len(accuracy_values), s2)]
    norms1, norms2 = [], []
    for j in range(len(norms)):
        norms1.append([statistics.mean(norms[j][i:i + s1]) for i in range(0, len(norms[j]), s1)])
        norms2.append([statistics.mean(norms[j][i:i + s2]) for i in range(0, len(norms[j]), s2)])

    # Labels for plotting
    labels = ['Parameters of layer 1', 'Parameters of layer 2', 'Parameters of layer 3', 'Parameters of layer 4',
              'Parameters of output layer']
    # Create a figure and axis for norms plot
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    # Norms plot
    for i in range(len(norms1)):
        ax[0].plot(norms1[i], label=labels[i])
    ax[0].set_xlabel(r'Epoch ($\times$ {s1})')
    ax[0].set_ylabel('Gradient Norm')
    ax[0].set_title('Combined Plot of Norms')
    ax[0].legend()
    ax[0].grid()
    # Loss and accuracy plot
    ax[1].plot(loss_values1, label='Loss', color='b')
    ax[1].set_xlabel(r'Epoch ($\times$ {s1})')
    ax[1].set_ylabel('Loss', color='b')  # Set the y-axis label for loss
    ax2 = ax[1].twinx()
    ax2.plot(accuracy_values1, label='Accuracy', color='r')
    ax2.set_ylabel('Accuracy (%)', color='r')  # Set the y-axis label for accuracy
    ax[1].set_title('Loss and Accuracy Plot')
    ax[1].grid()
    # Show and save the combined figure
    plt.savefig('SM_Training_Results1_Continuation.svg')
    plt.savefig('SM_Training_Results1_Continuation.png')

    # Create a figure and axis for norms plot
    fig1, ax1 = plt.subplots(1, 2, figsize=(16, 6))
    # Norms plot
    for i in range(len(norms2)):
        ax1[0].plot(norms2[i], label=labels[i])
    ax1[0].set_xlabel(r'Epoch ($\times$ {s2})')
    ax1[0].set_ylabel('Gradient Norm')
    ax[0].set_title('Combined Plot of Norms')
    ax1[0].legend()
    ax1[0].grid()
    # Loss and accuracy plot
    ax1[1].plot(loss_values2, label='Loss', color='b')
    ax1[1].set_xlabel(r'Epoch ($\times$ {s2})')
    ax1[1].set_ylabel('Loss', color='b')  # Set the y-axis label for loss
    ax3 = ax1[1].twinx()
    ax3.plot(accuracy_values2, label='Accuracy', color='r')
    ax3.set_ylabel('Accuracy (%)', color='r')  # Set the y-axis label for accuracy
    ax1[1].set_title('Loss and Accuracy Plot')
    ax1[1].grid()
    # Show and save the combined figure
    plt.savefig('SM_Training_Results2_Continuation.svg')
    plt.savefig('SM_Training_Results2_Continuation.png')
    plt.show()
