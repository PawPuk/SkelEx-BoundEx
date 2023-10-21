import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import tqdm
import statistics


# Define a simple feedforward neural network with ReLU activations for CIFAR-10
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 3072)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Add dropout with 50% probability
        self.fc2 = nn.Linear(3072, 3072)
        self.fc3 = nn.Linear(3072, 3072)
        self.fc5 = nn.Linear(3072, 10)  # 10 output classes for CIFAR-10

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)  # Flatten the input
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc5(x)
        return x


# Training loop with learning rate schedule
if __name__ == '__main__':
    # Define data transformations and load the CIFAR-10 dataset with data augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform_test)
    # Create data loaders with batch normalization
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)
    # Initialize the network, optimizer (with learning rate schedule), and loss function
    net = SimpleNN()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 500], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    norms = [[], [], [], []]
    # Lists to store loss and accuracy values
    loss_values = []
    accuracy_values = []
    # Training loop
    for epoch in tqdm.tqdm(range(600), desc='Epoch'):
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
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
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        # Append loss and accuracy values to the lists
        loss_values.append(running_loss / (i + 1))
        accuracy_values.append(accuracy)
    print("Finished Training")
    loss_values1 = [statistics.mean(loss_values[i:i + 5]) for i in range(0, len(loss_values), 5)]
    loss_values2 = [statistics.mean(loss_values[i:i + 25]) for i in range(0, len(loss_values), 25)]
    accuracy_values1 = [statistics.mean(accuracy_values[i:i + 5]) for i in range(0, len(accuracy_values), 5)]
    accuracy_values2 = [statistics.mean(accuracy_values[i:i + 25]) for i in range(0, len(accuracy_values), 25)]
    norms1, norms2 = [], []
    for j in range(len(norms)):
        norms1.append([statistics.mean(norms[j][i:i + 5]) for i in range(0, len(norms[j]), 5)])
        norms2.append([statistics.mean(norms[j][i:i + 25]) for i in range(0, len(norms[j]), 25)])

    # Labels for plotting
    labels = ['Parameters of layer 1', 'Parameters of layer 2', 'Parameters of layer 3', 'Parameters of output layer']
    # Create a figure and axis for norms plot
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    # Norms plot
    for i in range(4):
        ax[0].plot(norms1[i], label=labels[i])
    ax[0].set_xlabel(r'Epoch ($\times$ 5)')
    ax[0].set_ylabel('Gradient Norm')
    ax[0].set_title('Combined Plot of Norms')
    ax[0].legend()
    ax[0].grid()
    # Loss and accuracy plot
    ax[1].plot(loss_values1, label='Loss', color='b')
    ax[1].set_xlabel(r'Epoch ($\times$ 5)')
    ax[1].set_ylabel('Loss', color='b')  # Set the y-axis label for loss
    ax2 = ax[1].twinx()
    ax2.plot(accuracy_values1, label='Accuracy', color='r')
    ax2.set_ylabel('Accuracy (%)', color='r')  # Set the y-axis label for accuracy
    ax[1].set_title('Loss and Accuracy Plot')
    ax[1].grid()
    # Show and save the combined figure
    plt.savefig('Training_Results3.svg')
    plt.savefig('Training_Results3.png')

    # Create a figure and axis for norms plot
    fig1, ax1 = plt.subplots(1, 2, figsize=(16, 6))
    # Norms plot
    for i in range(4):
        ax1[0].plot(norms2[i], label=labels[i])
    ax1[0].set_xlabel(r'Epoch ($\times$ 25)')
    ax1[0].set_ylabel('Gradient Norm')
    ax[0].set_title('Combined Plot of Norms')
    ax1[0].legend()
    ax1[0].grid()
    # Loss and accuracy plot
    ax1[1].plot(loss_values2, label='Loss', color='b')
    ax1[1].set_xlabel(r'Epoch ($\times$ 25)')
    ax1[1].set_ylabel('Loss', color='b')  # Set the y-axis label for loss
    ax3 = ax1[1].twinx()
    ax3.plot(accuracy_values2, label='Accuracy', color='r')
    ax3.set_ylabel('Accuracy (%)', color='r')  # Set the y-axis label for accuracy
    ax1[1].set_title('Loss and Accuracy Plot')
    ax1[1].grid()
    # Show and save the combined figure
    plt.savefig('Training_Results4.svg')
    plt.savefig('Training_Results4.png')
    plt.show()
