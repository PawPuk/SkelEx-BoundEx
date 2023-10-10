from torch import nn


class ReLUNeuralNetwork(nn.Module):
    def __init__(self, layers_width, initialization=None):
        """

        @param layers_width: list with each entry specifying the width of the given layer
        @param initialization: set 'xavier' for Xavier uniform or leave None for default initialization
        """
        super(ReLUNeuralNetwork, self).__init__()
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


class DropoutReLUNeuralNetwork(nn.Module):
    def __init__(self, layers_width, dropout_prob=0.5, initialization=None):
        """
        @param layers_width: list with each entry specifying the width of the given layer
        @param dropout_prob: probability of dropout (default: 0.5)
        @param initialization: set 'xavier' for Xavier uniform or leave None for default initialization
        """
        super(DropoutReLUNeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        layers = []
        for index in range(len(layers_width)-1):
            layer = nn.Linear(layers_width[index], layers_width[index+1])
            if initialization == 'xavier':
                nn.init.xavier_uniform_(layer.weight)
            layers.append(layer)
            layers.append(nn.ReLU())
            # Add dropout layer after each hidden layer
            if index < len(layers_width) - 2:  # Exclude the output layer
                layers.append(nn.Dropout(p=dropout_prob))
        # Remove ReLU from the output layer
        layers.pop()
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x.float())
        logits = self.linear_relu_stack(x)
        return logits
