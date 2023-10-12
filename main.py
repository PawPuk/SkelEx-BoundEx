import math
import random
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from boundex import BoundEx
from dataset import Dataset2D
from hyperrectangle import Hyperrectangle
from ReLU_NN import ReLUNeuralNetwork, DropoutReLUNeuralNetwork
from skelex import SkelEx
from train import TrainedNeuralNetwork
from visualisation import Visualizer


if __name__ == '__main__':
    # You probably want to change those
    number_of_classes = 2
    number_of_parameters = 2
    layers_width = [number_of_parameters, 20, 10, number_of_classes]
    data_size = 1000
    number_of_epochs = 150
    dropout_rates = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85]
    # You probably don't want to change those
    train = False
    dataset = None  # set to 'balance_set' to work with the balance scale UCI dataset
    # You definitely don't want to change those
    global_point_bank = {}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # Prepare dataset and hyperrectangle
    my_data = (Dataset2D(class_size=3*data_size).data.to(dev), Dataset2D(class_size=data_size).data.to(dev))
    hyperrectangle = Hyperrectangle(my_data)
    accuracies = []
    for _ in range(len(dropout_rates)):
        accuracies.append([[], []])
    neuron_probabilities = []
    for i in range(1, len(layers_width) - 1):
        layer_probabilities = []
        for _ in range(layers_width[i]):
            layer_probabilities.append(random.random())
        neuron_probabilities.append(layer_probabilities)
    for _ in tqdm(range(200), desc="Processing"):
        # For fairness make sure both networks have the same initialisation
        torch.manual_seed(random.randint(1, int(math.pow(2, 32)) - 1))
        my_model1 = TrainedNeuralNetwork(ReLUNeuralNetwork(layers_width, initialization='xavier'), my_data,
                                         number_of_parameters, epochs=number_of_epochs, wd=1e-4, lr=1e-3, opt='ADAM',
                                         mode=dataset)
        trained_model1 = my_model1.main()
        my_model2 = TrainedNeuralNetwork(DropoutReLUNeuralNetwork(layers_width, initialization='xavier',
                                                                  dropout_prob=0.2),
                                         my_data, number_of_parameters, epochs=number_of_epochs, wd=1e-4, lr=1e-3,
                                         opt='ADAM', mode=dataset)
        trained_model2 = my_model2.main()
        parameters1, parameters2 = [], []
        for key in trained_model1.state_dict():
            parameters1.append(trained_model1.state_dict()[key])
        for key in trained_model2.state_dict():
            parameters2.append(trained_model2.state_dict()[key])
        for dropout_rate_index in range(len(dropout_rates)):
            dropout_rate = dropout_rates[dropout_rate_index]
            # We set random (depending on dropout_rate) number of neurons to 0 in each layer
            for layer_i in range(0, len(layers_width) - 2):
                for neuron_i in range(len(neuron_probabilities[layer_i])):
                    if neuron_probabilities[layer_i][neuron_i] < dropout_rate:
                        # We change neuron_i from layer_i to an identity map which is equivalent to removing it
                        parameters1[2*layer_i][neuron_i] = 0.0
                        parameters1[2*layer_i+1][neuron_i] = 0.0
                        parameters2[2*layer_i][neuron_i] = 0.0
                        parameters2[2*layer_i+1][neuron_i] = 0.0
            # Now we update the parameters of the network accordingly
            modified_state_dict1 = {}
            modified_state_dict2 = {}
            for i, key in enumerate(trained_model1.state_dict()):
                modified_state_dict1[key] = parameters1[i]
            for i, key in enumerate(trained_model2.state_dict()):
                modified_state_dict2[key] = parameters2[i]
            trained_model1.load_state_dict(modified_state_dict1)
            trained_model2.load_state_dict(modified_state_dict2)
            accuracies[dropout_rate_index][0].append(TrainedNeuralNetwork(trained_model1, my_data, number_of_parameters,
                                                                          epochs=0).main())
            accuracies[dropout_rate_index][1].append(TrainedNeuralNetwork(trained_model2, my_data, number_of_parameters,
                                                                          epochs=0).main())
    # Initialize empty lists to store means and standard deviations
    means_function1 = []
    std_dev_function1 = []
    means_function2 = []
    std_dev_function2 = []
    # Calculate means and standard deviations for each sublist
    for sublist in accuracies:
        function1_values = sublist[0]
        function2_values = sublist[1]
        mean_function1 = np.mean(function1_values)
        std_deviation_function1 = np.std(function1_values)
        mean_function2 = np.mean(function2_values)
        std_deviation_function2 = np.std(function2_values)
        means_function1.append(mean_function1)
        std_dev_function1.append(std_deviation_function1)
        means_function2.append(mean_function2)
        std_dev_function2.append(std_deviation_function2)
    # Create a figure
    plt.figure(figsize=(10, 6))
    # Plot the means of function 1 with error bars representing the standard deviations
    plt.errorbar(dropout_rates, means_function1, yerr=std_dev_function1, label='Function 1', marker='o', linestyle='-',
                 capsize=5)
    # Plot the means of function 2 with error bars representing the standard deviations
    plt.errorbar(dropout_rates, means_function2, yerr=std_dev_function2, label='Function 2', marker='s', linestyle='--',
                 capsize=5)
    # Add labels and legend
    plt.xlabel('Dropout rate')
    plt.ylabel('Accuracy')
    plt.legend()
    # Show the plot
    plt.grid(True)
    plt.show()
    print('Done!')
