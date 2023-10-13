import math
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from dataset import Dataset2D
from hyperrectangle import Hyperrectangle
from ReLU_NN import ReLUNeuralNetwork, DropoutReLUNeuralNetwork
from skelex import SkelEx
from train import TrainedNeuralNetwork


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
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # Prepare dataset and hyperrectangle
    my_data = (Dataset2D(class_size=3*data_size).data.to(dev), Dataset2D(class_size=data_size).data.to(dev))
    hyperrectangle = Hyperrectangle(my_data)
    linear_regions = []
    for _ in range(len(dropout_rates)):
        linear_regions.append([[], []])
    for _ in tqdm(range(50), desc="Processing"):
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
        # Run SkelEx on the two networks
        percentages = []
        for hidden_layer_index in range(len(layers_width) - 2):
            hidden_layer_percentages = []
            for i in range(layers_width[hidden_layer_index + 1]):
                hidden_layer_percentages.append(random.random())
            percentages.append(hidden_layer_percentages)
        for dropout_rate_index in range(len(dropout_rates)):
            global_point_bank = {}
            dropout_rate = dropout_rates[dropout_rate_index]
            skelex1 = SkelEx(trained_model1.parameters(), global_point_bank, hyperrectangle, dropout=dropout_rate)
            try:
                skeletons1 = skelex1.main(percentages)
                linear_regions[dropout_rate_index][0].append(len(skeletons1[0].linear_regions))
            except:
                print('Lost one measurement due to error.')
            global_point_bank = {}
            skelex2 = SkelEx(trained_model2.parameters(), global_point_bank, hyperrectangle, dropout=dropout_rate)
            try:
                skeletons2 = skelex2.main(percentages)
                linear_regions[dropout_rate_index][1].append(len(skeletons2[0].linear_regions))
            except:
                print('Lost one measurement due to error.')
    # Initialize empty lists to store means and standard deviations
    means_function1 = []
    std_dev_function1 = []
    means_function2 = []
    std_dev_function2 = []
    # Calculate means and standard deviations for each sublist
    for sublist in linear_regions:
        function1_values = sublist[0]
        function2_values = sublist[1]
        means_function1.append(np.mean(function1_values))
        std_dev_function1.append(np.std(function1_values))
        means_function2.append(np.mean(function2_values))
        std_dev_function2.append(np.std(function2_values))
    # Create a figure
    plt.figure(figsize=(10, 6))
    # Plot the means of function 1 with error bars representing the standard deviations
    plt.errorbar(dropout_rates, means_function1, yerr=std_dev_function1, label='Without Dropout', marker='o',
                 linestyle='-', capsize=5)
    # Plot the means of function 2 with error bars representing the standard deviations
    plt.errorbar(dropout_rates, means_function2, yerr=std_dev_function2, label='With Dropout', marker='s',
                 linestyle='--', capsize=5)
    # Add labels and legend
    plt.xlabel('Ratio of removed tessellations')
    plt.ylabel('Number of linear regions')
    plt.legend()
    # Show the plot
    plt.grid(True)
    plt.show()
    print('Done!')
