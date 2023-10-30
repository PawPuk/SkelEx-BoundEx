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
    layers_width = [number_of_parameters, 12, 10, number_of_classes]
    data_size = 1000
    number_of_epochs = 150
    small_epochs = 10
    # You probably don't want to change those
    train = False
    dataset = None  # set to 'balance_set' to work with the balance scale UCI dataset
    # You definitely don't want to change those
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # Prepare dataset and hyperrectangle
    my_data = (Dataset2D(class_size=3*data_size).data.to(dev), Dataset2D(class_size=data_size).data.to(dev))
    hyperrectangle = Hyperrectangle(my_data)
    linear_regions = []
    for _ in range(int(number_of_epochs / small_epochs) + 1):
        linear_regions.append([[], []])
    percentages = []
    for hidden_layer_index in range(len(layers_width) - 2):
        hidden_layer_percentages = []
        for i in range(layers_width[hidden_layer_index + 1]):
            hidden_layer_percentages.append(1.0)
        percentages.append(hidden_layer_percentages)
    for _ in tqdm(range(50), desc="Runs"):
        network1 = ReLUNeuralNetwork(layers_width, initialization='xavier')
        network2 = DropoutReLUNeuralNetwork(layers_width, initialization='xavier')
        skelex1 = SkelEx(network1.parameters(), {}, hyperrectangle, dropout=0)
        skelex2 = SkelEx(network2.parameters(), {}, hyperrectangle, dropout=0)
        try:
            skeletons1 = skelex1.main(percentages)
            linear_regions[0][0].append(len(skeletons1[0].linear_regions))
        except:
            print('Lost one measurement due to error.')
        try:
            skeletons2 = skelex2.main(percentages)
            linear_regions[0][1].append(len(skeletons2[0].linear_regions))
        except:
            print('Lost one measurement due to error.')
        for index in range(int(number_of_epochs / small_epochs)):
            # For fairness make sure both networks have the same initialisation
            torch.manual_seed(random.randint(1, int(math.pow(2, 32)) - 1))
            my_model1 = TrainedNeuralNetwork(network1, my_data, number_of_parameters, epochs=small_epochs, wd=1e-4,
                                             lr=1e-3, opt='ADAM', mode=dataset)
            network1 = my_model1.main()
            my_model2 = TrainedNeuralNetwork(network2, my_data, number_of_parameters, epochs=small_epochs, wd=1e-4,
                                             lr=1e-3, opt='ADAM', mode=dataset)
            network2 = my_model2.main()
            # Run SkelEx on the two networks
            skelex1 = SkelEx(network1.parameters(), {}, hyperrectangle, dropout=0)
            skelex2 = SkelEx(network2.parameters(), {}, hyperrectangle, dropout=0)
            try:
                skeletons1 = skelex1.main(percentages)
                linear_regions[index + 1][0].append(len(skeletons1[0].linear_regions))
            except:
                print('Lost one measurement due to error.')
            try:
                skeletons2 = skelex2.main(percentages)
                linear_regions[index + 1][1].append(len(skeletons2[0].linear_regions))
            except:
                print('Lost one measurement due to error.')

    print(linear_regions)
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
    plt.errorbar(np.arange(0, number_of_epochs + small_epochs, small_epochs), means_function1, yerr=std_dev_function1,
                 label='Without Dropout', marker='o', linestyle='-', capsize=5)
    plt.errorbar(np.arange(0, number_of_epochs + small_epochs, small_epochs), means_function2, yerr=std_dev_function2,
                 label='With Dropout', marker='o', linestyle='-', capsize=5)
    # Add labels and legend
    plt.xlabel('Epoch')
    plt.ylabel('Number of activation regions')
    plt.legend()
    # Show the plot
    plt.grid(True)
    plt.savefig('Experiment1.pdf')
    plt.savefig('Experiment1.png')
    plt.savefig('Experiment1.svg')
    plt.show()
    print('Done!')
