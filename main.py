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
    dropout_rates = [0, 0.8]
    # You probably don't want to change those
    train = False
    dataset = None  # set to 'balance_set' to work with the balance scale UCI dataset
    # You definitely don't want to change those
    global_point_bank = {}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # Prepare dataset and hyperrectangle
    my_data = (Dataset2D(class_size=3*data_size).data.to(dev), Dataset2D(class_size=data_size).data.to(dev))
    hyperrectangle = Hyperrectangle(my_data)
    linear_regions = []
    accuracies = [[], []]
    for _ in range(len(dropout_rates)):
        linear_regions.append([[], []])
    for _ in tqdm(range(2), desc="Processing"):
        # For fairness make sure both networks have the same initialisation
        torch.manual_seed(random.randint(1, int(math.pow(2, 32)) - 1))
        my_model1 = TrainedNeuralNetwork(ReLUNeuralNetwork(layers_width, initialization='xavier'), my_data,
                                         number_of_parameters, epochs=number_of_epochs, wd=1e-4, lr=1e-3, opt='ADAM',
                                         mode=dataset)
        trained_model1, accuracy = my_model1.main()
        accuracies[0].append(accuracy)
        my_model2 = TrainedNeuralNetwork(DropoutReLUNeuralNetwork(layers_width, initialization='xavier', dropout_prob=0.2),
                                         my_data, number_of_parameters, epochs=number_of_epochs, wd=1e-4, lr=1e-3,
                                         opt='ADAM', mode=dataset)
        trained_model2, accuracy = my_model2.main()
        accuracies[1].append(accuracy)
        # Run SkelEx on the two networks
        skeletons1 = []
        skeletons2 = []
        percentages = []
        for hidden_layer_index in range(len(layers_width) - 2):
            hidden_layer_percentages = []
            for i in range(layers_width[hidden_layer_index + 1]):
                hidden_layer_percentages.append(random.random())
            percentages.append(hidden_layer_percentages)
        for dropout_rate_index in range(len(dropout_rates)):
            dropout_rate = dropout_rates[dropout_rate_index]
            t0 = time()
            skelex1 = SkelEx(trained_model1.parameters(), global_point_bank, hyperrectangle, dropout=dropout_rate,)
            try:
                skeletons1.append(skelex1.main(percentages))
                linear_regions[dropout_rate_index][0].append(len(skeletons1[-1][0].linear_regions))
                # print("SkelEx finished within " + str(time()-t0) + " seconds when dropout was not used.")
            except:
                print('Lost one measurement due to error.')
            t0 = time()
            skelex2 = SkelEx(trained_model2.parameters(), global_point_bank, hyperrectangle, dropout=dropout_rate)
            try:
                skeletons2.append(skelex2.main(percentages))
                linear_regions[dropout_rate_index][1].append(len(skeletons2[-1][0].linear_regions))
                # print("SkelEx finished within " + str(time() - t0) + " seconds when dropout was used.")
            except:
                print('Lost one measurement due to error.')
    """# Run BoundEx
    fig, axs = plt.subplots(4, len(dropout_rates), figsize=(24, 8))
    for i in range(len(skeletons1)):
        boundary_extractor1 = BoundEx(skeletons1[i], hyperrectangle)
        boundary_extractor2 = BoundEx(skeletons2[i], hyperrectangle)
        classification_polygons1, lines_used1 = \
            boundary_extractor1.extract_decision_boundary_from_skeletons_of_decision_functions()
        classification_polygons2, lines_used2 = \
            boundary_extractor2.extract_decision_boundary_from_skeletons_of_decision_functions()
        print(f'The decision boundary when dropout is not used is created via {len(lines_used1)} line segments.')
        print(f'The decision boundary when dropout is used is created via {len(lines_used2)} line segments.')
        # Visualize results
        for r in range(4):
            axs[r, i].set_aspect(1)
            axs[r, i].set_xlim(hyperrectangle.x)
            axs[r, i].set_ylim(hyperrectangle.y)
            axs[r, i].set_xlabel(r'$x_1$', labelpad=2)
            axs[r, i].set_ylabel(r'$x_2$', labelpad=2)
        visualizer1 = Visualizer(skeletons1[i], trained_model1, hyperrectangle, number_of_classes)
        visualizer2 = Visualizer(skeletons2[i], trained_model2, hyperrectangle, number_of_classes)
        visualizer1.plot_skeleton(None, axs[0, i])
        visualizer1.plot_decision_boundary(boundary_extractor1, classification_polygons1, lines_used1, axs[1, i],
                                           save=True)
        visualizer2.plot_skeleton(None, axs[2, i])
        visualizer2.plot_decision_boundary(boundary_extractor2, classification_polygons2, lines_used2, axs[3, i],
                                           save=True)"""
    # Initialize empty lists to store means and standard deviations
    means_function1 = []
    std_dev_function1 = []
    means_function2 = []
    std_dev_function2 = []
    mean_accuracies_without_dropout = np.mean(accuracies[0])
    std_accuracies_without_dropout = np.std(accuracies[0])
    mean_accuracies_with_dropout = np.mean(accuracies[1])
    std_accuracies_with_dropout = np.std(accuracies[1])
    print('The average accuracy of networks without dropout was', mean_accuracies_without_dropout, u"\u00B1",
          std_accuracies_without_dropout, '. Whereas for networks with dropout this was', mean_accuracies_with_dropout,
          u"\u00B1", std_accuracies_with_dropout)
    # Calculate means and standard deviations for each sublist
    for sublist in linear_regions:
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
    plt.xlabel('Dropout')
    plt.ylabel('Number of Linear Regions')
    plt.legend()
    # Show the plot
    plt.grid(True)
    plt.show()
    print('Done!')
