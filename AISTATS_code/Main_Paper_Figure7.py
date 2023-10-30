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
    layers_width = [number_of_parameters, 12, 10, number_of_classes]
    data_size = 1000
    number_of_epochs = 250
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
    # For fairness make sure both networks have the same initialisation
    torch.manual_seed(random.randint(1, int(math.pow(2, 32)) - 1))
    my_model2 = TrainedNeuralNetwork(DropoutReLUNeuralNetwork(layers_width, initialization='xavier', dropout_prob=0.2),
                                     my_data, number_of_parameters, epochs=number_of_epochs, wd=1e-4, lr=1e-3,
                                     opt='ADAM', mode=dataset)
    trained_model2 = my_model2.main()
    my_model1 = TrainedNeuralNetwork(ReLUNeuralNetwork(layers_width, initialization='xavier'), my_data,
                                     number_of_parameters, epochs=number_of_epochs, wd=1e-4, lr=1e-3, opt='ADAM',
                                     mode=dataset)
    trained_model1 = my_model1.main()
    # Run SkelEx on the two networks
    percentages, skeletons1, skeletons2 = [], [], []
    for hidden_layer_index in range(len(layers_width) - 2):
        number_of_neurons = layers_width[hidden_layer_index+1]
        hidden_layer_percentages = [(x+1) / number_of_neurons for x in range(number_of_neurons)]
        random.shuffle(hidden_layer_percentages)
        percentages.append(hidden_layer_percentages)
    print(percentages)
    for dropout_rate_index in range(len(dropout_rates)):
        dropout_rate = dropout_rates[dropout_rate_index]
        t0 = time()
        skelex1 = SkelEx(trained_model1.parameters(), global_point_bank, hyperrectangle, dropout=dropout_rate)
        try:
            skeletons1.append(skelex1.main(percentages))
            print("SkelEx finished within " + str(time()-t0) + " seconds when dropout was not used.")
        except:
            skeletons1.append(None)
            print('Lost one measurement due to error.')
        t0 = time()
        skelex2 = SkelEx(trained_model2.parameters(), global_point_bank, hyperrectangle, dropout=dropout_rate)
        try:
            skeletons2.append(skelex2.main(percentages))
            print("SkelEx finished within " + str(time() - t0) + " seconds when dropout was used.")
        except:
            skeletons2.append(None)
            print('Lost one measurement due to error.')
    # Run BoundEx
    fig, axs = plt.subplots(4, len(dropout_rates), figsize=(24, 8))
    for i in range(len(skeletons1)):
        # Visualize results
        for r in range(4):
            axs[r, i].set_aspect(1)
            axs[r, i].set_xlim(hyperrectangle.x)
            axs[r, i].set_ylim(hyperrectangle.y)
            if r == 3:
                axs[r, i].set_xlabel(r'$x_1$', labelpad=2)
            if i == 0:
                axs[r, i].set_ylabel(r'$x_2$', labelpad=2)
        visualizer1 = Visualizer(skeletons1[i], trained_model1, hyperrectangle, number_of_classes)
        visualizer2 = Visualizer(skeletons2[i], trained_model2, hyperrectangle, number_of_classes)
        if skeletons1[i] is not None:
            try:
                boundary_extractor1 = BoundEx(skeletons1[i], hyperrectangle)
                classification_polygons1, lines_used1 = \
                    boundary_extractor1.extract_decision_boundary_from_skeletons_of_decision_functions()
                print(f'The decision boundary when dropout is not used is created via {len(lines_used1)} line segments.')
            except:
                pass
        if skeletons2[i] is not None:
            try:
                boundary_extractor2 = BoundEx(skeletons2[i], hyperrectangle)
                classification_polygons2, lines_used2 = \
                    boundary_extractor2.extract_decision_boundary_from_skeletons_of_decision_functions()
                print(f'The decision boundary when dropout is used is created via {len(lines_used2)} line segments.')
            except:
                pass
        try:
            visualizer1.plot_skeleton(None, axs[0, i])
            visualizer1.plot_decision_boundary(boundary_extractor1, classification_polygons1, lines_used1, axs[0, i],
                                               save=True)
        except:
            pass
        try:
            visualizer1.plot_decision_boundary(boundary_extractor1, classification_polygons1, lines_used1, axs[1, i],
                                               save=True)
        except:
            pass
        try:
            visualizer2.plot_skeleton(None, axs[2, i])
            visualizer2.plot_decision_boundary(boundary_extractor2, classification_polygons2, lines_used2, axs[2, i],
                                               save=True)
        except:
            pass
        try:
            visualizer2.plot_decision_boundary(boundary_extractor2, classification_polygons2, lines_used2, axs[3, i],
                                               save=True)
        except:
            pass
    plt.show()