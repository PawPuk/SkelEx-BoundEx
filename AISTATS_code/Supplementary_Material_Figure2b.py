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
    number_of_epochs = 100
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
    loss_values1, loss_values2 = [], []
    accuracy_values1, accuracy_values2 = [], []
    for _ in range(2 * int(number_of_epochs / small_epochs) + 1):
        linear_regions.append([[], []])
    for _ in range(2 * int(number_of_epochs / small_epochs)):
        loss_values1.append([])
        loss_values2.append([])
        accuracy_values1.append([])
        accuracy_values2.append([])
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
            network1, loss1, accuracy1 = my_model1.main()
            my_model2 = TrainedNeuralNetwork(network2, my_data, number_of_parameters, epochs=small_epochs, wd=1e-4,
                                             lr=1e-3, opt='ADAM', mode=dataset)
            network2, loss2, accuracy2 = my_model2.main()
            accuracy_values1[index].append(accuracy1)
            accuracy_values2[index].append(accuracy2)
            loss_values1[index].append(loss1)
            loss_values2[index].append(loss2)
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
        state_dict1 = network1.state_dict()
        state_dict2 = network2.state_dict()
        key_mapping1 = {'linear_relu_stack.0.weight': 'linear_relu_stack.0.weight',
                        'linear_relu_stack.0.bias': 'linear_relu_stack.0.bias',
                        'linear_relu_stack.2.weight': 'linear_relu_stack.3.weight',
                        'linear_relu_stack.2.bias': 'linear_relu_stack.3.bias',
                        'linear_relu_stack.4.weight': 'linear_relu_stack.6.weight',
                        'linear_relu_stack.4.bias': 'linear_relu_stack.6.bias'}
        key_mapping2 = {'linear_relu_stack.0.weight': 'linear_relu_stack.0.weight',
                        'linear_relu_stack.0.bias': 'linear_relu_stack.0.bias',
                        'linear_relu_stack.3.weight': 'linear_relu_stack.2.weight',
                        'linear_relu_stack.3.bias': 'linear_relu_stack.2.bias',
                        'linear_relu_stack.6.weight': 'linear_relu_stack.4.weight',
                        'linear_relu_stack.6.bias': 'linear_relu_stack.4.bias'}
        state_dict1 = {key_mapping1[old_key]: value for old_key, value in state_dict1.items()}
        state_dict2 = {key_mapping2[old_key]: value for old_key, value in state_dict2.items()}
        network1 = DropoutReLUNeuralNetwork(layers_width, initialization='xavier')
        network2 = ReLUNeuralNetwork(layers_width, initialization='xavier')
        network1.load_state_dict(state_dict1)
        network2.load_state_dict(state_dict2)
        for index in range(int(number_of_epochs / small_epochs), 2*int(number_of_epochs / small_epochs)):
            # For fairness make sure both networks have the same initialisation
            torch.manual_seed(random.randint(1, int(math.pow(2, 32)) - 1))
            my_model1 = TrainedNeuralNetwork(network1, my_data, number_of_parameters, epochs=small_epochs, wd=1e-4,
                                             lr=1e-3, opt='ADAM', mode=dataset)
            network1, loss1, accuracy1 = my_model1.main()
            my_model2 = TrainedNeuralNetwork(network2, my_data, number_of_parameters, epochs=small_epochs, wd=1e-4,
                                             lr=1e-3, opt='ADAM', mode=dataset)
            network2, loss2, accuracy2 = my_model2.main()
            accuracy_values1[index].append(accuracy1)
            accuracy_values2[index].append(accuracy2)
            loss_values1[index].append(loss1)
            loss_values2[index].append(loss2)
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

    def extract_statistics(l1, l2, l3, i):
        sub_l1 = l1[i]
        l2.append(np.mean(sub_l1))
        l3.append(np.std(sub_l1))

    # Initialize empty lists to store means and standard deviations
    means_function1 = []
    std_dev_function1 = []
    means_function2 = []
    std_dev_function2 = []
    means_loss1, means_loss2, means_accuracy1, means_accuracy2 = [], [], [], []
    std_loss1, std_loss2, std_accuracy1, std_accuracy2 = [], [], [], []
    # Calculate means and standard deviations for each sublist
    for index in range(len(linear_regions)):
        sublist = linear_regions[index]
        function1_values = sublist[0]
        function2_values = sublist[1]
        means_function1.append(np.mean(function1_values))
        std_dev_function1.append(np.std(function1_values))
        means_function2.append(np.mean(function2_values))
        std_dev_function2.append(np.std(function2_values))
    for index in range(len(accuracy_values1)):
        extract_statistics(accuracy_values1, means_accuracy1, std_accuracy1, index)
        extract_statistics(accuracy_values2, means_accuracy2, std_accuracy2, index)
        extract_statistics(loss_values1, means_loss1, std_loss1, index)
        extract_statistics(loss_values2, means_loss2, std_loss2, index)

    # Create a figure
    fig, ax = plt.subplots(1, 3, figsize=(24, 6))
    ax[0].errorbar(np.arange(number_of_epochs, 2 * number_of_epochs + small_epochs, small_epochs),
                   means_function1[int(number_of_epochs / small_epochs):],
                   yerr=std_dev_function1[int(number_of_epochs / small_epochs):],
                   marker='o', linestyle='-', capsize=5, color='orange')
    ax[0].errorbar(np.arange(0, number_of_epochs + small_epochs, small_epochs),
                   means_function1[:int(number_of_epochs / small_epochs + 1)],
                   yerr=std_dev_function1[:int(number_of_epochs / small_epochs + 1)], label='Without Dropout',
                   marker='o', linestyle='-', capsize=5, color='blue')
    ax[0].errorbar(np.arange(number_of_epochs, 2 * number_of_epochs + small_epochs, small_epochs),
                   means_function2[int(number_of_epochs / small_epochs):],
                   yerr=std_dev_function2[int(number_of_epochs / small_epochs):],
                   marker='o', linestyle='-', capsize=5, color='blue')
    ax[0].errorbar(np.arange(0, number_of_epochs + small_epochs, small_epochs),
                   means_function2[:int(number_of_epochs / small_epochs + 1)],
                   yerr=std_dev_function2[:int(number_of_epochs / small_epochs + 1)], label='With Dropout',
                   marker='o', linestyle='-', capsize=5, color='orange')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Number of activation regions')
    ax[0].legend()
    ax[0].grid(True)
    # Loss and accuracy plots
    ax[1].errorbar(np.arange(small_epochs, 2*number_of_epochs + small_epochs, step=small_epochs), means_loss1,
                   yerr=std_loss1, color='b')
    ax[1].set_xlabel(r'Epoch')
    ax[1].set_ylabel('Loss', color='b')  # Set the y-axis label for loss
    ax2 = ax[1].twinx()
    ax2.errorbar(np.arange(small_epochs, 2 * number_of_epochs + small_epochs, step=small_epochs), means_accuracy1,
                 yerr=std_accuracy1, color='r')
    ax2.set_ylabel('Accuracy (%)', color='r')  # Set the y-axis label for accuracy
    ax[1].set_title('Starting without dropout')
    ax[1].grid()

    ax[2].errorbar(np.arange(small_epochs, 2 * number_of_epochs + small_epochs, step=small_epochs), means_loss2,
                   yerr=std_loss2, color='b')
    ax[2].set_xlabel(r'Epoch')
    ax[2].set_ylabel('Loss', color='b')  # Set the y-axis label for loss
    ax3 = ax[2].twinx()
    ax3.errorbar(np.arange(small_epochs, 2 * number_of_epochs + small_epochs, step=small_epochs), means_accuracy2,
                 yerr=std_accuracy2, color='r')
    ax3.set_ylabel('Accuracy (%)', color='r')  # Set the y-axis label for accuracy
    ax[2].set_title('Starting with dropout')
    ax[2].grid()
    # Show and save the combined figure
    plt.savefig('Experiment4.pdf')
    plt.savefig('Experiment4.svg')
    plt.savefig('Experiment4.png')
    plt.show()
    print('Done!')
