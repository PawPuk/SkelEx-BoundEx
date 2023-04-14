import copy
import os
import pickle
from time import time
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
import torch

from boundex import BoundEx
from dataset import Dataset2D
from hyperrectangle import Hyperrectangle
from linear_region import LinearRegion
from ReLU_NN import ReLUNeuralNetwork
from skeleton import Skeleton
from skelex import SkelEx
from train import TrainedNeuralNetwork


def membership_functions(x1, x2):
    output = []
    for k in range(number_of_classes):
        membership_function = []
        for r in range(len(x1)):
            membership_function.append([])
            for c in range(len(x1[r])):
                result = trained_model(torch.tensor([[x1[r][c], x2[r][c]]]))
                membership_function[r].append(result[0][k].item())
        output.append(np.array(membership_function))
    return np.array(output)


def loss_f(x1, x2, class_index=0, loss=torch.nn.CrossEntropyLoss()):
    output = []
    for r in range(len(x1)):
        output.append([])
        for c in range(len(x1[r])):
            result = trained_model(torch.tensor([[x1[r][c], x2[r][c]]]))
            output[r].append(loss(result, torch.tensor([class_index])).item())
    return np.array(output)


def prepare_graph(title, mode='2D'):
    if mode == '3D':
        ax = plt.figure(title).add_subplot(projection='3d')
        ax.view_init(37, 176)
        ax.set_zlabel(r'$z$', labelpad=2)
    else:
        ax = plt.figure(title, figsize=(7, 7)).add_subplot()
        ax.set_xlim(hyperrectangle.x)
        ax.set_ylim(hyperrectangle.y)
    ax.set_xlabel(r'$x_1$', labelpad=2)
    ax.set_ylabel(r'$x_2$', labelpad=2)
    return ax

def add_skeleton_to_membership_landscape(mem_ax):
    """TODO:This does not look correct for edges of the skeleton that go through the decision boundary.
    To fix this we would have to make it so that for each such edge it is split into several edges.
    """
    for lr in skeletons_of_learned_membership_functions[0].linear_regions:
        xx, yy = lr.polygon.exterior.coords.xy
        for v_i in range(len(xx) - 1):
            start_zs = [skeletons_of_learned_membership_functions[k].values[(xx[v_i], yy[v_i])]
                        for k in range(number_of_classes)]
            end_zs = [skeletons_of_learned_membership_functions[k].values[(xx[v_i + 1], yy[v_i + 1])]
                      for k in range(number_of_classes)]
            mem_ax.plot([xx[v_i], xx[v_i + 1]], [yy[v_i], yy[v_i + 1]], [max(start_zs), max(end_zs)], c='black')


if __name__ == '__main__':
    # Set hyperparameters and generate data
    train = False
    create_figure_for_the_dataset = False
    create_2d_figures_for_membership_functions = False
    number_of_classes = 2
    number_of_parameters = 2
    layers_width = [number_of_parameters, 10, 10, number_of_classes]
    data_size = 1000
    dataset = None  # set to 'balance_set' to work with the balance scale UCI dataset
    number_of_epochs = 25
    global_point_bank = {}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        dev = torch.device('mps')

    if dataset == 'balance_scale':
        my_data, vis_data = Dataset2D()
    else:
        my_data = (Dataset2D(class_size=3*data_size).data.to(dev), Dataset2D(class_size=data_size).data.to(dev))
    hyperrectangle = Hyperrectangle(-1.3, 1.3, -1.3, 1.3)  # TODO: define using my_data!!!

    # Train NN and save it
    if os.path.isfile("model.pth") and not train:
        trained_model = ReLUNeuralNetwork(layers_width)
        trained_model.load_state_dict(torch.load("model.pth"))
    else:
        my_model = TrainedNeuralNetwork(ReLUNeuralNetwork(layers_width, initialization='xavier'), my_data,
                                        number_of_parameters, epochs=number_of_epochs, wd=1e-4, lr=1e-3, opt='ADAM',
                                        mode=dataset)
        trained_model = my_model.main()
        torch.save(trained_model.state_dict(), "model.pth")

    if os.path.isfile('skeletons.pkl') and not train:
        with open('skeletons.pkl', 'rb') as f:
            skeletons_of_learned_membership_functions = pickle.load(f)
        with open('points.pkl', 'rb') as f:
            global_point_bank = pickle.load(f)
    else:
        t0 = time()
        skelex = SkelEx(trained_model.parameters(), global_point_bank, hyperrectangle)
        skeletons_of_learned_membership_functions = skelex.main()
        print(time()-t0)
        with open('skeletons.pkl', 'wb') as f:
            pickle.dump(skeletons_of_learned_membership_functions, f)
        with open('points.pkl', 'wb') as f:
            pickle.dump(global_point_bank, f)
    boundary_extractor = BoundEx(skeletons_of_learned_membership_functions, hyperrectangle)
    classification_polygons, lines_used = \
        boundary_extractor.extract_decision_boundary_from_skeletons_of_membership_functions()
    print(f'The decision boundary is created via {len(lines_used)} line segments.')
    ax4 = prepare_graph('Skeleton tessellation')
    skeletons_of_learned_membership_functions[0].plot_skeleton('Membership function 2',
                                                                ax=ax4, point_bank=global_point_bank, mode=1)
    if dataset == 'balance_scale':
        boundary_extractor.plot(classification_polygons, lines_used, vis_data, add_data=True)
    else:
        boundary_extractor.plot(classification_polygons, lines_used, my_data, add_data=False, my_ax=ax4)
        # boundary_extractor.plot(classification_polygons, lines_used, my_data, add_data=True, my_ax=ax4)

    print('Done!')

    if create_figure_for_the_dataset:
        Dataset2D.plot(my_data, hyperrectangle)
        plt.savefig('Figures/spiral_dataset2.pdf')
    if create_2d_figures_for_membership_functions:
        for i1 in range(len(skeletons_of_learned_membership_functions)):
            skeletons_of_learned_membership_functions[i1].plot_skeleton('Membership function ' + str(i1))
            plt.savefig('Figures/membership_function ' + str(i1))

    plt.show()
