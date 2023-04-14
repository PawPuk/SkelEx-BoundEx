import os
import pickle
from time import time

import matplotlib.pyplot as plt
import torch

from boundex import BoundEx
from dataset import Dataset2D
from hyperrectangle import Hyperrectangle
from ReLU_NN import ReLUNeuralNetwork
from skelex import SkelEx
from train import TrainedNeuralNetwork
from visualisation import Visualizer


if __name__ == '__main__':
    # Set hyperparameters and generate data
    train = False
    create_figure_for_the_dataset = False
    create_2d_figures_for_decision_functions = False
    number_of_classes = 2
    number_of_parameters = 2
    layers_width = [number_of_parameters, 10, 10, 10, number_of_classes]
    data_size = 1000
    dataset = None  # set to 'balance_set' to work with the balance scale UCI dataset
    number_of_epochs = 50
    global_point_bank = {}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    """if torch.backends.mps.is_available():
        dev = torch.device('mps')"""
    # Prepare dataset and hyperrectangle
    if dataset == 'balance_scale':
        my_data, vis_data = Dataset2D()
    else:
        my_data = (Dataset2D(class_size=3*data_size).data.to(dev), Dataset2D(class_size=data_size).data.to(dev))
        vis_data = my_data
    hyperrectangle = Hyperrectangle(my_data)
    # Train NN and save it or load trained network
    if os.path.isfile("model.pth") and not train:
        trained_model = ReLUNeuralNetwork(layers_width)
        trained_model.load_state_dict(torch.load("model.pth"))
    else:
        my_model = TrainedNeuralNetwork(ReLUNeuralNetwork(layers_width, initialization='xavier'), my_data,
                                        number_of_parameters, epochs=number_of_epochs, wd=1e-4, lr=1e-3, opt='ADAM',
                                        mode=dataset)
        trained_model = my_model.main()
        torch.save(trained_model.state_dict(), "model.pth")
    # Load skeleton if SkelEx was already run. Run SkelEx otherwise
    if os.path.isfile('skeletons.pkl') and not train:
        with open('skeletons.pkl', 'rb') as f:
            skeletons_of_learned_decision_functions = pickle.load(f)
        with open('points.pkl', 'rb') as f:
            global_point_bank = pickle.load(f)
    else:
        t0 = time()
        skelex = SkelEx(trained_model.parameters(), global_point_bank, hyperrectangle)
        skeletons_of_learned_decision_functions = skelex.main()
        print("SkelEx finished within " + str(time()-t0) + " seconds.")
    # Run BoundEx
    boundary_extractor = BoundEx(skeletons_of_learned_decision_functions, hyperrectangle)
    classification_polygons, lines_used = \
        boundary_extractor.extract_decision_boundary_from_skeletons_of_decision_functions()
    print(f'The decision boundary is created via {len(lines_used)} line segments.')
    # Visualize results
    visualizer = Visualizer(skeletons_of_learned_decision_functions, trained_model, hyperrectangle, number_of_classes)
    ax = visualizer.prepare_graph("Skeleton tessellation", rotation=180)
    visualizer.plot_skeleton(None, ax)
    visualizer.plot_decision_boundary(boundary_extractor, classification_polygons, lines_used, ax, save=True)
    visualizer.draw_loss_landscape(25, 50, 0, class_index=1, save=True)   # set class_index=1 for blue class
    visualizer.draw_decision_landscape(25, 50, 0, skeleton=True, decision=True, heatmap=False, save=True)
    print('Done!')

    if create_figure_for_the_dataset:
        Dataset2D.plot(my_data, hyperrectangle)
        plt.savefig('Figures/spiral_dataset2.pdf')
    if create_2d_figures_for_decision_functions:
        for i1 in range(len(skeletons_of_learned_decision_functions)):
            skeletons_of_learned_decision_functions[i1].plot_skeleton('Membership function ' + str(i1))
            plt.savefig('Figures/decision_function ' + str(i1))

    plt.show()
