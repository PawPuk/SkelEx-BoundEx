import copy
import math
import os
import pickle
from statistics import stdev
import multiprocessing as mp
from multiprocessing import Array
from time import time
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

from boundex import BoundEx
from dataset import Dataset2D
from hyperrectangle import Hyperrectangle
from linear_region import LinearRegion
from ReLU_NN import ReLUNeuralNetwork
from skeleton import Skeleton
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


def basic_iterative_attack(loss, input_data, labels, copied_trained_model, eps, alpha,
                           mem_ax=None, loss_ax=None, iters=15, no_sign=False):
    fast_path = False
    gradients = []
    this_lr_grad = None
    fast_grad = None
    # Convert input data to NumPy array and then PyTorch tensor
    x = np.array(input_data, dtype=np.float32)
    x = torch.from_numpy(x).unsqueeze(0)
    # y = torch.tensor([labels])
    y = labels

    # Set model to evaluation mode
    copied_trained_model.eval()

    # Copy the original input to avoid modifying it
    adv_x = x.clone().detach()
    adv_x.requires_grad = True

    steps = [x[0].tolist()]
    # Run the iterations of the attack
    for _ in range(iters):
        if not torch.argmax(copied_trained_model(adv_x)).eq(y):
            """if MNIST_display:
                figure.add_subplot(2, 3, 2 + no_sign)
                plt.title('I-FG' + ['S', ''][no_sign] + 'M - ' + str(torch.argmax(trained_model(adv_x))))
                plt.imshow(adv_x.detach().numpy().squeeze(), cmap='gray')"""
            return steps, _, torch.norm(adv_x - x).item()
        # Compute the loss and the gradient of the loss with respect to the input
        cost = loss(copied_trained_model(adv_x), y)
        grad = torch.autograd.grad(cost, adv_x)[0]
        gradients.append(grad)
        """if _ > 1 and torch.all(torch.eq(grad[0], gradients[-3][0])) and not fast_path:
            this_lr_grad = grad
            fast_path = True
            grad = normalize_grad(torch.tensor([steps[-3]]) - adv_x, 0, no_sign)
            fast_grad = grad
        elif this_lr_grad is not None and torch.all(torch.eq(grad, this_lr_grad)) and fast_path:
            grad = fast_grad
        else:
            fast_path = False
            grad = normalize_grad(grad, 0, no_sign)"""
        grad = normalize_grad(grad, 0, no_sign)

        # Update the adversarial example by taking a step in the direction of the gradient
        adv_x = adv_x + alpha * grad

        # Clip the perturbation to ensure it stays within the epsilon bound
        adv_x = torch.min(torch.max(adv_x, x - eps), x + eps)

        # Ensure the adversarial example is still a valid input (i.e., within the input space)
        # adv_x = torch.clamp(adv_x, -1.3, 1.3)

        steps.append(adv_x[0].tolist())
        if loss_ax:
            loss_ax.scatter(adv_x[0].tolist()[0], adv_x[0].tolist()[1], loss(copied_trained_model(adv_x), y).item(),
                            c=['red', 'green'][no_sign])
        if mem_ax:
            mem_ax.scatter(adv_x[0].tolist()[0], adv_x[0].tolist()[1], torch.max(copied_trained_model(adv_x)).item(),
                           c=['red', 'green'][no_sign])
    # print('No adversarial example found using I-FG' + ['S', ''][no_sign] + 'M within ' + str(iters) + ' iterations.')
    return steps, 0, 0


def normalize_grad(grad, order, no_sign):
    if order == 0:
        if no_sign:
            return normalize_grad(grad, 1, no_sign)
        else:
            return normalize_grad(torch.sign(grad), 1, no_sign)
    else:
        """if dataset == 'MNIST':"""
        norm = math.sqrt(sum([math.pow(grad[0][0][row][col].item(), 2) for row in range(28)
                              for col in range(28)]))
        """else:
            norm = math.sqrt(math.pow(grad[0][0].item(), 2) + math.pow(grad[0][1].item(), 2))"""
        return grad / norm


def membership_attack(loss, input_data, labels, copied_trained_model,
                      mem_ax=None, loss_ax=None, eps=1, alpha=0.025, iters=5, no_sign=False):
    fast_path = False
    gradients = []
    this_lr_grad = None
    fast_grad = None
    x = np.array(input_data, dtype=np.float32)
    x = torch.from_numpy(x).unsqueeze(0)
    # y = torch.tensor([labels])
    y = labels

    # Set model to evaluation mode
    copied_trained_model.eval()

    # Copy the original input to avoid modifying it
    adv_x = x.clone().detach()
    adv_x.requires_grad = True

    steps = [x[0].tolist()]
    # Run the iterations of the attack
    for _ in range(iters):
        if not torch.argmax(copied_trained_model(adv_x)).eq(y):
            """if MNIST_display:
                figure.add_subplot(2, 3, 4 + no_sign)
                plt.title('I-FM' + ['S', ''][no_sign] + 'A - ' + str(torch.argmax(trained_model(adv_x))))
                plt.imshow(adv_x.detach().numpy().squeeze(), cmap='gray')"""
            return steps, _, torch.norm(adv_x - x).item()
        # Compute the loss and the gradient of the loss with respect to the input
        memberships = copied_trained_model(adv_x)[0][labels]
        grad = torch.autograd.grad(memberships, adv_x)[0]
        gradients.append(grad)
        """if _ > 1 and torch.all(torch.eq(grad[0], gradients[-3][0])) and not fast_path:
            this_lr_grad = grad
            fast_path = True
            grad = normalize_grad(torch.tensor([steps[-3]]) - adv_x, 0, no_sign)
            fast_grad = grad
        elif this_lr_grad is not None and torch.all(torch.eq(grad, this_lr_grad)) and fast_path:
            grad = fast_grad
        else:
            fast_path = False
            grad = normalize_grad(grad, 0, no_sign)"""
        grad = normalize_grad(grad, 0, no_sign)
        # Update the adversarial example by taking a step in the direction of the gradient
        adv_x = adv_x - alpha * grad

        # Clip the perturbation to ensure it stays within the epsilon bound
        adv_x = torch.min(torch.max(adv_x, x - eps), x + eps)

        # Ensure the adversarial example is still a valid input (i.e., within the input space)
        # adv_x = torch.clamp(adv_x, -1.3, 1.3)

        steps.append(adv_x[0].tolist())
        if loss_ax:
            loss_ax.scatter(adv_x[0].tolist()[0], adv_x[0].tolist()[1], loss(copied_trained_model(adv_x), y).item(),
                            c=['orange', 'olive'][no_sign])
        if mem_ax:
            mem_ax.scatter(adv_x[0].tolist()[0], adv_x[0].tolist()[1], torch.max(copied_trained_model(adv_x)).item(),
                           c=['orange', 'olive'][no_sign])
    # print('No adversarial example found using I-FM' + ['S', ''][no_sign] + 'A within ' + str(iters) + ' iterations.')
    return steps, 0, 0


def check_when_steps_change_gradient(steps, mode):
    if mode == 'loss':
        print('I-FGM is ascending along the path ' + str(steps))
    elif mode == 'mem':
        print('Iterative-Membership Attack is descending along the path ' + str(steps))
    else:
        raise ValueError('Incorrect value of the parameter mode')
    print('This means that it changes gradient at steps:')
    steps_direction = [steps[1][0]-steps[0][0], steps[1][1]-steps[0][1]]
    for step_index in range(2, len(steps)):
        current_direction = [steps[step_index][0]-steps[step_index-1][0],
                             steps[step_index][1]-steps[step_index-1][1]]
        if current_direction != steps_direction:
            print(f'    {step_index} to [{current_direction[0]:.6f}, 'f'{current_direction[1]:.6f}]')
            steps_direction = current_direction


def quantize_to_0(variable, error=1e-5):
    if -error < variable < error:
        return 0
    return variable


def calculate_pre_activations(w: List[List[float]], b: List[float], R: Hyperrectangle,
                              point_bank: Dict[Tuple[float, float], float]) -> List[Skeleton]:
    """ Given the bounding hyperrectangle, and the learned weights and biases extracts skeletons of the pre-activations
    of the first hidden layer

    @param w: learned weights
    @param b: learned biases
    @param R: bounding hyperrectangle
    @param point_bank: dictionary storing all points created by the network
    @return: n_1 skeletons, each representing the critical points of the pre-activations of the first hidden layer
    """
    skeletons = []
    for neuron_index in range(len(w)):
        values = {}
        # Create vertex for each corner of R
        tl = (R.x[0], R.y[1])
        tr = (R.x[1], R.y[1])
        br = (R.x[1], R.y[0])
        bl = (R.x[0], R.y[0])
        for v in [tl, tr, br, bl]:  # and calculate their values
            values[v] = w[neuron_index][0] * v[0] + w[neuron_index][1] * v[1] + b[neuron_index]
            values[v] = quantize_to_0(values[v])
            point_bank[v] = 0.5
        g = [quantize_to_0(v) for v in w[neuron_index]]
        # Convert to NewSkeleton class and pass to the list
        skeletons.append(Skeleton([LinearRegion(Polygon([tl, tr, br, bl]), g)], R, values))
    return skeletons


def merge_activations(act: List[Skeleton], w: List[List[float]], b: List[float],
                      point_bank: Dict[Tuple[float, float], float], critical_point_creation_index: float,
                      error=1e-5) -> List[Skeleton]:
    skeletons = []  # list containing skeleton of each neuron (n_l in total)
    for n2_index in range(len(b)):  # go through each neuron
        print(f'|  Neuron {n2_index+1}')
        w_i = 0  # Look for the first non-zero weight
        while w_i < len(w[n2_index]) and -error < w[n2_index][w_i] < error:
            w_i += 1
        if w_i == len(w[n2_index]):
            R = act[0].hyperrectangle
            tl = (R.x[0], R.y[1])
            tr = (R.x[1], R.y[1])
            br = (R.x[1], R.y[0])
            bl = (R.x[0], R.y[0])
            values = {tl: 0, tr: 0, br: 0, bl: 0}
            current_skeleton = Skeleton([LinearRegion(Polygon([tl, tr, br, bl]), [0, 0])], R, values)
        else:
            current_skeleton = copy.deepcopy(act[w_i])  # take the first neuron from previous layer
            current_skeleton *= w[n2_index][w_i]
            for n1_index in range(w_i+1, len(act)):
                if not -1e-5 < w[n2_index][n1_index] < 1e-5:
                    # add together all the neurons from previous layer that have big enough weight
                    skeleton = copy.deepcopy(act[n1_index])
                    skeleton *= w[n2_index][n1_index]
                    current_skeleton = current_skeleton.add_skeleton(skeleton, point_bank,
                                                                     critical_point_creation_index)
        current_skeleton += b[n2_index]
        skeletons.append(current_skeleton)
    return skeletons


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


def skelex(parameters):
    # Extract weights and biases of the trained NN, and calculate pre-activations of the first hidden layer
    weights = parameters[0].data.tolist()
    biases = parameters[1].data.tolist()

    print(
        '---------------------------------------------------------------------------------------------------\n|Layer 1')
    pre_activations = calculate_pre_activations(weights, biases, hyperrectangle, global_point_bank)
    for layer_index in range(1, len(parameters) // 2):
        print('| ReLU')
        activations = []
        for pre_activation_index in range(len(pre_activations)):
            pre_activation = pre_activations[pre_activation_index]
            activation = pre_activation.new_relu(global_point_bank, layer_index)
            activations.append(activation)
        print(f'|Layer {layer_index + 1}')
        pre_activations = merge_activations(activations, parameters[2 * layer_index].data.tolist(),
                                            parameters[2 * layer_index + 1].data.tolist(), global_point_bank,
                                            layer_index + 0.5)
    for skeleton in pre_activations:
        skeleton.test_validity()
    print(
        '------------------------------------------------------------------------------------------------------------\n'
        'A total of ' + str(len(pre_activations[0].linear_regions)) + ' linear regions were formed.')
    return pre_activations


def analyse_adversarial_example(x, label, my_not_found, my_speeds, copied_trained_model, iters=40):
    """if dataset != 'MNIST':
        my_ax = prepare_graph('Skeleton tessellation')
        skeletons_of_learned_membership_functions[0].plot_skeleton('Membership function 2',
                                                                   ax=my_ax, point_bank=global_point_bank, mode=1)"""
    """mem_ax = prepare_graph('membership landscape', '3D')
    membership_Zs = membership_functions(X, Y)
    max_f = np.maximum(membership_Zs[0], membership_Zs[1])
    mem_ax.plot_surface(X, Y, max_f, facecolors=np.where(membership_Zs[0] >= membership_Zs[1], 'b', 'm'), alpha=0.5,
                        linewidth=0)
    add_skeleton_to_membership_landscape(mem_ax)"""

    """loss_ax = prepare_graph('loss landscape', '3D')
    loss_Z = loss_f(X, Y, class_index=input_data_sample_label.item())
    loss_ax.plot_surface(X, Y, loss_Z, rstride=1, cstride=1, edgecolor='none', alpha=0.6, zorder=1)"""

    t1 = time()
    loss_steps, s1, d1 = basic_iterative_attack(torch.nn.CrossEntropyLoss(), x, label, copied_trained_model,
                                                1, 0.03, iters=iters, no_sign=True)
    if d1 == s1 == 0:
        my_not_found[0] += 1
    else:
        my_speeds[0].append((time() - t1) * 1000)

    t2 = time()
    signed_loss_steps, s2, d2 = basic_iterative_attack(torch.nn.CrossEntropyLoss(), x, label, copied_trained_model,
                                                       1, 0.03, iters=iters)
    if d2 == s2 == 0:
        my_not_found[1] += 1
    else:
        my_speeds[1].append((time() - t2) * 1000)
    # check_when_steps_change_gradient(loss_steps, 'loss')

    t3 = time()
    membership_steps, s3, d3 = membership_attack(torch.nn.CrossEntropyLoss(), x, label, copied_trained_model,
                                                 alpha=0.03, iters=iters, no_sign=True)
    if d3 == s3 == 0:
        my_not_found[2] += 1
    else:
        my_speeds[2].append((time() - t3) * 1000)

    t4 = time()
    signed_membership_steps, s4, d4 = membership_attack(torch.nn.CrossEntropyLoss(), x, label, copied_trained_model,
                                                        alpha=0.03, iters=iters)
    if d4 == s4 == 0:
        my_not_found[3] += 1
    else:
        my_speeds[3].append((time() - t4) * 1000)

    """if dataset != 'MNIST':
        adversarial_steps = [loss_steps, membership_steps, signed_loss_steps, signed_membership_steps]
        boundary_extractor.plot(classification_polygons, lines_used, my_data, steps=adversarial_steps, add_data=False,
                                my_ax=my_ax)"""
    """if d1 == 0 or d3 == 0:
        print(len(loss_steps))
        print(len(membership_steps))
        print(input_data_sample)
        print(d1, d2, d3, d4)
        my_ax = prepare_graph('Skeleton tessellation')
        skeletons_of_learned_membership_functions[0].plot_skeleton('Membership function 2',
                                                                   ax=my_ax, point_bank=global_point_bank, mode=1)
        adversarial_steps = [loss_steps, membership_steps, signed_loss_steps, signed_membership_steps]
        boundary_extractor.plot(classification_polygons, lines_used, my_data,
                                steps=adversarial_steps, add_data=False, my_ax=my_ax)
        plt.show()"""
    # check_when_steps_change_gradient(membership_steps, 'mem')
    return [loss_steps, membership_steps, signed_loss_steps, signed_membership_steps], [s1, s2, s3, s4], \
        [d1, d2, d3, d4]


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
    skeletons_of_learned_membership_functions = skelex(list(trained_model.parameters()))
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
