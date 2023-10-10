from copy import deepcopy
import pickle
import random
from typing import Dict, List, Tuple

from shapely.geometry import Polygon

from hyperrectangle import Hyperrectangle
from linear_region import LinearRegion
from skeleton import Skeleton


class SkelEx:
    def __init__(self, parameters, point_bank, hyperrectangle, dropout=0, error=1e-5):
        self.point_bank = point_bank
        self.R = hyperrectangle
        self.parameters = list(parameters)
        self.error = error
        self.dropout = dropout

    def quantize_to_0(self, variable):
        if -self.error < variable < self.error:
            return 0
        return variable

    def calculate_pre_activations(self, w: List[List[float]], b: List[float], R: Hyperrectangle,
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
                values[v] = self.quantize_to_0(values[v])
                point_bank[v] = 0.5
            g = [self.quantize_to_0(v) for v in w[neuron_index]]
            # Convert to NewSkeleton class and pass to the list
            skeletons.append(Skeleton([LinearRegion(Polygon([tl, tr, br, bl]), g)], R, values))
        return skeletons

    def merge_activations(self, act: List[Skeleton], w: List[List[float]], b: List[float],
                          point_bank: Dict[Tuple[float, float], float], critical_point_creation_index: float,
                          percentages, error=1e-5) -> List[Skeleton]:
        skeletons = []  # list containing skeleton of each neuron (n_l in total)
        w_i = percentages
        for n2_index in range(len(b)):  # go through each neuron
            # print(f'|  Neuron {n2_index + 1}')
            index = 0  # Look for the first non-zero weight
            while index < len(w[n2_index]) and w_i[index] < self.dropout:
                index += 1
            if index == len(w[n2_index]):
                R = act[0].hyperrectangle
                tl = (R.x[0], R.y[1])
                tr = (R.x[1], R.y[1])
                br = (R.x[1], R.y[0])
                bl = (R.x[0], R.y[0])
                values = {tl: 0, tr: 0, br: 0, bl: 0}
                current_skeleton = Skeleton([LinearRegion(Polygon([tl, tr, br, bl]), [0, 0])], R, values)
            else:
                current_skeleton = deepcopy(act[index])  # take the first neuron from previous layer
                current_skeleton *= w[n2_index][index]
                for n1_index in range(index + 1, len(act)):
                    if w_i[index] >= self.dropout:
                        # add together all the neurons from previous layer that have big enough weight
                        skeleton = deepcopy(act[n1_index])
                        skeleton *= w[n2_index][n1_index]
                        current_skeleton = current_skeleton.add_skeleton(skeleton, point_bank,
                                                                         critical_point_creation_index)
            current_skeleton += b[n2_index]
            skeletons.append(current_skeleton)
        return skeletons

    def save_skeleton(self, skeletons_of_learned_decision_functions):
        with open('skeletons.pkl', 'wb') as f:
            pickle.dump(skeletons_of_learned_decision_functions, f)
        with open('points.pkl', 'wb') as f:
            pickle.dump(self.point_bank, f)

    def main(self, percentages=None):
        # Extract weights and biases of the trained NN, and calculate pre-activations of the first hidden layer
        weights = self.parameters[0].data.tolist()
        biases = self.parameters[1].data.tolist()
        # print('-------------------------------------------------------------------------------------------\n|Layer 1')
        pre_activations = self.calculate_pre_activations(weights, biases, self.R, self.point_bank)
        for layer_index in range(1, len(self.parameters) // 2):
            # print('| ReLU')
            activations = []
            for pre_activation_index in range(len(pre_activations)):
                pre_activation = pre_activations[pre_activation_index]
                activation = pre_activation.relu(self.point_bank, layer_index)
                activations.append(activation)
            # print(f'|Layer {layer_index + 1}')
            pre_activations = self.merge_activations(activations, self.parameters[2 * layer_index].data.tolist(),
                                                     self.parameters[2 * layer_index + 1].data.tolist(),
                                                     self.point_bank, layer_index + 0.5, percentages[layer_index-1])
        for skeleton in pre_activations:
            skeleton.test_validity()
        """print('------------------------------------------------------------------------------------------------------\n'
              'A total of ' + str(len(pre_activations[0].linear_regions)) + ' linear regions were formed.')"""
        self.save_skeleton(pre_activations)
        return pre_activations
