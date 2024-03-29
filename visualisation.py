import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
import numpy as np
import torch


class Visualizer:
    def __init__(self, skeletons_of_learned_decision_functions, trained_model, hyperrectangle, number_of_classes):
        self.model = trained_model
        self.k = number_of_classes
        self.R = hyperrectangle
        self.skeletons = skeletons_of_learned_decision_functions

    def decision_functions(self, x1, x2):
        output = []
        for k in range(self.k):
            decision_function = []
            for r in range(len(x1)):
                decision_function.append([])
                for c in range(len(x1[r])):
                    result = self.model(torch.tensor([[x1[r][c], x2[r][c]]]))
                    decision_function[r].append(result[0][k].item())
            output.append(np.array(decision_function))
        return np.array(output)

    def loss_f(self, x1, x2, class_index=0, loss=torch.nn.CrossEntropyLoss()):
        output = []
        for r in range(len(x1)):
            output.append([])
            for c in range(len(x1[r])):
                result = self.model(torch.tensor([[x1[r][c], x2[r][c]]]))
                output[r].append(loss(result, torch.tensor([class_index])).item())
        return np.array(output)

    def prepare_graph(self, title, mode='2D', rotation=0):
        if mode == '3D':
            ax = plt.figure(title).add_subplot(projection='3d')
            ax.view_init(37, 176)
            ax.set_zlabel(r'$z$', labelpad=2)
        else:
            ax = plt.figure(title, figsize=(7, 7)).add_subplot()
            ax.set_xlim(self.R.x)
            ax.set_ylim(self.R.y)
            if rotation != 0:
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position('top')
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position('right')
                labels = ax.get_xticklabels()
                for label in labels:
                    label.set_rotation(rotation)
                labels = ax.get_yticklabels()
                for label in labels:
                    label.set_rotation(rotation)
        ax.set_xlabel(r'$x_1$', labelpad=2).set_rotation(rotation)
        ax.set_ylabel(r'$x_2$', labelpad=2).set_rotation(rotation)

        return ax

    def add_skeleton_to_decision_landscape(self, dec_ax):
        """TODO:This does not look correct for edges of the skeleton that go through the decision boundary.
        To fix this we would have to make it so that for each such edge it is split into several edges.
        """
        for lr in self.skeletons[0].linear_regions:
            xx, yy = lr.polygon.exterior.coords.xy
            for v_i in range(len(xx) - 1):
                start_zs = [self.skeletons[k].values[(xx[v_i], yy[v_i])] for k in range(self.k)]
                end_zs = [self.skeletons[k].values[(xx[v_i + 1], yy[v_i + 1])] for k in range(self.k)]
                dec_ax.plot([xx[v_i], xx[v_i + 1]], [yy[v_i], yy[v_i + 1]], [max(start_zs), max(end_zs)], c='black')

    def plot_skeleton(self, point_bank=None, ax=None):
        self.skeletons[0].plot_skeleton('Membership function 2', ax, point_bank=point_bank, mode=1)

    @staticmethod
    def plot_decision_boundary(boundary_extractor, classification_polygons, lines_used, ax, data=None, save=False):
        if data:
            boundary_extractor.plot(classification_polygons, lines_used, data, add_data=True, my_ax=ax)
        else:
            boundary_extractor.plot(classification_polygons, lines_used, data, add_data=False, my_ax=ax)
        if save:
            plt.savefig('Figures/decision_boundary_2D_projection.pdf')

    def draw_loss_landscape(self, elev, azim, roll, class_index=0, save=False):
        X, Y = np.meshgrid(np.linspace(self.R.x[0], self.R.x[1], 50), np.linspace(self.R.y[0], self.R.y[1], 50))
        loss_ax = self.prepare_graph('loss landscape', '3D')
        loss_ax.view_init(elev=elev, azim=azim, roll=roll)
        loss_Z = self.loss_f(X, Y, class_index=class_index)
        loss_ax.plot_surface(X, Y, loss_Z, rstride=1, cstride=1, edgecolor='none', alpha=0.6, zorder=1)
        if save:
            plt.savefig('Figures/loss_landscape.pdf')

    def draw_decision_landscape(self, elev, azim, roll, skeleton=False, decision=False, heatmap=False, save=False):
        if decision and heatmap:
            raise Exception('Cannot have both `decision` and `heatmap` set to True')
        colors_list = ['b', 'm', 'c', 'r', 'g', 'y', 'k', 'w', 'darkgrey', 'brown']
        rgb_colors = np.array([mcolors.to_rgb(color) for color in colors_list])
        X, Y = np.meshgrid(np.linspace(self.R.x[0], self.R.x[1], 50), np.linspace(self.R.y[0], self.R.y[1], 50))
        dec_ax = self.prepare_graph('decision landscape', '3D')
        dec_ax.view_init(elev=elev, azim=azim, roll=roll)
        decision_Zs = self.decision_functions(X, Y)

        # Compute the maximum decision value among all classes
        max_f = np.max(decision_Zs, axis=0)
        if decision:
            # Assign colors based on the class index
            class_indices = np.argmax(decision_Zs, axis=0)
            color_values = np.empty((*class_indices.shape, 3))  # Create a 3D array
            for r in range(class_indices.shape[0]):
                for c in range(class_indices.shape[1]):
                    color_values[r][c] = rgb_colors[class_indices[r][c]]
            dec_ax.plot_surface(X, Y, max_f, facecolors=color_values, alpha=0.6, linewidth=0)
        if heatmap:
            dec_ax.plot_surface(X, Y, max_f, cmap=cm.coolwarm, alpha=0.8, linewidth=0, antialiased=False)
        if skeleton:
            self.add_skeleton_to_decision_landscape(dec_ax)
        if save:
            name = 'Figures/'
            if skeleton:
                name += 's'
            if decision:
                name += 'd'
            if heatmap:
                name += 'h'
            name += '_decision_landscape.pdf'
            plt.savefig(name)
