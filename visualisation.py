import matplotlib.pyplot as plt
from matplotlib import cm
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

    def prepare_graph(self, title, mode='2D'):
        if mode == '3D':
            ax = plt.figure(title).add_subplot(projection='3d')
            ax.view_init(37, 176)
            ax.set_zlabel(r'$z$', labelpad=2)
        else:
            ax = plt.figure(title, figsize=(7, 7)).add_subplot()
            ax.set_xlim(self.R.x)
            ax.set_ylim(self.R.y)
        ax.set_xlabel(r'$x_1$', labelpad=2)
        ax.set_ylabel(r'$x_2$', labelpad=2)
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

    def plot_skeleton(self, point_bank, ax):
        self.skeletons[0].plot_skeleton('Membership function 2', ax, point_bank=point_bank, mode=1)

    @staticmethod
    def plot_decision_boundary(boundary_extractor, classification_polygons, lines_used, ax, data):
        boundary_extractor.plot(classification_polygons, lines_used, data, add_data=True, my_ax=ax)

    def draw_loss_landscape(self, class_index=0):
        X, Y = np.meshgrid(np.linspace(self.R.x[0], self.R.x[1], 50), np.linspace(self.R.y[0], self.R.y[1], 50))
        loss_ax = self.prepare_graph('loss landscape', '3D')
        loss_Z = self.loss_f(X, Y, class_index=class_index)
        loss_ax.plot_surface(X, Y, loss_Z, rstride=1, cstride=1, edgecolor='none', alpha=0.6, zorder=1)

    def draw_decision_landscape(self, skeleton=False, decision=False, heatmap=False):
        if decision and heatmap:
            raise Exception('Cannot have both `decision` and `heatmap` set to True')
        X, Y = np.meshgrid(np.linspace(self.R.x[0], self.R.x[1], 50), np.linspace(self.R.y[0], self.R.y[1], 50))
        dec_ax = self.prepare_graph('decision landscape', '3D')
        decision_Zs = self.decision_functions(X, Y)
        max_f = np.maximum(decision_Zs[0], decision_Zs[1])
        if decision:
            dec_ax.plot_surface(X, Y, max_f, facecolors=np.where(decision_Zs[0] >= decision_Zs[1], 'b', 'm'), alpha=0.5,
                                linewidth=0)
        if heatmap:
            dec_ax.plot_surface(X, Y, max_f, cmap=cm.coolwarm, alpha=0.8, linewidth=0, antialiased=False)
        if skeleton:
            self.add_skeleton_to_decision_landscape(dec_ax)