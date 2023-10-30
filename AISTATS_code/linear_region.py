from math import floor


class LinearRegion:
    def __init__(self, polygon, gradient):
        self.polygon = polygon
        self.gradient = gradient

    def __eq__(self, other):
        if isinstance(other, LinearRegion):
            return self.polygon.equals(other.polygon)
        return False

    def plot(self, ax, point_bank=None, lw=2, my_color='black', alpha=1, linestyle='solid'):
        xx, yy = self.polygon.exterior.coords.xy
        index = 0
        while index < len(xx) - 1:
            vertex = (xx[index], yy[index])
            next_vertex = (xx[index + 1], yy[index + 1])
            color = ['b', 'm', 'c', 'r', 'g', 'y', 'k', 'w', 'darkgrey', 'brown']
            if point_bank:
                ax.scatter(vertex[0], vertex[1], s=25, c=color[floor(point_bank[vertex] - 1)], zorder=2, clip_on=False)
            else:
                ax.scatter(vertex[0], vertex[1], s=25, c=color[0], zorder=2, clip_on=False)
            ax.plot([vertex[0], next_vertex[0]], [vertex[1], next_vertex[1]], c=my_color, linewidth=lw, zorder=1,
                    alpha=alpha, linestyle=linestyle)
            index += 1
        for geom in self.polygon.interiors:
            xx, yy = geom.coords.xy
            index = 0
            while index < len(xx) - 1:
                vertex = (xx[index], yy[index])
                next_vertex = (xx[index + 1], yy[index + 1])
                color = ['b', 'm', 'c', 'r', 'g', 'y', 'k', 'w', 'darkgrey', 'brown']
                if point_bank:
                    ax.scatter(vertex[0], vertex[1], s=25, c=color[floor(point_bank[vertex] - 1)], zorder=2,
                               clip_on=False)
                else:
                    ax.scatter(vertex[0], vertex[1], s=25, c=color[0], zorder=2, clip_on=False)
                ax.plot([vertex[0], next_vertex[0]], [vertex[1], next_vertex[1]], c=my_color, linewidth=lw, zorder=1,
                        alpha=alpha, linestyle=linestyle)
                index += 1

    def new_plot(self, ax, values, point_bank=None, mode=0):
        xx, yy = self.polygon.exterior.coords.xy
        self.plot_vertices_and_lines(xx, yy, ax, values, point_bank, mode)
        for geom in self.polygon.interiors:  # for linear regions with holes
            xx, yy = geom.coords.xy
            self.plot_vertices_and_lines(xx, yy, ax, values, point_bank, mode)

    @staticmethod
    def plot_vertices_and_lines(xx, yy, ax, values, point_bank, mode=0):
        index = 0
        while index < len(xx) - 1:
            vertex = (xx[index], yy[index])
            next_vertex = (xx[index + 1], yy[index + 1])
            colors = ['b', 'm', 'c', 'r', 'g', 'y', 'k', 'w', 'darkgrey', 'brown']
            # Plot vertices
            if point_bank and mode == 1:
                ax.scatter(vertex[0], vertex[1], s=25, c=colors[floor(point_bank[vertex] * 2 - 1)], zorder=2,
                           clip_on=False)
            elif mode == 0:
                if values[vertex] > 0:
                    color = 'red'
                elif values[vertex] < 0:
                    color = 'blue'
                else:
                    color = 'yellow'
                ax.scatter(vertex[0], vertex[1], s=25, c=color, zorder=2, clip_on=False)
            else:
                color = 'black'
            # Plot edges
            ax.plot([vertex[0], next_vertex[0]], [vertex[1], next_vertex[1]], c='black', linewidth=2, zorder=1,
                    alpha=1, linestyle='solid')
            index += 1
