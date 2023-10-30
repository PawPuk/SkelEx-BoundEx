from typing import Dict, List, Tuple, Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Point, Polygon

from dataset import Dataset2D
from linear_region import LinearRegion


class BoundEx:
    def __init__(self, skeletons_of_decision_functions, hyperrectangle):
        self.skeletons = skeletons_of_decision_functions
        self.hyperrectangle = hyperrectangle

    @staticmethod
    def test_order_of_linear_regions(linear_regions: List[LinearRegion]):
        """We know that all decision functions have the same skeleton (coordinate-wise). We want to assume that those
        linear regions have been created in the same order. We check whether this is the case here.

        @param linear_regions: list of linear regions (they all should be equal)
        """
        poly = linear_regions[0].polygon
        for i in range(1, len(linear_regions)):
            poly1 = linear_regions[i].polygon
            if not poly.equals(poly1):  # previously almost_equals (worked but depreciated)
                raise ValueError(str(poly) + " is different than " + str(poly1))

    @staticmethod
    def find_intersection(e1: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
                          e2: Tuple[Tuple[float, float, float], Tuple[float, float, float]]):
        x, y, z = (e1[0][0], e1[0][1], e1[0][2])
        x1, y1, z1 = (e1[1][0], e1[1][1], e1[1][2])
        zz = e2[0][2]
        zz1 = e2[1][2]
        dx, dy, dz = (x1 - x, y1 - y, z1 - z)
        dzz = zz1 - zz
        A = np.array([[dy, -dy], [dz, -dzz]])
        B = np.array([0, zz - z])
        A1 = np.array([[dx, -dx], [dz, -dzz]])
        B1 = np.array([0, zz - z])
        if np.linalg.det(A):
            solution = np.linalg.solve(A, B)
        else:
            solution = np.linalg.solve(A1, B1)
        return x + dx * solution[0], y + dy * solution[1]

    @staticmethod
    def find_v1(intersection, edge):
        if edge[0][0] != edge[1][0]:
            t = (intersection[0] - edge[0][0]) / (edge[1][0] - edge[0][0])
        else:
            t = (intersection[1] - edge[0][1]) / (edge[1][1] - edge[0][1])
        return edge[0][2] + t * (edge[1][2] - edge[0][2])

    def find_coordinates(self, start_values: List[float], end_values: List[float], start_p: Tuple[float, float],
                         end_p: Tuple[float, float], points: Dict[int, List[Tuple[float, float]]]):
        cma = self.argmax(start_values)  # current maximum argument
        p1 = start_p
        v1 = self.skeletons[cma].values[p1]
        decisions = [cma]
        while cma != self.argmax(end_values):
            e1 = ((p1[0], p1[1], v1), (end_p[0], end_p[1], self.skeletons[cma].values[end_p]))
            intersection = None
            new_decision = -1  # TODO: should not be necessary
            for i in [index for index in range(len(self.skeletons)) if index not in decisions]:
                e2 = ((start_p[0], start_p[1], self.skeletons[i].values[start_p]),
                      (end_p[0], end_p[1], self.skeletons[i].values[end_p]))
                current_intersection = self.find_intersection(e1, e2)
                # check if current_intersection lies on the edge e=(p1, p2)
                if LineString([p1, end_p]).distance(Point(current_intersection)) < 1e-5:
                    if intersection is None:
                        intersection = current_intersection
                        new_decision = i
                    # check if current_intersection is closer to p1 than intersection
                    elif Point(p1).distance(Point(current_intersection)) < Point(p1).distance(Point(intersection)):
                        intersection = current_intersection
                        new_decision = i
            self.append_to_dictionary(points, cma, intersection)
            self.append_to_dictionary(points, new_decision, intersection)
            v1 = self.find_v1(intersection, e1)
            p1 = intersection
            cma = new_decision
            decisions.append(new_decision)
        return points

    @staticmethod
    def append_to_dictionary(d, k, v):
        if k in d:
            d[k].append(v)
        else:
            d[k] = [v]

    @staticmethod
    def argmax(l):
        return max(enumerate(l), key=lambda x: x[1])[0]

    def extract_decision_boundary_from_skeletons_of_decision_functions(self):
        # TODO: test this once again with hyperrectangle starting at 0
        classification_polygons = {}
        lines_used = []
        for lr_index in range(len(self.skeletons[0].linear_regions)):
            classification_points = {}
            linear_region_variations = []
            for skeleton_index in range(len(self.skeletons)):
                linear_region_variations.append(self.skeletons[skeleton_index].linear_regions[lr_index])
            self.test_order_of_linear_regions(linear_region_variations)
            # find points where the decision changes and put them into decision dictionary
            if linear_region_variations[0].polygon.interiors:
                raise NotImplementedError

            xx, yy = linear_region_variations[0].polygon.exterior.coords.xy
            last_p = (xx[0], yy[0])
            last_values = [self.skeletons[i].values[last_p] for i in range(len(self.skeletons))]
            self.append_to_dictionary(classification_points, self.argmax(last_values), last_p)
            for point_index in range(1, len(xx)):  # go through each point of the linear region
                this_p = (xx[point_index], yy[point_index])
                this_values = [self.skeletons[i].values[this_p] for i in range(len(self.skeletons))]
                if self.argmax(last_values) != self.argmax(this_values):  # find edges on which decision changes
                    self.find_coordinates(last_values, this_values, last_p, this_p, classification_points)
                    if [last_p, this_p] not in lines_used and [this_p, last_p] not in lines_used:
                        lines_used.append([last_p, this_p])
                classification_points[self.argmax(this_values)].append(this_p)
                last_values = this_values
                last_p = this_p
            area1 = linear_region_variations[0].polygon.area
            area2 = 0
            for m in classification_points:
                if len(classification_points[m]) != 2:
                    area2 += Polygon(classification_points[m]).area
                    self.append_to_dictionary(classification_polygons, m, Polygon(classification_points[m]))
                else:
                    area2 += LineString(classification_points[m]).area
                    self.append_to_dictionary(classification_polygons, m, LineString(classification_points[m]))
            if abs(area1 - area2) > 1e-1:
                print(linear_region_variations[0].polygon)
                for m in classification_points:
                    print(Polygon(classification_points[m]))
                raise Exception(str(area1) + ' != ' + str(area2))
        for m in classification_polygons:
            while len(classification_polygons[m]) > 1:
                if classification_polygons[m][0].intersection(classification_polygons[m][1]).area > 1e-1:
                    print(classification_polygons[m][0].intersection(classification_polygons[m][1]).area)
                    raise Exception
                a1 = classification_polygons[m][0].area + classification_polygons[m][1].area
                classification_polygons[m][0] = classification_polygons[m][0].union(classification_polygons[m][1])
                a2 = classification_polygons[m][0].area
                if abs(a1 - a2) > 1e-1:
                    print(a1)
                    print(a2)
                    raise Exception
                del classification_polygons[m][1]
        return classification_polygons, lines_used

    @staticmethod
    def classify(point, decision_boundary):
        for decision in range(len(decision_boundary) - 1):
            poly = decision_boundary[decision][0]
            if poly.contains(point):
                return decision
        return len(decision_boundary) - 1

    def plot(self, classification_polygons, lines_used, data, add_data=False, my_ax=None, skeleton=None):
        color = ['b', 'm', 'c', 'r', 'g', 'y', 'k', 'w', 'darkgrey', 'brown']
        if my_ax is None:
            ax = plt.figure('Decision boundary', figsize=(7, 7)).add_subplot()
            ax.set_xlim(self.hyperrectangle.x)
            ax.set_ylim(self.hyperrectangle.y)
            ax.set_xlabel(r'$x_1$', labelpad=2)
            ax.set_ylabel(r'$x_2$', labelpad=2)
        else:
            ax = my_ax
        if add_data:
            Dataset2D.plot(data, self.hyperrectangle, ax)
        for m in range(len(classification_polygons)):
            for l in classification_polygons[m]:
                p = gpd.GeoSeries(l)
                p.plot(ax=ax, color=color[m], zorder=1, alpha=0.2)
            if my_ax is not None:
                for l in lines_used:
                    pass
                    # ax.scatter(l[0][0], l[0][1], s=10, c='black', zorder=3, clip_on=False)
                    # ax.scatter(l[1][0], l[1][1], s=10, c='black', zorder=3, clip_on=False)
                    # ax.plot([l[0][0], l[1][0]], [l[0][1], l[1][1]], zorder=2, color='black')
            if skeleton is not None:
                for lr in skeleton.linear_regions:
                    for l in lines_used:
                        if isinstance(lr.polygon.intersection(LineString(l)), LineString) and \
                                lr.polygon.distance(LineString(l)) == 0:
                            lr.new_plot(ax, skeleton.values)
                            break
                for l in lines_used:
                    ax.plot([l[0][0], l[1][0]], [l[0][1], l[1][1]], zorder=2, color='white')