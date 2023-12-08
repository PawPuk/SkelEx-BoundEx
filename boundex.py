from typing import Dict, List, Tuple, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Point, Polygon, MultiPoint

from dataset import Dataset2D
from linear_region import LinearRegion


class BoundEx:
    def __init__(self, skeletons_of_decision_functions, hyperrectangle):
        self.skeletons = skeletons_of_decision_functions
        self.hyperrectangle = hyperrectangle

    @staticmethod
    def test_equality_of_variants(linear_regions: List[LinearRegion]):
        """We want to verify if all variants of the linear region collected from different tessellations are equal as
        they should be.

        @param linear_regions: list of linear regions (they all should be equal)
        """
        poly = linear_regions[0].polygon
        for i in range(1, len(linear_regions)):
            poly1 = linear_regions[i].polygon
            if not poly.equals(poly1):  # previously almost_equals (worked but depreciated)
                raise ValueError(str(poly) + " is different than " + str(poly1))

    @staticmethod
    def find_line_segment_intersection(e1: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
                                       e2: Tuple[Tuple[float, float, float], Tuple[float, float, float]]):
        """ Given two edges of a 3D shape find the intersection if it exists

        :param e1: edge 1 as a tuple of (x, y, z) coordinates
        :param e2: edge 2 as a tuple of (x, y, z) coordinates
        :return: (x, y) of the intersection or None if no intersection
        """
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
    def calculate_value_of_point_in_polygon(point: Tuple[float, float], polygon: Polygon,
                                            values: Dict[Tuple[float, float], float], gradient: Tuple[float, float]):
        poly_x, poly_y = polygon.exterior.coords.xy
        x, y, z = poly_x[0], poly_y[0], values[poly_x[0], poly_y[0]]
        value = (point[0] - x) * gradient[0] + (point[1] - y) * gradient[1] + z
        return value

    def find_polygon_intersection(self, subpolygon1: Polygon, polygon1: Polygon, polygon2: Polygon,
                                  values1: Dict[Tuple[float, float], float], gradient1: Tuple[float, float],
                                  values2: Dict[Tuple[float, float], float], gradient2: Tuple[float, float]) -> \
            Union[Tuple[Polygon, Polygon], Tuple[Polygon, None], Tuple[None, Polygon]]:
        """

        :param subpolygon1: subpolygon; this is where polygon1 had the highest value among the checked lr variants
        :param polygon1: polygon of one of the earlier classes
        :param polygon2: polygon (a new variant of current linear region)
        :param values1: dictionary storing values of all vertices of polygon1
        :param values2: dictionary storing values of all vertices of polygon2
        :param gradient1: gradient of polygon1
        :param gradient2: gradient of polygon2
        :return:
        """

        xx1, yy1 = subpolygon1.exterior.coords.xy
        subpolygon1_vertices, subpolygon2_vertices = [], []
        for vertex_index in range(1, len(xx1)):
            prev_zz1 = self.calculate_value_of_point_in_polygon((xx1[vertex_index - 1], yy1[vertex_index - 1]),
                                                                polygon1, values1, gradient1)
            prev_zz2 = self.calculate_value_of_point_in_polygon((xx1[vertex_index - 1], yy1[vertex_index - 1]),
                                                                polygon2, values2, gradient2)
            zz1 = self.calculate_value_of_point_in_polygon((xx1[vertex_index], yy1[vertex_index]), polygon1, values1,
                                                           gradient1)
            zz2 = self.calculate_value_of_point_in_polygon((xx1[vertex_index], yy1[vertex_index]), polygon2, values2,
                                                           gradient2)
            # Find which class (xx1[vertex_index], yy1[vertex_index]) belongs to
            target_list = subpolygon2_vertices if zz1 < zz2 else subpolygon1_vertices
            target_list.append((xx1[vertex_index], yy1[vertex_index]))
            if zz1 == zz2:
                subpolygon2_vertices.append((xx1[vertex_index], yy1[vertex_index]))
            # Check if (xx1[vertex_index - 1], yy1[vertex_index - 1]) belongs to different class
            if (prev_zz1 > prev_zz2 and zz1 < zz2) or (prev_zz1 < prev_zz2 and zz1 > zz2):
                x_start, x_end = xx1[vertex_index - 1], xx1[vertex_index]
                y_start, y_end = yy1[vertex_index - 1], yy1[vertex_index]
                x, y = self.find_line_segment_intersection(((x_start, y_start, prev_zz1), (x_end, y_end, zz1)),
                                                           ((x_start, y_start, prev_zz2), (x_end, y_end, zz2)))
                subpolygon1_vertices.append((x, y))
                subpolygon2_vertices.append((x, y))
        subpolygon1 = Polygon(MultiPoint(subpolygon1_vertices).convex_hull) if len(subpolygon1_vertices) > 2 else None
        subpolygon2 = Polygon(MultiPoint(subpolygon2_vertices).convex_hull) if len(subpolygon2_vertices) > 2 else None
        return subpolygon1, subpolygon2

    def boundex(self):
        classification_polygons = {}
        lines_used = []
        # Iterate through all activation regions of the output tessellations
        for lr_index in range(len(self.skeletons[0].linear_regions)):
            # Collect all k versions of the final tessellations
            linear_region_variations = [self.skeletons[skeleton_index].linear_regions[lr_index]
                                        for skeleton_index in range(len(self.skeletons))]
            # Asses if they are the same (they should be)
            self.test_equality_of_variants(linear_region_variations)
            # Store which parts of lr belong to which class
            current_subpolygons = {0: linear_region_variations[0].polygon}
            # Iterate through all lr variants (in 3D)
            for class_index in range(1, len(self.skeletons)):
                current_subpolygons[class_index] = []
                current_lr = linear_region_variations[class_index].polygon
                current_gradient = linear_region_variations[class_index].gradient
                # Next go through variants from 0 to (class_index - 1) and see if they intersect with current_lr in 3D
                for subpolygon_class in range(class_index):
                    subpolygon = current_subpolygons[subpolygon_class]
                    subpolygon_gradient = linear_region_variations[subpolygon_class].gradient
                    # Continue iff part of the lr belongs to subpolygon_class
                    if subpolygon is not None:
                        subpolygon1, subpolygon2 = self.find_polygon_intersection(
                            subpolygon, linear_region_variations[subpolygon_class].polygon, current_lr,
                            self.skeletons[subpolygon_class].values, subpolygon_gradient,
                            self.skeletons[class_index].values,  current_gradient)
                        """if lr_index == 159:
                            polygons.append(subpolygon1)
                            polygons.append(subpolygon2)"""
                        current_subpolygons[subpolygon_class] = subpolygon1
                        if subpolygon2 is not None:
                            current_subpolygons[class_index].append(subpolygon2)
                if len(current_subpolygons[class_index]) > 0:
                    current_subpolygon = current_subpolygons[class_index][0]
                    for i in range(1, len(current_subpolygons[class_index])):
                        current_subpolygon = current_subpolygon.union(current_subpolygons[class_index][i])
                    current_subpolygons[class_index] = current_subpolygon
                else:
                    current_subpolygons[class_index] = None
            for class_index in current_subpolygons.keys():
                subpolygon = current_subpolygons[class_index]
                if subpolygon is not None:
                    if class_index in classification_polygons.keys():
                        classification_polygons[class_index].append(subpolygon)
                    else:
                        classification_polygons[class_index] = [subpolygon]
        return classification_polygons, []

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