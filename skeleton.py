import copy
import math
from typing import Tuple, List, Dict, Union, Any

import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPoint
from shapely.geometry import Point, GeometryCollection, MultiPolygon
from shapely.ops import split

from linear_region import LinearRegion
from hyperrectangle import Hyperrectangle


class Skeleton:
    def __init__(self, linear_regions: List[LinearRegion], hyperrectangle: Hyperrectangle,
                 values: Dict[Tuple[float, float], float]):
        self.linear_regions = linear_regions
        self.hyperrectangle = hyperrectangle
        self.values = values

    def __add__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            for key in self.values:
                self.values[key] += other
                if -pow(10, -10) < self.values[key] < pow(10, -10):
                    self.values[key] = 0
            return self
        else:
            raise TypeError("Can only multiply by a float or int")

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            for key in self.values:
                self.values[key] *= other
                if -pow(10, -10) < self.values[key] < pow(10, -10):
                    self.values[key] = 0
            for linear_region in self.linear_regions:
                for index in range(len(linear_region.gradient)):
                    linear_region.gradient[index] *= other
                    if -pow(10, -10) < linear_region.gradient[index] < pow(10, -10):
                        linear_region.gradient[index] = 0
            return self
        else:
            raise TypeError("Can only multiply by a float or int")

    @staticmethod
    def find_the_closest_point_from_the_bank(point_bank: Dict[Tuple[float, float], float], p: Point, index: float,
                                             values: Union[Dict[Tuple[float, float], float], None],
                                             error=pow(10, -13)) -> Tuple[float, float]:
        # go through all the points from the point_bank
        xx, yy = p.coords.xy
        if (xx[0], yy[0]) in point_bank:  # adding this increases speed as dictionary lookup is O(1).
            return xx[0], yy[0]
        for bank_p in point_bank:
            bank_x, bank_y = bank_p
            if abs(xx[0] - bank_x) < error and abs(yy[0] - bank_y) < error:
                # p was already in bank_p (using 'in' would not work due to floating point error)
                return bank_p
        xx[0], yy[0] = round(xx[0], 15), round(yy[0], 15)
        # p is not in point_bank (it's a new, just created point) so add it there
        point_bank[(xx[0], yy[0])] = index
        # add to values (this critical point was made during applying ReLU)
        if values and (xx[0], yy[0]) not in values:
            values[(xx[0], yy[0])] = 0
        return xx[0], yy[0]

    def calculate_value_of_point_inside_skeleton(self, point: Tuple[float, float], skeleton: "Skeleton",
                                                 gradient: List[float]) -> float:
        for ar in skeleton.linear_regions:
            """print('-------------------------------------------------------------------------')
            print(ar.polygon)
            print(point)
            print(ar.polygon.contains(Point(point)))
            print(self.is_point_inside_polygon(point, ar.polygon))
            print(self.distance_to_segment(point, (-1.2, -1.5), (-1.2, 1.4)))
            print(ar.polygon)"""
            """# Extract coordinates from the polygon and point
            x, y = ar.polygon.exterior.xy
            point_x, point_y = point

            # Create a figure and axis
            fig, ax = plt.subplots()

            # Plot the polygon
            ax.fill(x, y, facecolor='lightblue', edgecolor='blue', linewidth=2, label='Polygon')

            # Plot the point
            ax.plot(point_x, point_y, marker='o', color='red', markersize=5, label='Point')

            # Set axis limits
            ax.set_xlim(0, 4)
            ax.set_ylim(0, 4)

            # Add labels and legend
            plt.text(point_x + 0.1, point_y + 0.1, 'Point', color='red')
            plt.legend()

            # Show the plot
            plt.gca().set_aspect('equal', adjustable='box')
            plt.grid()
            plt.show()"""
            if self.is_point_inside_polygon(point, ar.polygon):
                poly_x, poly_y = ar.polygon.exterior.coords.xy
                x, y, z = poly_x[0], poly_y[0], skeleton.values[poly_x[0], poly_y[0]]
                value = (point[0] - x) * gradient[0] + (point[1] - y) * gradient[1] + z
                return value
        raise Exception

    def update_values(self, intersection: Polygon, skeleton2: "Skeleton", new_skeleton: "Skeleton"):
        xx, yy = intersection.exterior.coords.xy
        gradient = new_skeleton.linear_regions[-1].gradient
        v = 0
        for i in range(1, len(xx)):
            p = (xx[i], yy[i])
            if p not in new_skeleton.values:
                if p in self.values:
                    v += self.values[p]
                else:
                    v += self.calculate_value_of_point_inside_skeleton(p, self, gradient)
                if p in skeleton2.values:
                    v += skeleton2.values[p]
                else:
                    v += self.calculate_value_of_point_inside_skeleton(p, skeleton2, gradient)
                new_skeleton.values[p] = v

    def add_skeleton(self, skeleton2: "Skeleton", global_point_bank: Dict[Tuple[float, float], int], index: float) \
            -> "Skeleton":
        """ Add skeleton (passed as an argument) to self (without altering either of them)

        @param skeleton2: skeleton that will be added to self
        @param global_point_bank: used to monitor whether a point was not created before (sometimes the same point might
         be considered as two separate points due to the floating point error, checking against a point bank avoids it)
        @param index: used to indicate at which step was a critical point formed
        @return: skeleton formed after adding skeleton1 to self
        """
        new_skeleton = Skeleton([], self.hyperrectangle, {})  # represents the sum of two skeletons
        print('aaa')
        for lr in self.linear_regions:
            for lr1 in skeleton2.linear_regions:
                # go through all linear regions from skeleton1 and self. For every pair find their intersection
                intersection = self.find_intersection(lr.polygon, lr1.polygon, global_point_bank, index)
                # proceed only if the intersection is non-empty and a polygon (area check discards buggy intersections)
                if intersection is not None:
                    gradient_of_intersection = [lr.gradient[i] + lr1.gradient[i] for i in range(2)]
                    new_skeleton.linear_regions.append(LinearRegion(intersection, gradient_of_intersection))
                    self.update_values(intersection, skeleton2, new_skeleton)
        new_skeleton.test_validity(global_point_bank)
        return new_skeleton

    @staticmethod
    def distance_to_segment(p: Tuple[float, float], seg_start: Tuple[float, float], seg_end: Tuple[float, float]):
        x, y = p
        x1, y1 = seg_start
        x2, y2 = seg_end
        # Calculate the distance between point (x, y) and line segment (x1, y1)-(x2, y2)
        dx = x2 - x1
        dy = y2 - y1
        if dx == dy == 0:  # The segment is a point
            return ((x - x1) ** 2 + (y - y1) ** 2) ** 0.5
        t = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)
        if t < 0:
            px, py = x1, y1
        elif t > 1:
            px, py = x2, y2
        else:
            px, py = x1 + t * dx, y1 + t * dy
        return ((x - px) ** 2 + (y - py) ** 2) ** 0.5

    def is_point_inside_polygon(self, point: Tuple[float, float], polygon: Polygon, epsilon=1e-15) -> bool:
        """We use the "crossing number" algorithm to find if point lies within polygon"""
        xx, yy = polygon.exterior.coords.xy
        x, y = point
        inside = False
        for i in range(len(xx) - 1):
            x1, y1 = xx[i], yy[i]
            x2, y2 = xx[i+1], yy[i+1]
            if y1 < y2:
                if y1 <= y < y2 and (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1) > 0:
                    inside = not inside
            elif y1 > y2:
                if y2 <= y < y1 and (x1 - x2) * (y - y2) - (x - x2) * (y1 - y2) > 0:
                    inside = not inside
            else:  # Horizontal edge
                if y == y1 and min(x1, x2) <= x <= max(x1, x2):
                    return True  # Point lies on an edge
            # Check if the point is within epsilon distance of the edge
            dist = self.distance_to_segment(point, (x1, y1), (x2, y2))
            # print(point, (x1, y1), (x2, y2), dist)
            if dist < epsilon:
                return True
        return inside

    @staticmethod
    def points_to_slope_intercept(point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        # Calculate the slope (a)
        a = (y2 - y1) / (x2 - x1) if x2 - x1 != 0 else float('inf')
        # Calculate the y-intercept (b)
        b = y1 - a * x1 if x2 - x1 != 0 else x1
        return a, b

    def find_line_intersection(self, p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float],
                               p4: Tuple[float, float], point_bank: Dict[Tuple[float, float], int], index: float) \
            -> Union[List[Tuple[float, float]], None]:
        a1, b1 = self.points_to_slope_intercept(p1, p2)
        a2, b2 = self.points_to_slope_intercept(p3, p4)
        segment1 = sorted([p1, p2])
        segment2 = sorted([p3, p4])
        # Check if they lie on the same line
        if a1 == a2:
            if b1 == b2:
                # Check if they intersect
                if not (segment1[1] < segment2[0] or segment2[1] < segment1[0]):
                    intersection_start = max(segment1[0], segment2[0])
                    intersection_end = min(segment1[1], segment2[1])
                    x1, y1 = self.find_the_closest_point_from_the_bank(point_bank,
                                                                       Point(intersection_start[0], intersection_start[1]),
                                                                       index, None)
                    x2, y2 = self.find_the_closest_point_from_the_bank(point_bank,
                                                                       Point(intersection_end[0], intersection_end[1]),
                                                                       index, None)
                    return [(x1, y1), (x2, y2)]
            return None
        else:
            if a1 == float('inf') or a2 == float('inf'):
                if a1 == float('inf'):
                    infinity_index = 0
                else:
                    infinity_index = 1
                x = [b1, b2][infinity_index]
                y = [a1, a2][(infinity_index + 1) % 2] * x + [b1, b2][(infinity_index + 1) % 2]
            else:
                x = (b2 - b1) / (a1 - a2)
                y = a1 * x + b1
            if segment1[0][0] <= x <= segment1[1][0] and segment2[0][0] <= x <= segment2[1][0]:
                if min(p1[1], p2[1]) <= y <= max(p1[1], p2[1]) and min(p3[1], p4[1]) <= y <= max(p3[1], p4[1]):
                    x, y = self.find_the_closest_point_from_the_bank(point_bank, Point(x, y), index, None)
                    return [(x, y)]
            return None

    def find_intersection(self, polygon1: Polygon, polygon2: Polygon, point_bank: Dict[Tuple[float, float], int],
                          index: float) -> Union[Polygon, None]:
        xx1, yy1 = polygon1.exterior.coords.xy
        xx2, yy2 = polygon2.exterior.coords.xy
        intersection_points = []
        edges_intersect = False
        for i in range(1, len(xx2)):
            p = (xx2[i], yy2[i])
            if self.is_point_inside_polygon(p, polygon1) and p not in intersection_points:
                intersection_points.append(p)
        for i1 in range(1, len(xx1)):
            prev_p1 = (xx1[i1-1], yy1[i1-1])
            p1 = (xx1[i1], yy1[i1])
            if self.is_point_inside_polygon(p1, polygon2) and p1 not in intersection_points:
                intersection_points.append(p1)  # If p is inside polygon2 it is part of the intersection
            # So is the case if p lies on the edge of polygon1 that intersects polygon2
            for i2 in range(1, len(xx2)):
                prev_p2 = (xx2[i2-1], yy2[i2-1])
                p2 = (xx2[i2], yy2[i2])
                edge_intersection = self.find_line_intersection(prev_p1, p1, prev_p2, p2, point_bank, index)
                if edge_intersection is not None:
                    if len(edge_intersection) == 1 and edge_intersection[0] not in intersection_points:
                        edges_intersect = True
                    for p in edge_intersection:
                        if p not in intersection_points:
                            intersection_points.append(p)
        if len(intersection_points) > 2:
            if edges_intersect:
                return Polygon(MultiPoint(intersection_points).convex_hull)
            else:
                intersection = Polygon(MultiPoint(intersection_points).convex_hull)
                xx, yy = intersection.exterior.coords.xy
                for i in range(len(xx) - 1):
                    p = (xx[i] + xx[i+1]) / 2, (yy[i] + yy[i+1]) / 2
                    if not (self.is_point_inside_polygon(p, polygon1) and self.is_point_inside_polygon(p, polygon2)):
                        return None
                return intersection
        return None

    @staticmethod
    def find_relu_intersection(start_point, end_point):
        # Check if the line segment is parallel to the z = 0 plane
        if start_point[2] == end_point[2]:
            return None  # No intersection, the line is parallel to the plane
        # Calculate the parameter "t" at which the line intersects the z = 0 plane
        t = -start_point[2] / (end_point[2] - start_point[2])
        # Calculate the intersection point
        x = start_point[0] + t * (end_point[0] - start_point[0])
        y = start_point[1] + t * (end_point[1] - start_point[1])
        return x, y

    def division_by_relu(self, prev_p, p, positive_points, negative_points, new_skeleton, point_bank, index):
        x, y = self.find_relu_intersection((*prev_p, self.values[prev_p]), (*p, self.values[p]))
        x, y = self.find_the_closest_point_from_the_bank(point_bank, Point(x, y), index, None)
        positive_points.append((x, y))
        negative_points.append((x, y))
        new_skeleton.values[(x, y)] = 0

    def relu(self, global_point_bank: Dict[Tuple[float, float], int], index: float) -> "Skeleton":
        new_skeleton = Skeleton([], self.hyperrectangle, copy.deepcopy(self.values))
        for activation_region in self.linear_regions:
            positive_points = []
            negative_points = []
            if activation_region.gradient != [0, 0]:
                xx, yy = activation_region.polygon.exterior.xy
                p = (xx[0], yy[0])
                if self.values[p] == 0:
                    positive = self.values[(xx[1], yy[1])] > 0  # non-zero gradient so next point has non-zero value
                else:
                    positive = self.values[p] > 0
                for i in range(1, len(xx)):  # iterate through all vertices of activation region
                    prev_p = (xx[i-1], yy[i-1])
                    p = (xx[i], yy[i])
                    if self.values[p] > 0:
                        if not positive:  # intersection with 0
                            positive = True
                            self.division_by_relu(prev_p, p, positive_points, negative_points, new_skeleton,
                                                  global_point_bank, index)
                        positive_points.append(p)
                    elif self.values[p] < 0:
                        if positive:  # intersection with 0
                            positive = False
                            self.division_by_relu(prev_p, p, positive_points, negative_points, new_skeleton,
                                                  global_point_bank, index)
                        new_skeleton.values[p] = 0  # ReLU changes negative vertices to neutral (0)
                        negative_points.append(p)
                # Append the activation regions created by ReLU to new_skeleton
                if len(positive_points) > 0:
                    new_skeleton.linear_regions.append(
                        LinearRegion(Polygon(positive_points), activation_region.gradient))
                if len(negative_points) > 0:
                    new_skeleton.linear_regions.append(LinearRegion(Polygon(negative_points), [0, 0]))
            else:
                xx, yy = activation_region.polygon.exterior.xy
                positive = self.values[(xx[0], yy[0])] > 0
                for i in range(len(xx)):
                    if not positive:
                        new_skeleton.values[(xx[i], yy[i])] = 0
                if positive:
                    new_skeleton.linear_regions.append(LinearRegion(
                        activation_region.polygon, activation_region.gradient))
                else:
                    new_skeleton.linear_regions.append(LinearRegion(activation_region.polygon, [0, 0]))
        new_skeleton.test_validity(global_point_bank)
        return new_skeleton

    def plot_skeleton(self, title, ax, mode=0, save=False, point_bank=None):
        # prepare graph
        for linear_region in self.linear_regions:
            linear_region.new_plot(ax, self.values, point_bank, mode=mode)
        if save:
            plt.savefig(title + '.pdf')

    def test_validity(self, point_bank=None, full_test=True, skeleton_to_test=None, error=1e-14):
        """ Test whether a skeleton covers the whole hyperrectangle, and if the linear regions do not overlap

        """
        if not skeleton_to_test:
            skeleton_to_test = self
        for ar in skeleton_to_test.linear_regions:
            xx, yy = ar.polygon.exterior.coords.xy
            for i in range(len(xx)):
                p = (xx[i], yy[i])
                if p not in skeleton_to_test.values:
                    raise Exception
                if point_bank is not None and p not in point_bank:
                    print(p, "is not in the point bank")
                    raise Exception
        linear_region_union = skeleton_to_test.linear_regions[0].polygon
        # Check if any two linear regions overlap (if their intersection is a Polygon)
        for lr in skeleton_to_test.linear_regions[1:]:
            """print('----------------------------------------------------------------')
            print(lr.polygon)
            print(linear_region_union)
            print(self.find_intersection(lr.polygon, linear_region_union, {}, 0))"""
            """if self.find_intersection(lr.polygon, linear_region_union, {}, 0) is not None:
                # Determine the common axis limits
                x_min = min(lr.polygon.bounds[0], linear_region_union.bounds[0], self.find_intersection(lr.polygon, linear_region_union, {}, 0).bounds[0])
                x_max = max(lr.polygon.bounds[2], linear_region_union.bounds[2], self.find_intersection(lr.polygon, linear_region_union, {}, 0).bounds[2])
                y_min = min(lr.polygon.bounds[1], linear_region_union.bounds[1], self.find_intersection(lr.polygon, linear_region_union, {}, 0).bounds[1])
                y_max = max(lr.polygon.bounds[3], linear_region_union.bounds[3], self.find_intersection(lr.polygon, linear_region_union, {}, 0).bounds[3])

                # Create separate figures and axes for each polygon
                fig1, ax1 = plt.subplots()
                fig2, ax2 = plt.subplots()
                fig3, ax3 = plt.subplots()

                # Set the common axis limits for all plots
                ax1.set_xlim(x_min, x_max)
                ax1.set_ylim(y_min, y_max)
                ax2.set_xlim(x_min, x_max)
                ax2.set_ylim(y_min, y_max)
                ax3.set_xlim(x_min, x_max)
                ax3.set_ylim(y_min, y_max)
                # Plot the first polygon
                ax1.fill(*lr.polygon.exterior.xy, color='red', alpha=0.5)
                ax1.set_aspect('equal')
                ax1.set_title('Polygon 1')

                # Plot the second polygon
                ax2.fill(*linear_region_union.exterior.xy, color='green', alpha=0.5)
                ax2.set_aspect('equal')
                ax2.set_title('Polygon 2')

                # Plot the third polygon
                ax3.fill(*self.find_intersection(lr.polygon, linear_region_union, {}, 0).exterior.xy, color='blue', alpha=0.5)
                ax3.set_aspect('equal')
                ax3.set_title('Polygon 3')

                # Show the plots
                plt.show()"""
            intersection = linear_region_union.intersection(lr.polygon)
            if intersection and intersection.area > error:
                if isinstance(intersection, (Polygon, MultiPolygon)):
                    if intersection.area > error:
                        print(intersection.area)
                        raise Exception
                elif isinstance(intersection, GeometryCollection):
                    for geom in intersection.geoms:
                        if isinstance(geom, (Polygon, MultiPolygon)):
                            if intersection.area > 1e-1:
                                print(intersection.area)
                                raise Exception
            linear_region_union = linear_region_union.union(lr.polygon)
        if full_test:
            # Check if linear_region_union covers the whole hyperrectangle
            R = Polygon(skeleton_to_test.hyperrectangle.convert_to_polygon())
            dif = R.difference(linear_region_union)

            # Create a figure and axis
            fig, ax = plt.subplots()

            # Plot each polygon in the list
            for ar in skeleton_to_test.linear_regions:
                polygon = ar.polygon
                x, y = polygon.exterior.xy
                ax.fill(x, y, facecolor='lightblue', edgecolor='blue', linewidth=2)

            # Set axis limits
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)

            # Show the plot
            plt.gca().set_aspect('equal', adjustable='box')
            plt.grid()
            plt.show()
            if dif:
                if dif.area > error:
                    print(dif)
                    print(dif.area)
                    raise Exception
