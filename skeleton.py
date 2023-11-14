import copy
import math
import numpy as np
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
                if -pow(10, -15) < self.values[key] < pow(10, -15):
                    self.values[key] = 0
            return self
        else:
            raise TypeError("Can only multiply by a float or int")

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            for key in self.values:
                self.values[key] *= other
                if -pow(10, -15) < self.values[key] < pow(10, -15):
                    self.values[key] = 0
            for linear_region in self.linear_regions:
                for index in range(len(linear_region.gradient)):
                    linear_region.gradient[index] *= other
                    if -pow(10, -15) < linear_region.gradient[index] < pow(10, -15):
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

    @staticmethod
    def find_closest_vertex(polygon: Polygon, point: Tuple[float, float]) -> int:
        xx, yy = polygon.exterior.coords.xy
        distances = [math.sqrt((point[0] - xx[i]) ** 2 + (point[1] - yy[i]) ** 2) for i in range(len(xx) - 1)]
        index_of_closest_point = min(range(len(distances)), key=lambda i: distances[i])
        return index_of_closest_point

    @staticmethod
    def calculate_value_of_point_inside_skeleton(point: Tuple[float, float], skeleton: "Skeleton") -> float:
        """ Given a point extract what value it has inside skeleton.

        :param point: Point which value we want to compute
        :param skeleton: We look for the value of the point within this skeleton
        :return: Value of point inside skeleton
        """
        for ar in skeleton.linear_regions:
            """print('-------------------------------------------------------------------------')
            print(ar.polygon)
            print(point)
            print(ar.polygon.contains(Point(point)))
            print(self.is_point_inside_polygon(point, ar.polygon))
            print(self.distance_to_segment(point, (-0.029835981856429, -1.5), (-1.2, 1.32745553881469)))
            # Extract coordinates from the polygon and point
            x, y = ar.polygon.exterior.xy
            point_x, point_y = point

            # Create a figure and axis
            fig, ax = plt.subplots()

            # Plot the polygon
            ax.fill(x, y, facecolor='lightblue', edgecolor='blue', linewidth=2, label='Polygon')

            # Plot the point
            ax.plot(point_x, point_y, marker='o', color='red', markersize=5, label='Point')

            # Set axis limits
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)

            # Add labels and legend
            plt.text(point_x + 0.1, point_y + 0.1, 'Point', color='red')
            plt.legend()

            # Show the plot
            plt.gca().set_aspect('equal', adjustable='box')
            plt.grid()
            plt.show()"""
            if skeleton.is_point_inside_polygon(point, ar.polygon):
                """if not ar.polygon.contains(Point(point)) and ar.polygon.distance(Point(point)) > 1e-15:
                    fig, ax = plt.subplots()
                    x, y = ar.polygon.exterior.xy
                    ax.plot(x, y, color='blue', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
                    ax.scatter(*point, color='red', s=50, zorder=3)
                    plt.show()
                    raise Exception"""
                i = skeleton.find_closest_vertex(ar.polygon, point)  # we do this to reduce the floating point error
                poly_x, poly_y = ar.polygon.exterior.coords.xy
                x, y, z = poly_x[i], poly_y[i], skeleton.values[poly_x[i], poly_y[i]]
                value = (point[0] - x) * ar.gradient[0] + (point[1] - y) * ar.gradient[1] + z
                return value
            """elif ar.polygon.contains(Point(point)) or ar.polygon.distance(Point(point)) < 1e-14:
                print(ar.polygon.distance(Point(point)))
                fig, ax = plt.subplots()
                x, y = ar.polygon.exterior.xy
                ax.plot(x, y, color='blue', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
                ax.scatter(*point, color='red', s=50, zorder=3)
                plt.show()"""
        fig, ax = plt.subplots()
        for ar in skeleton.linear_regions:
            x, y = ar.polygon.exterior.xy
            ax.plot(x, y, color='blue', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
        ax.scatter(*point, color='red', s=50, zorder=3)
        plt.show()
        raise Exception  # activation regions tessellate the input space so the if statement should always be executed

    def update_values(self, intersection: Polygon, skeleton2: "Skeleton", new_skeleton: "Skeleton"):
        """ Iterate vertices of intersection and calculate their values given the information on skeleton2 and self

        :param intersection: Polygon representing the intersection of activation regions
        :param skeleton2: Other Skeleton used in summation
        :param new_skeleton: Current result of summing self with skeleton2
        """
        xx, yy = intersection.exterior.coords.xy
        for i in range(1, len(xx)):
            v = 0
            p = (xx[i], yy[i])
            if p in self.values:
                v += self.values[p]
            else:
                v += self.calculate_value_of_point_inside_skeleton(p, self)
            if p in skeleton2.values:
                v += skeleton2.values[p]
            else:
                v += skeleton2.calculate_value_of_point_inside_skeleton(p, skeleton2)
            if p not in new_skeleton.values:
                new_skeleton.values[p] = v
            elif abs(new_skeleton.values[p] - v) > 1e-15:
                print('ERROR', abs(new_skeleton.values[p] - v))
                if p in self.values:
                    print(True, self.values[p])
                else:
                    print(False)
                if p in skeleton2.values:
                    print(True, skeleton2.values[p])
                else:
                    print(False)

    @staticmethod
    def points_to_slope_intercept(point1, point2):
        """ Given two points that lie on the same line extract the a & b from y = ax + b

        :param point1: (x1, y1) lying on the line
        :param point2: (x2, y2) lying on the line
        :return: a & b from y = ax + b
        """
        x1, y1 = point1
        x2, y2 = point2
        # Calculate the slope (a)
        a = (y2 - y1) / (x2 - x1) if x2 - x1 != 0 else float('inf')
        # Calculate the y-intercept (b)
        b = y1 - a * x1 if x2 - x1 != 0 else x1
        return a, b

    def find_line_segment_intersection(self, p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float],
                                       p4: Tuple[float, float], point_bank: Dict[Tuple[float, float], int],
                                       index: float, epsilon=1e-15) -> Union[List[Tuple[float, float]], None]:
        """ Given two line segments defined using their endpoints finds their intersection.

        :param p1: start of the first line segment
        :param p2: end of the first line segment
        :param p3: start of the first second segment
        :param p4: end of the first second segment
        :param point_bank: used to store all vertices generated during SkelEx (to quantize)
        :param index: used to store the time of creation
        :param epsilon: used to remedy the floating point imprecision
        :return:
        """
        a1, b1 = self.points_to_slope_intercept(p1, p2)
        a2, b2 = self.points_to_slope_intercept(p3, p4)
        if abs(a1 - a2) < epsilon:  # check if the line segments are parallel
            if abs(b1 - b2) < epsilon:  # check if they are collinear
                # We check both conditions below, as 'a' can be 0 (1st condition holds) or inf (2nd condition holds)
                if (min(p3[0], p4[0]) - epsilon <= p1[0] <= max(p3[0], p4[0]) + epsilon or
                    min(p3[0], p4[0]) - epsilon <= p2[0] <= max(p3[0], p4[0]) + epsilon) and \
                        min(p3[1], p4[1]) - epsilon <= p1[1] <= max(p3[1], p4[1]) + epsilon or \
                        min(p3[1], p4[1]) - epsilon <= p2[1] <= max(p3[1], p4[1]) + epsilon:
                    # find endpoints of intersection (line segment)
                    x_start = max(min(p1[0], p2[0]), min(p3[0], p4[0]))
                    x_end = min(max(p1[0], p2[0]), max(p3[0], p4[0]))
                    y_start = max(min(p1[1], p2[1]), min(p3[1], p4[1]))
                    y_end = min(max(p1[1], p2[1]), max(p3[1], p4[1]))
                    # Quantize both points
                    x1, y1 = self.find_the_closest_point_from_the_bank(point_bank, Point(x_start, y_start), index,
                                                                       None)
                    x2, y2 = self.find_the_closest_point_from_the_bank(point_bank, Point(x_end, y_end), index, None)
                    return [(x1, y1), (x2, y2)]
        else:
            if a1 == float('inf') or a2 == float('inf'):  # formula slightly differs for vertical line segments
                if a1 == float('inf'):
                    infinity_index = 0
                else:
                    infinity_index = 1
                x = [b1, b2][infinity_index]  # taken from the vertical line
                y = [a1, a2][(infinity_index + 1) % 2] * x + [b1, b2][(infinity_index + 1) % 2]
            else:
                x = (b2 - b1) / (a1 - a2)
                y = a1 * x + b1
            # Check if (x, y) lies on both lines segments (p1, p2) and (p3, p4)
            if (min(p1[0], p2[0]) - epsilon <= x <= max(p1[0], p2[0]) + epsilon and
                min(p3[0], p4[0]) - epsilon <= x <= max(p3[0], p4[0]) + epsilon) and \
                    (min(p1[1], p2[1]) - epsilon <= y <= max(p1[1], p2[1]) + epsilon and
                     min(p3[1], p4[1]) - epsilon <= y <= max(p3[1], p4[1]) + epsilon):
                # Quantize the point of intersection
                x, y = self.find_the_closest_point_from_the_bank(point_bank, Point(x, y), index, None)
                return [(x, y)]
        return None

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

    def is_point_inside_polygon(self, point: Tuple[float, float], polygon: Polygon, epsilon=1e-16) -> bool:
        """We use the "crossing number" algorithm to find if point lies within polygon"""
        ray = (point[0] + 1e10, point[1])
        ray_intersections = 0
        xx, yy = polygon.exterior.coords.xy
        for i in range(len(xx) - 1):
            p = xx[i], yy[i]
            next_p = xx[i+1], yy[i+1]
            # Check if the point is within epsilon distance of the edge
            if self.distance_to_segment(point, p, next_p) <= epsilon:
                return True
            current_intersections = self.find_line_segment_intersection(point, ray, p, next_p, {}, 0)
            if current_intersections is not None:
                ray_intersections += len(current_intersections)
            """if dist < epsilon and (point[0] < self.hyperrectangle.x[0] or self.hyperrectangle.x[1] < point[0] or
                                   point[1] < self.hyperrectangle.x[0] or self.hyperrectangle.y[1] < point[1]):"""
        return ray_intersections % 2 == 1

    def find_intersection(self, polygon1: Polygon, polygon2: Polygon, point_bank: Dict[Tuple[float, float], int],
                          index: float) -> Union[Polygon, None]:
        xx1, yy1 = polygon1.exterior.coords.xy
        xx2, yy2 = polygon2.exterior.coords.xy
        intersection_points = []
        # edges_intersect = False
        for i in range(1, len(xx2)):
            # If a point from polygon2 is inside polygon1 then it is part of their intersection
            p = (xx2[i], yy2[i])
            if self.is_point_inside_polygon(p, polygon1) and p not in intersection_points:
                intersection_points.append(p)
        for i1 in range(1, len(xx1)):
            prev_p1 = (xx1[i1-1], yy1[i1-1])
            p1 = (xx1[i1], yy1[i1])
            # If a point P from polygon1 is inside polygon2 then it is part of their intersection ...
            if self.is_point_inside_polygon(p1, polygon2) and p1 not in intersection_points:
                intersection_points.append(p1)
            # ... So is the case if P lies on the edge (prev_p1, p1) that intersects any edge of polygon2
            for i2 in range(1, len(xx2)):
                prev_p2 = (xx2[i2-1], yy2[i2-1])
                p2 = (xx2[i2], yy2[i2])
                edge_intersection = self.find_line_segment_intersection(prev_p1, p1, prev_p2, p2, point_bank, index)
                if edge_intersection is not None:  # If len == 0 then there are no intersection
                    for p in edge_intersection:
                        if p not in intersection_points:  # make sure the point is not a vertex of the polygons
                            # edges_intersect = True
                            intersection_points.append(p)
        if len(intersection_points) > 2:
            """if edges_intersect:
                # This is definitely a valid intersection as the edges of polygons cross (non-parallel crossings)
                return Polygon(MultiPoint(intersection_points).convex_hull)
            else:
                intersection = Polygon(MultiPoint(intersection_points).convex_hull)
                xx, yy = intersection.exterior.coords.xy
                center = [0.0, 0.0]
                for i in range(len(xx) - 1):
                    center[0] += xx[i]
                    center[1] += yy[i]
                center[0] /= (len(xx) - 1)
                center[1] /= (len(yy) - 1)
                if self.is_point_inside_polygon(tuple(center), polygon1, epsilon=0) and \
                        self.is_point_inside_polygon(tuple(center), polygon2, epsilon=0):
                    return intersection"""
            return Polygon(MultiPoint(intersection_points).convex_hull)
        return None

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
        """if index >= 1.5:
            fig, ax = plt.subplots()

            # Set axis limits
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.grid()"""
        self.test_validity(global_point_bank)
        skeleton2.test_validity(global_point_bank)
        for lr in self.linear_regions:
            for lr1 in skeleton2.linear_regions:
                # go through all linear regions from skeleton1 and self. For every pair find their intersection
                intersection = self.find_intersection(lr.polygon, lr1.polygon, global_point_bank, index)
                """# Plot the first polygon
                plt.figure(1)
                x1, y1 = lr.polygon.exterior.xy
                plt.fill(x1, y1, facecolor='lightblue', edgecolor='blue', linewidth=2)
                plt.title('Polygon 1')
                plt.gca().set_aspect('equal', adjustable='box')
                plt.grid()
                plt.xlim(-2, 2)
                plt.ylim(-2, 2)

                # Plot the second polygon
                plt.figure(2)
                x2, y2 = lr1.polygon.exterior.xy
                plt.fill(x2, y2, facecolor='lightgreen', edgecolor='green', linewidth=2)
                plt.title('Polygon 2')
                plt.gca().set_aspect('equal', adjustable='box')
                plt.grid()
                plt.xlim(-2, 2)
                plt.ylim(-2, 2)

                if intersection is not None:
                    # Plot the third polygon
                    plt.figure(3)
                    x3, y3 = intersection.exterior.xy
                    plt.fill(x3, y3, facecolor='lightcoral', edgecolor='red', linewidth=2)
                    plt.title('Polygon 3')
                    plt.gca().set_aspect('equal', adjustable='box')
                    plt.grid()
                    plt.xlim(-2, 2)
                    plt.ylim(-2, 2)
                else:
                    print(None)

                # Show all three figures
                plt.show()"""
                # proceed only if the intersection is non-empty and a polygon (area check discards buggy intersections)
                if intersection is not None:
                    """if index >= 1.5:
                        print(intersection)
                        # Plot each polygon in the list
                        x, y = intersection.exterior.xy
                        ax.fill(x, y, facecolor='lightblue', edgecolor='blue', linewidth=2)
                        plt.pause(4)  # Pause for 10 seconds"""
                    gradient_of_intersection = [lr.gradient[i] + lr1.gradient[i] for i in range(2)]
                    new_skeleton.linear_regions.append(LinearRegion(intersection, gradient_of_intersection))
                    self.update_values(intersection, skeleton2, new_skeleton)
        # Create a figure and axis
        """print(index)
        if index >= 1.5:
            fig1, ax1 = plt.subplots()
            fig2, ax2 = plt.subplots()
            fig3, ax3 = plt.subplots()
            figs = [fig1, fig2, fig3]
            axs = [ax1, ax2, ax3]
            i = 0
            for skeleton_to_test in (self, skeleton2, new_skeleton):
                # Plot each polygon in the list
                for ar in skeleton_to_test.linear_regions:
                    polygon = ar.polygon
                    x, y = polygon.exterior.xy
                    axs[i].fill(x, y, facecolor='lightblue', edgecolor='blue', linewidth=2)

                # Set axis limits
                axs[1].set_xlim(-2, 2)
                axs[1].set_ylim(-2, 2)

                # Show the plot
                plt.gca().set_aspect('equal', adjustable='box')
                plt.grid()
                i += 1
            print(len(self.linear_regions))
            print(len(skeleton2.linear_regions))
            print(len(new_skeleton.linear_regions))
            plt.show()"""
        """print('---------------------------------------------------------------------------------')
        for skeleton in [self, skeleton2, new_skeleton]:
            for lr in skeleton.linear_regions:
                print('gradient - ', lr.gradient)
                xx, yy = lr.polygon.exterior.coords.xy
                for i in range(len(xx)):
                    print(xx[i], yy[i], skeleton.values[(xx[i], yy[i])])
            print()
            print('MOVING TO THE NEXT SKELETON')
            print()
        print('---------------------------------------------------------------------------------')
        fig, ax = plt.subplots()
        for lr in new_skeleton.linear_regions:
            xx, yy = lr.polygon.exterior.coords.xy
            ax.set_aspect('equal')
            ax.set_title('Shapely Polygon')
            ax.plot(xx, yy)
        plt.show()"""
        new_skeleton.test_validity(global_point_bank)
        return new_skeleton

    @staticmethod
    def find_relu_intersection(start_point: Tuple[float, float, float], end_point: Tuple[float, float, float]) \
            -> Union[Tuple[float, float]]:
        """Find an intersection of a line segment in 3D subspace with the x-y axis. We can be sure that the line segment
        is neither parallel nor perpendicular to the x-y axis.

        :param start_point: Tuple of coordinates of the start of the line segment
        :param end_point:  Tuple of coordinates of the end of the line segment
        :return: Tuple of coordinates of the intersection of the line segment with the x-y axis
        """
        # Calculate the parameter "t" at which the line intersects the z = 0 plane
        t = -start_point[2] / (end_point[2] - start_point[2])
        if t < 1e-10 or 1-t < 1e-10:
            raise Exception
        # Calculate the intersection point
        x = start_point[0] + t * (end_point[0] - start_point[0])
        y = start_point[1] + t * (end_point[1] - start_point[1])
        return x, y

    def division_by_relu(self, prev_p: Tuple[float, float], p: Tuple[float, float],
                         positive_points: List[Tuple[float, float]], negative_points: List[Tuple[float, float]],
                         new_skeleton: "Skeleton", point_bank: Dict[Tuple[float, float], int], index: float):
        """Given an edge that crosses 0 (defined using its end points) finds the coordinates of intersection with 0.
        Next it quantifies that point and stores it updates positive_points, negative_points new_skeleton and point_bank

        :param prev_p: point at the start of the edge that crosses 0
        :param p: point at the end of the edge that crosses 0
        :param positive_points: list of points above 0 found so far
        :param negative_points: list of points below 0 found so far
        :param new_skeleton: Skeleton storing the activation regions generated by applying ReLU to self
        :param point_bank: dictionary storing the points created so far (Quantification of points)
        :param index: float used to keep track at which stage the critical points were formed
        """
        x, y = self.find_relu_intersection((prev_p[0], prev_p[1], self.values[prev_p]), (p[0], p[1], self.values[p]))
        x, y = self.find_the_closest_point_from_the_bank(point_bank, Point(x, y), index, None)
        positive_points.append((x, y))
        negative_points.append((x, y))
        new_skeleton.values[(x, y)] = 0

    def relu(self, global_point_bank: Dict[Tuple[float, float], int], index: float) -> "Skeleton":
        """Applies ReLU to all activation regions in self

        :param global_point_bank: dictionary storing the points created so far (Quantification of points)
        :param index: float used to keep track at which stage the critical points were formed
        :return: Skeleton object that is the result of applying ReLU to self
        """
        new_skeleton = Skeleton([], self.hyperrectangle, copy.deepcopy(self.values))
        for activation_region in self.linear_regions:
            positive_points, negative_points = [], []
            if activation_region.gradient != [0, 0]:
                xx, yy = activation_region.polygon.exterior.xy
                p = (xx[0], yy[0])
                if self.values[p] == 0:
                    positive = self.values[(xx[1], yy[1])] > 0  # non-zero gradient so next point has non-zero value
                    start_index = 2
                else:
                    positive = self.values[p] > 0
                    start_index = 1
                for i in range(start_index, len(xx)):  # iterate through all vertices of activation region
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
                    else:  # rare edge case
                        positive_points.append(p)
                        negative_points.append(p)
                # Append the activation regions created by ReLU to new_skeleton
                """wrong_polygon = Polygon(
                    [(-1.144049954516422, - 0.270724221068919), (-1.2, - 0.258535862164962), (-1.2, - 0.10614054589095),
                     (-1.111173355494278, 0.074840416927113), (-0.946840967808732, 0.217677869711059),
                     (-0.946650498110091, 0.217682768829026), (-1.144049954516422, -0.270724221068919)])
                if Polygon(positive_points).equals(wrong_polygon):
                    print('pos', activation_region.polygon)
                elif Polygon(negative_points).equals(wrong_polygon):
                    print('neg', activation_region.polygon)
                    for polygon in [Polygon(positive_points), Polygon(negative_points)]:
                        fig, ax = plt.subplots()
                        ax.set_aspect('equal', 'box')
                        x, y = polygon.exterior.xy
                        ax.plot(x, y)
                        x, y = activation_region.polygon.exterior.xy
                        ax.plot(x, y)
                    print(self.values)
                    plt.show()"""
                if len(positive_points) > 0:
                    new_skeleton.linear_regions.append(
                        LinearRegion(Polygon(MultiPoint(positive_points).convex_hull), activation_region.gradient))
                if len(negative_points) > 0:
                    new_skeleton.linear_regions.append(LinearRegion(Polygon(MultiPoint(negative_points).convex_hull),
                                                                    [0, 0]))
            else:  # rare edge case
                xx, yy = activation_region.polygon.exterior.xy
                positive = self.values[(xx[0], yy[0])] > 0
                if not positive:
                    for i in range(len(xx)):
                        new_skeleton.values[(xx[i], yy[i])] = 0
                    new_skeleton.linear_regions.append(LinearRegion(activation_region.polygon, [0, 0]))
                else:
                    new_skeleton.linear_regions.append(LinearRegion(
                        activation_region.polygon, activation_region.gradient))
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
        xx, yy = skeleton_to_test.linear_regions[0].polygon.exterior.coords.xy
        gradient = skeleton_to_test.linear_regions[0].gradient
        for i in range(1, len(xx)):
            if abs(skeleton_to_test.values[(xx[i], yy[i])] - (skeleton_to_test.values[(xx[i - 1], yy[i - 1])] +
                   (xx[i] - xx[i - 1]) * gradient[0] + (yy[i] - yy[i - 1]) * gradient[1])) > 5e-15:
                print("-----------------------------------------------------")
                print(i, skeleton_to_test.values[(xx[i], yy[i])], skeleton_to_test.values[(xx[i - 1], yy[i - 1])] +
                      (xx[i] - xx[i - 1]) * gradient[0] + (yy[i] - yy[i - 1]) * gradient[1])
                print(skeleton_to_test.linear_regions[0].gradient)
                for i1 in range(len(xx)):
                    print(xx[i1], yy[i1], skeleton_to_test.values[(xx[i1], yy[i1])])
                # Plot the Shapely polygon
                fig, ax = plt.subplots()
                ax.set_aspect('equal')
                ax.set_title('Shapely Polygon')
                ax.plot(xx, yy)
                plt.show()
                raise Exception
        # Check if any two linear regions overlap (if their intersection is a Polygon)
        test_index = 1
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
            xx, yy = lr.polygon.exterior.coords.xy
            gradient = lr.gradient
            for i in range(1, len(xx)):
                if abs(skeleton_to_test.values[(xx[i], yy[i])] - (skeleton_to_test.values[(xx[i-1], yy[i-1])] +
                       (xx[i] - xx[i-1]) * gradient[0] + (yy[i] - yy[i-1]) * gradient[1])) > 1e-14:
                    print("-----------------------------------------------------")
                    print(i, skeleton_to_test.values[(xx[i], yy[i])], skeleton_to_test.values[(xx[i-1], yy[i-1])] +
                          (xx[i] - xx[i-1]) * gradient[0] + (yy[i] - yy[i-1]) * gradient[1])
                    print(lr.gradient)
                    for i1 in range(len(xx)):
                        print(xx[i1], yy[i1], skeleton_to_test.values[(xx[i1], yy[i1])])
                    # Plot the Shapely polygon
                    fig, ax = plt.subplots()
                    ax.set_aspect('equal')
                    ax.set_title('Shapely Polygon')
                    ax.plot(xx, yy)
                    plt.show()
                    raise Exception
            test_index += 1
            intersection = linear_region_union.intersection(lr.polygon)
            if intersection and intersection.area > error:
                if isinstance(intersection, (Polygon, MultiPolygon)):
                    if intersection.area > error:
                        print(intersection.area)
                        raise Exception
                elif isinstance(intersection, GeometryCollection):
                    for geom in intersection.geoms:
                        if isinstance(geom, (Polygon, MultiPolygon)):
                            if intersection.area > error:
                                print(intersection.area)
                                colors = []  # To store colors for each polygon

                                # Create a figure and axis
                                fig, ax = plt.subplots(1, figsize=(8, 8))
                                polygons = [lr.polygon for lr in skeleton_to_test.linear_regions]
                                for i, polygon in enumerate(polygons):
                                    # Check for intersection with previously added polygons
                                    intersect = False
                                    for j in range(i):
                                        intersection = polygon.intersection(polygons[j])
                                        if intersection is not None and intersection.area > 1e-14:
                                            intersect = True
                                            break

                                    # Assign 'red' if there's an intersection, otherwise use 'blue'
                                    color = 'red' if intersect else 'blue'
                                    colors.append(color)

                                    # Extract the coordinates from the Shapely polygon
                                    x, y = polygon.exterior.xy

                                    # Plot the polygon with the assigned color
                                    ax.fill(x, y, color=color, alpha=0.5)

                                    # Set axis limits
                                    ax.set_xlim(-2, 2)
                                    ax.set_ylim(-2, 2)

                                    # Show the plot
                                    print(polygon)
                                    if color == 'red':
                                        print('-----------------------------------------')
                                    plt.draw()
                                    plt.pause(0.25)

                                # Close the plot window after the last polygon
                                plt.show()
                                raise Exception
            linear_region_union = linear_region_union.union(lr.polygon)
        if full_test:
            # Check if linear_region_union covers the whole hyperrectangle
            R = Polygon(skeleton_to_test.hyperrectangle.convert_to_polygon())
            dif = R.difference(linear_region_union)

            """# Create a figure and axis
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
            plt.show()"""
            if dif:
                if dif.area > error:
                    print(dif)
                    print(dif.area)
                    raise Exception
