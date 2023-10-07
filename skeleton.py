import copy
import math
from typing import Tuple, List, Dict, Union, Any

import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.geometry import LineString, MultiLineString, Point, GeometryCollection, MultiPolygon
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
    def quantize_to_0(variable, error=1e-5):
        if -error < variable < error:
            return 0
        return variable

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
        # p is not in point_bank (it's a new, just created point) so add it there
        point_bank[(xx[0], yy[0])] = index
        # add to values (this critical point was made during applying ReLU)
        if values and (xx[0], yy[0]) not in values:
            values[(xx[0], yy[0])] = 0
        return xx[0], yy[0]

    def calculate_new_value_of_point_given_polygon(self, polygon: Polygon, point: Point, gradient: List[float],
                                                   v: Dict[Tuple[float, float], float]) -> \
            Tuple[Tuple[float, float], float]:
        xx, yy = polygon.exterior.xy
        p1 = Point(xx[0], yy[0])
        min_len = LineString([point, p1]).length
        closest_point_index = 0
        # find the point on the exterior that is the closest to the parameter point
        for i in range(1, len(xx) - 1):
            p1 = Point(xx[i], yy[i])
            l = LineString([point, p1]).length
            if l < min_len:
                min_len = l
                closest_point_index = i
        x, y = point.coords.xy
        value = self.quantize_to_0(
            (x[0] - xx[closest_point_index]) * gradient[0] + (y[0] - yy[closest_point_index]) * gradient[1] + \
            v[(xx[closest_point_index], yy[closest_point_index])])
        return (x[0], y[0]), value

    def calculate_starting_value(self, starting_point: Tuple[float, float], starting_value: float,
                                 v1: Dict[Tuple[float, float], float], skeleton: "Skeleton") -> \
            Tuple[Tuple[float, float], float]:
        if starting_point in v1:
            v = self.quantize_to_0(starting_value + v1[starting_point])
            return starting_point, v
        else:
            p = Point(starting_point[0], starting_point[1])
            # find the linear region that contains the starting_point
            for linear_region in skeleton.linear_regions:
                # contains -> p lies inside             touches -> p lies on the exterior
                if linear_region.polygon.contains(p) or linear_region.polygon.touches(p) or \
                        linear_region.polygon.distance(p) < 1e-10:
                    p, v = self.calculate_new_value_of_point_given_polygon(linear_region.polygon, p,
                                                                           linear_region.gradient, v1)
                    v = self.quantize_to_0(v + starting_value)
                    return p, v
        raise NotImplementedError('This edge case was not implemented yet')  # just in case I missed an edge case

    def quantize_intersection_terior(self, skeleton1: "Skeleton", gradient: List[float],
                                     point_bank: Dict[Tuple[float, float], float], index: float, xx: List[float],
                                     yy: List[float]) -> Tuple[Dict[Tuple[float, float], Union[float, Any]], Polygon]:
        intersection_values = {}
        new_intersection_points = []
        v1 = skeleton1.values
        start_p = self.find_the_closest_point_from_the_bank(point_bank, Point(xx[0], yy[0]), index, None)
        # Calculate the value of this point (value on skeleton1 + value on self)
        _, start_v = self.calculate_starting_value(start_p, 0, v1, skeleton1)
        start_p, start_v = self.calculate_starting_value(start_p, start_v, self.values, self)
        # Update variables
        new_intersection_points.append(start_p)
        intersection_values[start_p] = start_v
        last_output_value = start_v
        previous_p = start_p
        for i in range(1, len(xx) - 1):
            # Go along the exterior and calculate the value of each point using gradient, and value of previous point
            x, y = self.find_the_closest_point_from_the_bank(point_bank, Point(xx[i], yy[i]), index, None)
            value = self.quantize_to_0(
                (x - previous_p[0]) * gradient[0] + (y - previous_p[1]) * gradient[1] + last_output_value)
            new_intersection_points.append((x, y))
            intersection_values[(x, y)] = value
            last_output_value = value
            previous_p = (x, y)
        # Remove the unnecessary points - ones that do not lie on the edges of the line segments
        new_intersection = self.remove_collinear_points(Polygon(new_intersection_points), False)
        return intersection_values, new_intersection

    def add_intersection_to_skeleton(self, gradient: List[float], intersection: Polygon, skeleton1: "Skeleton",
                                     global_point_bank: Dict[Tuple[float, float], float], index: float,
                                     new_skeleton: "Skeleton", error=1e-5) -> "Skeleton":
        xx, yy = intersection.exterior.coords.xy
        values, shell = self.quantize_intersection_terior(skeleton1, gradient,
                                                          global_point_bank, index, xx, yy)
        holes = []
        if intersection.interiors:
            for geom in intersection.interiors:
                xx, yy = geom.coords.xy
                values1, h = self.quantize_intersection_terior(skeleton1, gradient,
                                                               global_point_bank, index, xx, yy)
                holes.append(h)
        if abs(Polygon(shell, holes).area - intersection.area) > 1e-10:
            print(intersection)
            print(Polygon(shell, holes))
            print(Polygon(shell, holes).difference(intersection).area)
            raise Exception
        new_skeleton.linear_regions.append(LinearRegion(Polygon(shell, holes), gradient))
        # Update values of the new_skeleton
        for key in values:  # TODO: go through values1 as well
            if key not in new_skeleton.values:
                new_skeleton.values[key] = values[key]
            elif abs(new_skeleton.values[key] - values[key]) > error:
                # If this shows, then the values of the critical points must have been contaminated
                print('\n     !!!ERROR FOR ' + str(key) + '!!!')
                print(new_skeleton.values[key] - values[key])
        return new_skeleton

    def add_skeleton(self, skeleton1: "Skeleton", global_point_bank: Dict[Tuple[float, float], int], index: float,
                     error=1e-5) -> "Skeleton":
        """ Add skeleton (passed as an argument) to self (without altering either of them)

        @param skeleton1: skeleton that will be added to self
        @param global_point_bank: used to monitor whether a point was not created before (sometimes the same point might
         be considered as two separate points due to the floating point error, checking against a point bank avoids it)
        @param index: used to indicate at which step was a critical point formed
        @param error: rounding error (used to tackle the Floating Point Error)
        @return: skeleton formed after adding skeleton1 to self
        """
        new_skeleton = Skeleton([], self.hyperrectangle, {})  # represents the sum of two skeletons
        for lr in self.linear_regions:
            for lr1 in skeleton1.linear_regions:
                # go through all linear regions from skeleton1 and self. For every pair find their intersection
                intersection = lr.polygon.intersection(lr1.polygon)
                # proceed only if the intersection is non-empty and a polygon (area check discards buggy intersections)
                if intersection and intersection.area > error:
                    gradient_of_intersection = [self.quantize_to_0(lr.gradient[i] + lr1.gradient[i]) for i in range(2)]
                    if isinstance(intersection, Polygon):
                        new_skeleton = self.add_intersection_to_skeleton(gradient_of_intersection, intersection,
                                                                         skeleton1,
                                                                         global_point_bank, index, new_skeleton)
                    elif isinstance(intersection, (GeometryCollection, MultiPolygon)):
                        for geom in intersection.geoms:
                            if isinstance(geom, Polygon) and geom.area > error:
                                new_skeleton = self.add_intersection_to_skeleton(gradient_of_intersection, geom,
                                                                                 skeleton1,
                                                                                 global_point_bank, index, new_skeleton)
        new_skeleton.test_validity()
        return new_skeleton

    def quantize_polygon(self, polygon: Polygon, point_bank: Dict[Tuple[float, float], int], index: float,
                         values: Dict[Tuple[float, float], float]) -> Polygon:
        xx, yy = polygon.exterior.xy
        exterior = []
        for i in range(len(xx)):
            p = Point(xx[i], yy[i])
            p = self.find_the_closest_point_from_the_bank(point_bank, p, index, values)
            exterior.append(p)
        if polygon.interiors:
            raise NotImplementedError
        return Polygon(exterior)

    @staticmethod
    def magnitude(v):
        return math.sqrt(v[0] * v[0] + v[1] * v[1])

    @staticmethod
    def is_intersection_proper_line(lr1: Polygon, lr2: Polygon):
        if lr1.touches(lr2) or lr1.intersects(lr2) or lr1.intersection(lr2):
            if isinstance(lr1.intersection(lr2), GeometryCollection):
                for geom in lr1.intersection(lr2).geoms:
                    if isinstance(geom, (LineString, MultiLineString, Polygon, MultiPolygon)):
                        return True
            elif isinstance(lr1.intersection(lr2), (LineString, MultiLineString, Polygon, MultiPolygon)):
                return True
        # We include Polygons as intersection options, as due to a bug sometimes polygons are returned instead of lines
        return False

    def assign_point_based_on_value(self, p, positive_points, negative_points, positive):
        if self.values[p] > 0:
            positive = True
            positive_points.append(p)
        elif self.values[p] < 0:
            positive = False
            negative_points.append(p)

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
        z = 0  # Since it intersects the z = 0 plane
        return x, y, z

    def division_by_relu(self, prev_p, p, positive_points, negative_points, new_skeleton, point_bank, index):
        x, y, z = self.find_relu_intersection((*prev_p, self.values[prev_p]), (*p, self.values[p]))
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
                    new_skeleton.linear_regions.append(LinearRegion(activation_region, activation_region.gradient))
                else:
                    new_skeleton.linear_regions.append(LinearRegion(activation_region, [0, 0]))
        new_skeleton.test_validity()
        return new_skeleton

    def new_relu(self, global_point_bank: Dict[Tuple[float, float], int], index: float) -> "Skeleton":
        new_skeleton = Skeleton([], self.hyperrectangle, {})
        negative_regions = []
        new_values = {}
        # go through all linear regions
        for linear_region in self.linear_regions:
            # ReLU does not do anything to flat linear regions
            if linear_region.gradient != [0, 0]:
                # find the line representing ReLU
                xx, yy = linear_region.polygon.exterior.xy
                p = (xx[0], yy[0])
                # go from p along the -gradient until you reach 0
                distance_vector = [(self.values[p] * g) / math.pow(self.magnitude(linear_region.gradient), 2) for g in
                                   linear_region.gradient]
                p1 = (p[0] - distance_vector[0], p[1] - distance_vector[1])
                relu_vector = [-linear_region.gradient[1], linear_region.gradient[0]]
                scaling_factor = self.magnitude([self.hyperrectangle.x[1] - self.hyperrectangle.x[0],
                                                 self.hyperrectangle.y[1] - self.hyperrectangle.y[0]]) / self.magnitude(
                    relu_vector)
                # divide the linear region using ReLU (relu_vector)
                p2 = (p1[0] + scaling_factor * relu_vector[0], p1[1] + scaling_factor * relu_vector[1])
                p1 = (p2[0] - 2 * scaling_factor * relu_vector[0], p2[1] - 2 * scaling_factor * relu_vector[1])
                relu_line = LineString([p1, p2])
                split_regions = split(linear_region.polygon, relu_line)
                for region in split_regions.geoms:
                    # convert each region using point bank
                    altered_region = self.quantize_polygon(region, global_point_bank, index, self.values)
                    # group the result into positive and negative regions and change values of the negative regions to 0
                    xx, yy = altered_region.exterior.xy
                    negative = False
                    for i in range(len(xx)):
                        if self.values[(xx[i], yy[i])] > 0:
                            new_skeleton.linear_regions.append(LinearRegion(altered_region, linear_region.gradient))
                            break
                        elif self.values[(xx[i], yy[i])] < 0:
                            negative = True
                            new_values[(xx[i], yy[i])] = 0
                    if negative:
                        negative_regions.append(altered_region)
            else:
                xx, yy = linear_region.polygon.exterior.xy
                negative = False
                for i in range(len(xx)):
                    if self.values[(xx[i], yy[i])] <= 0:
                        new_values[(xx[i], yy[i])] = 0
                        negative = True
                if negative:
                    negative_regions.append(linear_region.polygon)
                else:
                    new_skeleton.linear_regions.append(linear_region)

        # merge all neighboring negative sections
        i = 0
        merged_regions = []
        while i < len(negative_regions):
            next_lr_index = i + 1
            found_neighbors = False
            while next_lr_index < len(negative_regions):
                lr = negative_regions[i]
                next_lr = negative_regions[next_lr_index]
                if self.is_intersection_proper_line(lr, next_lr):
                    negative_regions[i] = lr.union(next_lr)
                    found_neighbors = True
                    del negative_regions[next_lr_index]
                    next_lr_index -= 1
                next_lr_index += 1
            if found_neighbors:
                i -= 1
            else:
                merged_regions.append(negative_regions[i])
            i += 1
        for merged_region in merged_regions:
            merged_region = self.remove_collinear_points(merged_region)
            new_skeleton.linear_regions.append(LinearRegion(merged_region, [0, 0]))
        new_skeleton.values = self.values
        for p in new_skeleton.values:
            if p in new_values:
                new_skeleton.values[p] = 0
        new_skeleton.test_validity()
        return new_skeleton

    def plot_skeleton(self, title, ax, mode=0, save=False, point_bank=None):
        # prepare graph
        for linear_region in self.linear_regions:
            linear_region.new_plot(ax, self.values, point_bank, mode=mode)
        if save:
            plt.savefig(title + '.pdf')

    @staticmethod
    def clean_points(entity, error=1e-10):
        xx, yy = entity.coords.xy
        i = 0
        while i < len(xx) - 2:
            point = (xx[i], yy[i])
            point_to_check = (xx[i + 1], yy[i + 1])
            next_point = (xx[i + 2], yy[i + 2])
            l = LineString([point, next_point])
            if l.distance(Point(point_to_check)) < error:
                del xx[i + 1]
                del yy[i + 1]
            else:
                i += 1
        # The above loop omits first and last checks which are performed below
        if len(xx) > 1:
            l = LineString([(xx[-2], yy[-2]), (xx[0], yy[0])])
            if l.distance(Point((xx[-1], yy[-1]))) < error:
                del xx[-1]
                del yy[-1]
            l = LineString([(xx[-1], yy[-1]), (xx[1], yy[1])])
            if l.distance(Point((xx[0], yy[0]))) < error:
                del xx[0]
                del yy[0]
        return [(xx[i], yy[i]) for i in range(len(xx))]

    def remove_collinear_points(self, polygon: Polygon, poly=True) -> Union[Polygon, List[Tuple[float, float]]]:
        shell = self.clean_points(polygon.exterior)
        holes = []
        if polygon.interiors:
            for geom in polygon.interiors:
                holes.append(self.clean_points(geom))
        if poly:
            return Polygon(shell=shell, holes=holes)
        return shell

    def test_validity(self, full_test=True, skeleton_to_test=None):
        """ Test whether a skeleton covers the whole hyperrectangle, and if the linear regions do not overlap

        """
        if not skeleton_to_test:
            skeleton_to_test = self
        linear_region_union = skeleton_to_test.linear_regions[0].polygon
        # Check if any two linear regions overlap (if their intersection is a Polygon)
        for lr in skeleton_to_test.linear_regions[1:]:
            intersection = linear_region_union.intersection(lr.polygon)
            if intersection and intersection.area > 1e-2:
                if isinstance(intersection, (Polygon, MultiPolygon)):
                    if intersection.area > 1e-2:
                        print(intersection.area)
                        raise Exception
                elif isinstance(intersection, GeometryCollection):
                    for geom in intersection.geoms:
                        if isinstance(geom, (Polygon, MultiPolygon)):
                            if intersection.area > 1e-2:
                                print(intersection.area)
                                raise Exception
            linear_region_union = linear_region_union.union(lr.polygon)
        if full_test:
            # Check if linear_region_union covers the whole hyperrectangle
            R = Polygon(skeleton_to_test.hyperrectangle.convert_to_polygon())
            dif = R.difference(linear_region_union)
            if dif:
                if dif.area > pow(10, -2):
                    print(dif)
                    print(dif.area)
                    raise Exception
