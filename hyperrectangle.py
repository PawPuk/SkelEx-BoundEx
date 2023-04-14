import torch

from shapely.geometry import Polygon


class Hyperrectangle:
    def __init__(self, my_data):
        self.train_data, self.test_data = my_data
        self.x, self.y = self.find_bounds()

    def __str__(self):
        return "x is bounded by - " + str(self.x) + ".\ny is bounded by - " + str(self.y)

    def find_bounds(self):
        maxs = torch.max(torch.vstack((torch.max(self.train_data, 0)[0], torch.max(self.test_data, 0)[0])), 0)[0]
        mins = torch.min(torch.vstack((torch.min(self.train_data, 0)[0], torch.min(self.test_data, 0)[0])), 0)[0]
        x = (torch.round(mins[0] - abs(0.5*mins[0]), decimals=1).item(),
             torch.round(maxs[0] + abs(0.5*maxs[0]), decimals=1).item())
        y = (torch.round(mins[1] - abs(0.5*mins[1]), decimals=1).item(),
             torch.round(maxs[1] + abs(0.5*maxs[1]), decimals=1).item())
        return x, y


    def convert_to_polygon(self):
        p1 = (self.x[0], self.y[0])
        p2 = (self.x[0], self.y[1])
        p3 = (self.x[1], self.y[1])
        p4 = (self.x[1], self.y[0])
        return Polygon([p1, p2, p3, p4])
