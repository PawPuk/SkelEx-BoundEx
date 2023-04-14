from shapely.geometry import Polygon


class Hyperrectangle:
    def __init__(self, x_start, x_end, y_start, y_end):
        self.x = (x_start, x_end)
        self.y = (y_start, y_end)

    def __str__(self):
        return "x bounds - " + str(self.x) + ".\ny bounds - " + str(self.y)

    def convert_to_polygon(self):
        p1 = (self.x[0], self.y[0])
        p2 = (self.x[0], self.y[1])
        p3 = (self.x[1], self.y[1])
        p4 = (self.x[1], self.y[0])
        return Polygon([p1, p2, p3, p4])
