import numpy as np
from PIL import Image
import random
import sys
from collections import deque

class FloodFill():
    def __init__(self, path):
        self.image = self.load_image(path)
        # white background
        self.background = np.full((1, 3), 255)
        # black edges
        self.bounds = np.full((1, 3), 0)
        # Todo: maybe later split this in interior and background -> do not really see a reason why background should
        # Todo: be filled, but we will see
        # fill color
        self.fill = np.full((1, 3), 50)

    def load_image(self, path):
        image = Image.open(path)
        return np.asarray(image).copy()

    def startFill(self):
        shape_x, shape_y, _ = self.image.shape
        startingPointColor = self.bounds
        x_coord, y_coord = 0, 0
        while (startingPointColor == self.bounds).all():
            x_coord = random.randint(0, shape_x-1)
            y_coord = random.randint(0, shape_y-1)
            startingPointColor = self.image[x_coord][y_coord]
        self.floodfill((x_coord, y_coord))

        # self.floodFillNoRecursion((x_coord, y_coord))


        filled_image = Image.fromarray(self.image)
        filled_image.save("..\\..\\output\\filled_image.png")

    def floodFillBFS(self, coord):
        stack = deque()
        stack.append(coord)

        while len(stack) > 0:
            x, y = stack.pop()
            self.image[x][y] = self.fill

            shape_x, shape_y, _ = self.image.shape
            if x+1 < shape_x and (self.image[x+1][y] == self.background).all():
                stack.append((x+1, y))

            if x-1 >= 0 and (self.image[x-1][y] == self.background).all():
                stack.append((x-1, y))

            if y+1 < shape_y and (self.image[x][y+1] == self.background).all():
                stack.append((x, y+1))

            if y-1 >= 0 and (self.image[x][y-1] == self.background).all():
                stack.append((x, y-1))


    # Todo: find all startingPoints and start Flood Fill (simultaniously ?)
    def findStartPoints(self):
        pass

