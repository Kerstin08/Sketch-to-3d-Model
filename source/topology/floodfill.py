import numpy as np
from PIL import Image
import random
import sys
from collections import deque

class FloodFill():
    def __init__(self, path):
        self.image = self.load_image(path)
        self.background = np.full((1, 3), 255)
        self.bounds = np.full((1, 3), 0)

    def load_image(self, path):
        image = Image.open(path)
        return np.asarray(image).copy()

    def startFill(self):
        start_points = self.find_start_points()
        for i in start_points:
            self.flood_fill_BFS(i)

        filled_image = Image.fromarray(self.image)
        filled_image.save("..\\..\\output\\filled_image.png")

    def flood_fill_BFS(self, seed):
        stack = deque()
        stack.append(seed)
        fill = self.image[seed[0]][seed[1]]

        while len(stack) > 0:
            x, y = stack.pop()
            self.image[x][y] = fill
            shape_x, shape_y, _ = self.image.shape
            if x + 1 < shape_x and (self.image[x + 1][y] == self.background).all():
                stack.append((x + 1, y))

            if x - 1 >= 0 and (self.image[x - 1][y] == self.background).all():
                stack.append((x - 1, y))

            if y + 1 < shape_y and (self.image[x][y + 1] == self.background).all():
                stack.append((x, y + 1))

            if y - 1 >= 0 and (self.image[x][y - 1] == self.background).all():
                stack.append((x, y - 1))

    def find_start_points(self):
        seeds = deque()
        shape_x, shape_y, _ = self.image.shape
        for x in range(shape_x):
            for y in range(shape_y):
                if not (self.image[x][y] == self.background).all() and not (self.image[x][y] == self.bounds).all():
                    seeds.append((x, y))
        return seeds

