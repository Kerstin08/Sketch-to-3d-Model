import numpy as np
from PIL import Image
import random
import sys
from collections import deque

class FloodFill():
    def __init__(self, path):
        self.background = 255
        self.bounds = 0
        self.fill = 0
        self.image = self.load_image(path)


    def load_image(self, path):
        image = Image.open(path).convert("L")
        image_arr = np.asarray(image).copy()
        image_pad = np.pad(image_arr, 1, mode='constant', constant_values=255)
        return image_pad

    def startFill(self):
        start_points = self.find_start_points()
        for i in start_points:
            self.flood_fill_BFS(i)
        filled_image = Image.fromarray(self.image)
        filled_image.save("..\\..\\output\\filled_image.png")
        return self.image

    def flood_fill_BFS(self, seed):
        stack = deque()
        stack.append(seed)

        while len(stack) > 0:
            x, y = stack.pop()
            self.image[x][y] = self.fill
            if self.image[x + 1][y] == self.background:
                stack.append((x + 1, y))

            if self.image[x - 1][y] == self.background:
                stack.append((x - 1, y))

            if self.image[x][y + 1] == self.background:
                stack.append((x, y + 1))

            if self.image[x][y - 1] == self.background:
                stack.append((x, y - 1))

    def find_start_points(self):
        seeds = deque()
        shape_x, shape_y = self.image.shape
        for x in range(shape_x):
            for y in range(shape_y):
                if not (self.image[x][y] == self.background).all() and not (self.image[x][y] == self.bounds).all():
                    seeds.append((x, y))
        return seeds

