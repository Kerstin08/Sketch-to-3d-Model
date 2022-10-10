from PIL import Image
from collections import deque

background = 255
bounds = 0
fill = 0

def startFill(image):
    start_points = find_start_points(image)
    for i in start_points:
        flood_fill_BFS(image, i)

    filled_image = Image.fromarray(image)
    filled_image.save(r"C:\Users\Kerstin\Documents\MasterThesis\thesis_images\filled_image.png")
    return image

def flood_fill_BFS(image, seed):
    stack = deque()
    stack.append(seed)

    while len(stack) > 0:
        x, y = stack.pop()
        shape_x, shape_y = image.shape
        if x < 1 or y < 1 or x > shape_x-2 or y > shape_y-2:
            continue

        image[x][y] = fill
        if image[x + 1][y] == background:
            stack.append((x + 1, y))

        if image[x - 1][y] == background:
            stack.append((x - 1, y))

        if image[x][y + 1] == background:
            stack.append((x, y + 1))

        if image[x][y - 1] == background:
            stack.append((x, y - 1))

        if image[x + 1][y + 1] == background:
            stack.append((x + 1, y + 1))

        if image[x + 1][y - 1] == background:
            stack.append((x + 1, y - 1))

        if image[x - 1][y + 1] == background:
            stack.append((x - 1, y + 1))

        if image[x - 1][y - 1] == background:
            stack.append((x - 1, y - 1))

def find_start_points(image):
    seeds = deque()
    shape_x, shape_y = image.shape
    for x in range(shape_x):
        for y in range(shape_y):
            if not (image[x][y] == background).all() and not (image[x][y] == bounds).all():
                seeds.append((x, y))
    return seeds