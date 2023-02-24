# 8-connected stack-based floodfill to preprocess for hole determination and create silhouette image
import os.path
import typing

import numpy
from PIL import Image
from collections import deque
from pathlib import Path

from source.util import OpenEXR_utils
from source.util import data_type
from source.util import sketch_utils

background = 1
bounds = 0
fill = 0


def startFill(
        image: numpy.ndarray,
        image_path: str,
        output_dir: str,
        write_debug_png: bool = True
) -> typing.Tuple[numpy.ndarray, str]:
    image = image / 255
    start_points = find_start_points(image)
    for i in start_points:
        flood_fill_BFS(image, i)

    image = sketch_utils.unpad(image, 1)
    filename = Path(image_path)
    exr_path = os.path.join(output_dir, filename.stem + '_filled.exr')
    OpenEXR_utils.writeImage(image, data_type.Type.silhouette, exr_path)

    if write_debug_png:
        image_png = image * 255
        filled_image = Image.fromarray(image_png).convert('RGB')
        png_path = os.path.join(output_dir, filename.stem + '_filled.png')
        filled_image.save(png_path)

    return image, exr_path


def flood_fill_BFS(
        image: numpy.ndarray,
        seed: int
):
    stack = deque()
    stack.append(seed)

    while len(stack) > 0:
        x, y = stack.pop()
        shape_x, shape_y = image.shape
        if x < 1 or y < 1 or x > shape_x - 2 or y > shape_y - 2:
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


def find_start_points(
        image: numpy.ndarray
) -> deque:
    seeds = deque()
    shape_x, shape_y = image.shape
    for x in range(shape_x):
        for y in range(shape_y):
            if not (image[x][y] == background).all() and not (image[x][y] == bounds).all():
                seeds.append((x, y))
    return seeds
