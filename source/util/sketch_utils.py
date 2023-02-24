# Preprocess operations for sketches
import os

import numpy
import numpy as np
from PIL import Image
from pathlib import Path

background = 255
bounds = 0


def load_image(
        image_path: str,
        pad_image: bool
) -> numpy.ndarray:
    if not os.path.exists(image_path):
        raise Exception("File not found {}.".format(image_path))
    image = Image.open(image_path).convert('L')
    image_arr = np.asarray(image).copy()
    if pad_image:
        image_arr = np.pad(image_arr, 1, mode='constant', constant_values=255)
    return image_arr


def unpad(
        img: numpy.ndarray,
        pad_width: int
) -> numpy.ndarray:
    shape_x, shape_y = img.shape
    n_shape_r, n_shape_l = 0 + pad_width, shape_x - pad_width
    return img[n_shape_r:n_shape_l, n_shape_r:n_shape_l]


def clean_userinput(
        image_path: str,
        output_path: str
):
    image = load_image(image_path, False)
    filename = Path(image_path)
    for i in range(len(image[0])):
        for j in range(len(image[1])):
            if image[i, j] > bounds and image[i, j] < background:
                image[i, j] = background

    cleaned_image_path = os.path.join(output_path, filename.name)
    cleaned_image = Image.fromarray(image)
    cleaned_image.save(cleaned_image_path)
