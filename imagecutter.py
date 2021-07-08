"""
Author: <Pascal Gruber>
Matr.Nr.: <12005914>
Exercise <4>
"""

import numpy as np
import dill as pkl


def imageCutter(image_array, border_x, border_y):
    if not isinstance(image_array, np.ndarray):
        raise NotImplementedError("Array is not of type np.ndarray")

    if len(image_array.shape) != 2 or image_array.shape[0] == 0 or image_array.shape[1] == 0:
        raise NotImplementedError("Array is of wrong shape")

    bx_t, bx_b = border_x
    by_l, by_r = border_y

    if not isinstance(bx_t, int) or not isinstance(bx_b, int) or not isinstance(by_l, int) or not isinstance(by_r, int):
        raise ValueError("Border input is not and integer")

    if bx_t < 1 or bx_b < 1 or by_l < 1 or by_r < 1:
        raise ValueError("One of the borders is smaller than 1")

    height = image_array.shape[0]
    width = image_array.shape[1]

    if width - (by_l + by_r) < 16:
        raise ValueError(
            f"Top and Bottom border cut it smaller than 16, only leaving: {image_array.shape[0] - (by_l + by_r)}")

    if height - (bx_t + bx_b) < 16:
        raise ValueError(
            f"Left and Right border cut it smaller than 16, only leaving: {image_array.shape[1] - (bx_t + bx_b)}")

    input_array = np.zeros(image_array.shape).astype(image_array.dtype)
    known_array = np.zeros(image_array.shape).astype(image_array.dtype)

    input_array[bx_t:height - bx_b, by_l:width - by_r] = image_array[bx_t:height - bx_b, by_l:width - by_r]

    known_array[bx_t:height - bx_b, by_l:width - by_r] = 1

    target_array = image_array[known_array == 0]

    return input_array, known_array, target_array


if __name__ == "__main__":
    with open("unittest_inputs_outputs.pkl", "rb") as ufh:
        all_inputs_outputs = pkl.load(ufh)
        all_inputs = all_inputs_outputs['inputs']
        all_outputs = all_inputs_outputs['outputs']

    array = np.arange(900).reshape((30, 30)).astype(np.float32)
    print(imageCutter(all_inputs[2][0], (75, 13), (131, 99)))
