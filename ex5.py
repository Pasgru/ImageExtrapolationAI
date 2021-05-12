"""
Author: <Pascal Gruber>
Matr.Nr.: <12005914>
Exercise <5>
"""
import pickle
from PIL import Image
import numpy as np
import random


def random_img(input_arrays, known_arrays, border_x, border_y, sample_ids):
    target = input_arrays[known_arrays == 0]
    for i in range(len(target)):
        target[i] = random.randint(0, 255)
    return target


def mean_img(input_arrays, known_arrays, border_x, border_y, sample_ids):
    mean = np.mean(input_arrays)
    target = input_arrays[known_arrays == 0]
    target[:] = mean
    return target


def make_img(data):
    for i in data:
        test = i[0]
        img = Image.fromarray(i[0], 'L')
        img.save(f"images/{i[4]}.png")


def pkl_to_data(file):
    with open(file, 'rb') as file:
        data = pickle.load(file)

    input_arrays = data['input_arrays']
    known_arrays = data['known_arrays']
    border_x = data['borders_x']
    border_y = data['borders_y']
    sample_ids = data['sample_ids']

    for i in range(len(input_arrays)):
        yield input_arrays[i], known_arrays[i], border_x[i], border_y[i], sample_ids[i]


if __name__ == "__main__":
    '''
    Random: -8356
    Mean: -3787
    '''
    file = "example_testset.pkl"
    data = pkl_to_data(file)

    targets = []
    for input_arrays, known_arrays, border_x, border_y, sample_ids in data:
        target = random_img(input_arrays, known_arrays, border_x, border_y, sample_ids)

        input_arrays[known_arrays == 0] = target
        img = Image.fromarray(input_arrays, 'L')
        img.save(f"outputIMG/{sample_ids}.png")

        targets.append(target)

    with open("solution.pkl", "wb") as submission:
        pickle.dump(targets, submission)
