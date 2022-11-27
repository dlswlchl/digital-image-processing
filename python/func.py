import cv2
from matplotlib import pyplot as plt
import numpy as np
import sklearn.decomposition
import sklearn.neighbors

def gaussian_distance(x, y, sigma):
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))


def calculate_weight(image, i, j, k, l, sigma_r, sigma_s):
    intensity_distance = gaussian_distance(image[i][j], image[k][l], sigma_r)
    location_distance = gaussian_distance(np.array((i, j,)), np.array((k, l,)), sigma_s)
    return intensity_distance * location_distance