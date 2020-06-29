import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib import colors
import numpy as np


def image_gallery(images, titles=None):
    """
    Utility to display images
    :param images:  (list[np.ndarray]) images to show
    :param titles:  (list[str]) image titles
    :return:
    """
    N = len(images)
    if N == 1:
        plt.imshow(images[0])
        plt.show()
        return None
    q = int(np.floor(np.sqrt(N)))
    p = int(np.ceil(np.sqrt(N)))
    if p * q < N:
        q = p
    f, axarr = plt.subplots(p, q, figsize=(2 * N//3, 2 * N //3))
    for i in range(0, p):
        for j in range(0, q):
            idx = q * i + j
            if idx < N:
                if titles is not None:
                    axarr[i, j].set_title(titles[idx])
                axarr[i, j].imshow(images[idx])
    plt.show()
    return None

