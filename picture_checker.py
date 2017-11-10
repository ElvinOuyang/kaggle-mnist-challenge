import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def display(img):
    one_image = img.reshape(28, 28)
    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)
