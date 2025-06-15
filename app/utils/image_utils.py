import cv2
from matplotlib import pyplot as plt

def show_image(im, title='Image'):
    """
    Exibe uma imagem BGR usando matplotlib, convertendo para RGB.
    """
    color = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.imshow(color)
    plt.title(title)
    plt.axis('off')
    plt.show()