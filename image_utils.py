import numpy as np


def normalize(img: np.ndarray, div255=True) -> np.ndarray:
    """
    normalize using standard pytorch model normalization parameters
    expects RGB image with 0-255 pixels as input
    returns float type
    """
    img = img.astype(float)
    if div255:
        img /= 255
    img -= np.array([0.485, 0.456, 0.406])
    img /= np.array([0.229, 0.224, 0.225])
    return img


def denormalize(img: np.ndarray) -> np.ndarray:
    """
    normalize using standard pytorch model normalization parameters
    expects image with float pixels as input
    returns np.uint8 type
    """
    img *= np.array([0.229, 0.224, 0.225])
    img += np.array([0.485, 0.456, 0.406])
    img *= 255
    return img.astype(np.uint8)
