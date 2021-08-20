import numpy as np


def normalize(img: np.ndarray, div255=True, mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]) -> np.ndarray:
    """
    normalize using standard pytorch model normalization parameters
    expects RGB image with 0-255 pixels as input
    returns float type
    """
    img = img.astype(float)
    if div255:
        img /= 255
    mean = np.array(mean)
    std = np.array(std)
    if len(img.shape)  == 2 or img.shape[2] == 1:
        mean = mean.mean()
        std = std.mean()
    img -= mean
    img /= std
    return img


def denormalize(img: np.ndarray, mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]) -> np.ndarray:
    """
    normalize using standard pytorch model normalization parameters
    expects image with float pixels as input
    returns np.uint8 type
    """
    mean = np.array(mean)
    std = np.array(std)
    if len(img.shape) == 2 or img.shape[2] == 1:
        mean = mean.mean()
        std = std.mean()
    img *= std
    img += mean
    img *= 255
    return img.astype(np.uint8)
