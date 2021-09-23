from typing import List, Optional, Dict

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F


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


def pad_batch(img_batch: List[Tensor], pad_kwargs: Dict = {},
              bboxes_batch: Optional[List[Tensor]] = None,
              masks_batch: Optional[List[Tensor]] = None) -> Dict:
    """
    Useful for collate_fn when we want to keep image aspect ratios in a batch
    This pads all imgs to a common size. The common height and common width
    are determined to be the maximum of the heights/widths of the inputs
    respectively.

    Args:
        img_batch - A list of tensors representing individual images
        pad_kwargs - Dictionary of keyword arguments to pass to
            `torch.nn.functional.pad` as applied to the images.
        bboxes_batch (optional) - A list of tensors representing bboxes for
            each image. The bboxes are expected to be in xyxy format and in
            absolute units. Therefore, each tensor should have shape (N, 4),
            where N is the number of bboxes.
        masks_batch (optional) - A list of 3D tensors each having the same
            height and width as the corresponding image. Each tensor should
            have shape (N, H, W), where N matches the number of bboxes.
    Returns:
        (dict) - A dictionary with keys for the transformed versions of each
            of the inputs. Specifically:
            {
                'img_batch': List[Tensor],
                'bboxes_batch': List[Tensor] (optional),
                'masks_batch': List[Tensor] (optional)
            }
    # TODO add bbox, keypoint and mask shifting to match the image padding
    """
    ret = {}  # return dict
    # Fill in for missing inputs
    if bboxes_batch is None:
        bboxes_batch = [None] * len(img_batch)
    if masks_batch is None:
        masks_batch = [None] * len(img_batch)
    # Pad all images to same size keeping into account the bbox shifts
    max_h, max_w = np.max([img.shape[-2:] for img in img_batch], axis=0)
    for i, (img, bboxes, masks) in enumerate(
            zip(img_batch, bboxes_batch, masks_batch)):
        pad_left, pad_right, pad_top, pad_bottom = 0, 0, 0, 0
        if (diff_w := max_w - img.shape[-1]) > 0:
            pad_left = diff_w//2
            pad_right = diff_w - pad_left
        if (diff_h := max_h - img.shape[-2]) > 0:
            pad_top = diff_h//2
            pad_bottom = diff_h - pad_top
        if any([pad_left, pad_right, pad_top, pad_bottom]):
            img_batch[i] = F.pad(
                img, (pad_left, pad_right, pad_top, pad_bottom), **pad_kwargs)
            if bboxes is not None:
                bboxes[:, 0::2] += pad_left
                bboxes[:, 1::2] += pad_top
            if masks is not None:
                masks_batch[i] = F.pad(
                    masks, (pad_left, pad_right, pad_top, pad_bottom))
    ret['img_batch'] = img_batch
    if bboxes_batch is not None:
        ret['bboxes_batch'] = bboxes_batch
    if masks_batch is not None:
        ret['masks_batch'] = masks_batch
    return ret
