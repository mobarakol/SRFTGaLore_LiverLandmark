import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import xml.etree.ElementTree as ET
import json
from skimage.metrics import structural_similarity as SSIM
from glob import glob
import random
import os
import numpy as np

import matplotlib.pyplot as plt

class LandmarkDataset(Dataset):
    def __init__(self, root='../L3D_Dataset', transform=None, mode='train'):
        # file path
        self.img_path_all = glob(root + '/images/*.png')  # Update the path and pattern
        
        if transform:
            self.transform = transform      
        else:
            self.transform = T.ToTensor()

        self.mode = mode

    def __len__(self):
        return len(self.img_path_all)

    def __getitem__(self, idx):
        img_path = self.img_path_all[idx]
        image = load_image(img_path)
        mask = load_mask(img_path)
        depth = load_depth(img_path)
        image = self.transform(image)
        mask = torch.from_numpy(mask).long()
        depth = self.transform(depth)

        if self.mode == 'train':
            return image, depth, mask, str(img_path)
        else:
            return image, depth, mask, str(img_path)


def load_image(path):
    img = cv2.imread(str(path))
    img = cv2.resize(img, (1024, 1024))

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_depth(path):
    img = cv2.imread(str(path))
    img = cv2.resize(img, (1024, 1024))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    return img.astype(np.uint8)


def load_mask(path):
    mask_path = str(path).replace('images', 'masks_gt').replace('.png', '_convert.png')
    bgr_mask = cv2.imread(str(mask_path))
    if bgr_mask is None:
        print(f"Error: can't read {mask_path}")
        return np.zeros(shape=(4, 1024, 1024), dtype=np.uint8)

    rgb_mask = cv2.cvtColor(bgr_mask, cv2.COLOR_BGR2RGB)
    rgb_mask = cv2.resize(rgb_mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)

    # create class map
    class_map = np.zeros((1024, 1024), dtype=np.uint8) 
    class_map[np.all(rgb_mask == [255, 0, 0], axis=2)] = 1  # red   -> Class 1
    class_map[np.all(rgb_mask == [0, 255, 0], axis=2)] = 2  # green -> Class 2
    class_map[np.all(rgb_mask == [0, 0, 255], axis=2)] = 3  # blue  -> Class 3

    # create One-Hot encoded masks
    masks = np.zeros(shape=(4, 1024, 1024), dtype=np.uint8)

    # fill 4 channels based on class map
    masks[0][class_map == 0] = 255  # background
    masks[1][class_map == 1] = 255  # Class 1
    masks[2][class_map == 2] = 255  # Class 2
    masks[3][class_map == 3] = 255  # Class 3

    return masks


def apply_color(mask):
    h, w = mask.shape
    color_map = {
        0: (0, 0, 0),      # Black
        1: (255, 0, 0),    # Red
        2: (0, 255, 0),    # Green
        3: (0, 0, 255)     # Blue
    }
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        color_mask[mask == class_id] = color
    return color_mask


def save_and_show_img(image, pred_mask, gt_mask, filename, loss, save_path):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    image = np.clip(image, 0, 255).astype(np.uint8)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), subplot_kw=dict(xticks=[], yticks=[]))
    fig.suptitle(f"{os.path.basename(filename)} | Loss: {loss:.4f}", fontsize=12)

    pred_color = apply_color(pred_mask)
    gt_color = apply_color(gt_mask)

    axs[0].imshow(image)
    axs[0].set_title("Input Image")

    axs[1].imshow(pred_color, interpolation="none")
    axs[1].set_title("Predicted Mask")

    axs[2].imshow(gt_color, interpolation="none")
    axs[2].set_title("Ground Truth")

    # save image
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig)


def ssim(img1, img2):
    img1 = cv2.resize(img1, (224, 224))
    img2 = cv2.resize(img2, (224, 224))
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    ss = SSIM(img1, img2)
    return ss
