import random

import cv2
import numpy as np
from PIL import Image
from albumentations import Compose, HorizontalFlip, ShiftScaleRotate, Transpose, Resize, RandomBrightnessContrast, \
    RGBShift, Sharpen, HueSaturationValue

from isegm.data.datasets.mask import MaskDataset
from isegm.utils.vis import draw_with_blend_and_clicks


def run_sample_loop():
    """
    Run a loop that shows samples of data augmentations.
    """
    crop_size = (500, 500)
    train_augmentator = Compose([
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(scale_limit=(0.0, 0.2), border_mode=cv2.BORDER_CONSTANT, rotate_limit=(45, 45), p=1),
        Transpose(),
        Resize(450, 450),
        RandomBrightnessContrast(p=0.5, brightness_limit=(0.0, 0.1), contrast_limit=(0.0, 0.15)),
        RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        Sharpen(p=0.5),
        HueSaturationValue(p=0.5)
    ])

    train = "datasets/mask/train"

    trainset = MaskDataset(
        dataset_path=train,
        min_object_area=750,
        augmentator=train_augmentator)

    random.seed(33)
    while True:
        try:
            rand_int = random.randint(0, 3000)
            sample_aug = trainset[rand_int]
            img_aug = sample_aug["images"]
            img_aug = img_aug.numpy()
            img_aug = np.transpose(img_aug, (1, 2, 0))
            print("AUG shape:", img_aug.shape)

            sample = trainset.get_sample(rand_int)
            img = sample.image
            mask = sample.gt_mask
            img_vis = draw_with_blend_and_clicks(img, mask=mask, alpha=0.5)
            print("IMG Shape:", img_vis.shape)

            break

            img = Image.fromarray(img_vis)
            img.show()
            img.close()

            img = Image.fromarray((img_aug * 255).astype(np.uint8))
            img.show()
            img.close()

            press = input("Enter something in terminal to show next sample.")

            print("------")
        except IndexError:
            pass


if __name__ == '__main__':
    run_sample_loop()
