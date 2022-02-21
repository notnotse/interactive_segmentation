import json
import os
import random
import shutil
import sys

import PIL.Image
import cv2
import numpy as np
from tqdm import tqdm


def _cv2_to_bitmap(array):
    """
    Turns a cv2 image into a bitmap.
    :param array: cv2 image.
    :return: Bitmap.
    """
    # Split the three channels
    r, g, b = np.split(array, 3, axis=2)
    r = r.reshape(-1)
    g = r.reshape(-1)
    b = r.reshape(-1)

    # Standard RGB to grayscale
    bitmap = list(map(lambda x: 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2], zip(r, g, b)))
    bitmap = np.array(bitmap).reshape([array.shape[0], array.shape[1]])
    bitmap = np.dot((bitmap > 128).astype(float), 255)
    return PIL.Image.fromarray(bitmap.astype(np.uint8))


def _coco_to_bitmap_dataset(img_dir, json_file, save_dir, margin=100):
    """
    Takes coco annotation data and creates training data as cropped images and bitmaps.
    :param img_dir: Image dir path.
    :param json_file: JSON-file with all annotation data.
    :param save_dir: Where to save the result.
    :param margin: Margin for how much to add around each object in terms of pixels.
    """
    with open(json_file) as in_file:

        coco_dict = json.load(in_file)
        img_id_to_img_name = {}
        gt_save_path = save_dir + "\\gt"
        img_save_path = save_dir + "\\img"

        # Create dirs.
        if not os.path.exists(gt_save_path):
            os.makedirs(gt_save_path)
        if not os.path.exists(img_save_path):
            os.makedirs(img_save_path)
        # Load id-name.
        for img_id in coco_dict["images"]:
            img_id_to_img_name[img_id["id"]] = img_id["file_name"]

        print("Creating bitmaps & cropping images...")
        for i, obj_dict in tqdm(enumerate(coco_dict["annotations"], start=1),
                                total=len(coco_dict["annotations"])):

            img_path = os.path.join(img_dir, img_id_to_img_name[obj_dict["image_id"]])

            img = cv2.imread(img_path)

            img_height, img_width, _ = img.shape

            final_name = "annotate_" + str(i)

            bmp_path = os.path.join(gt_save_path, final_name + ".bmp")
            img_path = os.path.join(img_save_path, final_name + ".jpg")

            x, y, w, h = [int(num) for num in obj_dict["bbox"]]
            annotations = obj_dict["segmentation"][0]
            count = 0
            coords = []

            # Group up all XY-coordinates.
            while True:
                arr = [int(num) for num in annotations[count:count + 2]]
                count += 2
                if len(arr) == 0:
                    break
                coords.append(arr)

            poly = np.array(coords)

            # Adjust crop coordinates.
            y = 0 if y - margin < 0 else y - margin
            h = img_height if h + margin > img_height else h + margin * 2
            x = 0 if x - margin < 0 else x - margin
            w = img_width if w + margin > img_width else w + margin * 2

            # Crop & save.
            crop_img = img[y:y + h, x:x + w]
            cv2.imwrite(img_path, crop_img)

            img = np.zeros(img.shape, dtype=np.uint8)
            cv2.polylines(img, [poly], True, (255, 255, 255))
            cv2.fillPoly(img, [poly], (255, 255, 255))
            crop_img = img[y:y + h, x:x + w]

            bitmap = _cv2_to_bitmap(crop_img)
            bitmap.save(bmp_path)


def _bitmap_dataset_val_split(dir_path, val_split=0.15):
    """
    Take all the training data and create a validation split.
    Example: 100 instances - val_split=0.1 -> 90/10 Split.
    :param dir_path: Path of full dataset.
    :param val_split: Validation split percentage.
    """
    gt_dir = dir_path + "\\gt"
    img_dir = dir_path + "\\img"
    train_dir_img = "train_val_split\\train\\img"
    train_dir_gt = "train_val_split\\train\\gt"
    val_dir_img = "train_val_split\\val\\img"
    val_dir_gt = "train_val_split\\val\\gt"
    all_dirs = [train_dir_img, train_dir_gt, val_dir_img, val_dir_gt]

    gt_len = len(os.listdir(gt_dir))
    img_len = len(os.listdir(img_dir))
    assert gt_len == img_len

    # Create dirs.
    for dir in all_dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

    img_names = set()
    img_name_list = []
    val_count = round(gt_len * val_split)

    for img in os.listdir(img_dir):
        img_name = img.split(".")[0]
        if img_name in img_names:
            raise ValueError("Files with the same name exists.")
        img_names.add(img_name)
        img_name_list.append(img_name)

    random.shuffle(img_name_list)
    random.shuffle(img_name_list)

    for i, img_name in tqdm(enumerate(img_name_list)):
        src_img_path = img_dir + "\\" + img_name + ".jpg"
        src_mask_path = gt_dir + "\\" + img_name + ".bmp"

        if i < val_count:
            dst_img_path = val_dir_img + "\\" + img_name + ".jpg"
            dst_mask_path = val_dir_gt + "\\" + img_name + ".bmp"
            shutil.copy(src_img_path, dst_img_path)
            shutil.copy(src_mask_path, dst_mask_path)
        else:
            dst_img_path = train_dir_img + "\\" + img_name + ".jpg"
            dst_mask_path = train_dir_gt + "\\" + img_name + ".bmp"
            shutil.copy(src_img_path, dst_img_path)
            shutil.copy(src_mask_path, dst_mask_path)


def convert_cvat_coco_data(data_folder, save_dir):
    """
    Converts cvat COCO annotation data into trainable bitmaps.
    :param data_folder: Cvat data folder. Contents: annotations & images directory.
    :param save_dir: Where to save all the training data.
    """
    img_dir = data_folder + "/images"
    json_file = data_folder + "/annotations/instances_default.json"
    _coco_to_bitmap_dataset(img_dir, json_file, save_dir, margin=100)
    _bitmap_dataset_val_split(save_dir, 0.2)


if __name__ == "__main__":
    data_dir = sys.argv[1:][0]
    save_dir = sys.argv[1:][1]
    convert_cvat_coco_data(data_dir, save_dir)
