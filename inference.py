#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 09:43:26 2022.

@author: anup
"""
import os
import cv2
import torch
import warnings
import albumentations
import numpy as np
import glob
from albumentations.pytorch.transforms import ToTensorV2
from torch import device as device_
warnings.filterwarnings("ignore")

device = device_("cuda" if torch.cuda.is_available() else "cpu")

LAEBLS = {
    0: "No finding",
    1: "Allianz"
}


def get_model_name(path):
    """Get the latest model path."""
    no_files = len(glob.glob(os.path.join(path, '*.bin')))
    model_name = 'model_' + str(no_files) + '.bin'
    return model_name


MODEL = torch.load(os.path.join('models', get_model_name('models')))
MODEL.to(device)
MODEL.eval()


def get_image_output(path):
    """Process the image to get output."""
    file_name = ''.join(os.path.basename(path).split('.')[:-1])
    dir_name = os.path.dirname(path)
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = albumentations.Compose([
        albumentations.Normalize(mean=(0, 0, 0), std=(
            1, 1, 1), max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0)])
    transformed = transform(image=image)

    transformed_image = transformed["image"].to(device)
    transformed_image.shape
    model_out = MODEL([transformed_image])
    bounding_boxes = model_out[0]['boxes'].cpu(
    ).detach().numpy().astype(np.int32).tolist()
    classes = model_out[0]['labels'].cpu().detach(
    ).numpy().astype(np.int32).tolist()
    scores = model_out[0]['scores'].cpu().detach(
    ).numpy().astype(np.float16).tolist()
    result = zip(bounding_boxes, scores, classes)
    result = [i for i in result if i[1] > 0.99]
    for index, (box, score, label) in enumerate(result):
        cropped_image = image[box[1]:box[3], box[0]:box[2]]
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
        instance_crop_name = os.path.join(
            dir_name, file_name + '_' + str(index) + '_' + 'cropped' + '.png')
        cv2.imwrite(instance_crop_name, cropped_image)
        cv2.rectangle(image,
                      (box[0], box[1]),
                      (box[2], box[3]),
                      (220, 0, 0),
                      1)
        display_text = str(LAEBLS[label]) + " " + str(round(score, 4))
        cv2.putText(image,
                    display_text,
                    (box[0], box[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (220, 0, 0),
                    1)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    out_file_name = os.path.join(dir_name, file_name + '_bounding_box.png')
    cv2.imwrite(out_file_name, image)
