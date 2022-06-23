#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 08:28:56 2022.

@author: anup
"""

import os
import random
import glob
import pandas as pd
import numpy as np
from PIL import Image


background_blue = glob.glob(os.path.join(
    'data', 'backgrounds', 'blue', '*.jpg'))
background_white = glob.glob(os.path.join(
    'data', 'backgrounds', 'white', '*.jpg'))
background_multi = glob.glob(os.path.join(
    'data', 'backgrounds', 'multi', '*.jpg'))

background_blue_list = [
    (i, os.path.join('data', 'logo', 'blue.jpeg')) for i in background_blue]
background_white_list = [
    (i, os.path.join('data', 'logo', 'white.jpeg'))
    for i in background_white]
background_multi_list = [
    (i, os.path.join('data', 'logo', 'multi.png'))
    for i in background_multi]


image_ratio = [0.2, 0.4, 0.6, 0.8, 1]
frame_no = 0

dataset_list = []
for ratio in image_ratio:
    background_blue_rand = random.choices(background_blue_list, k=10)
    background_white_rand = random.choices(background_white_list, k=10)
    background_multi_rand = random.choices(background_multi_list, k=10)
    image_base = background_blue_rand + background_white_rand +\
        background_multi_rand
    for index, image_details in enumerate(image_base):
        Image1 = Image.open(image_details[0])
        source_width, source_height = Image1.size
        new_source_width = np.int32(round(source_width*(1000/source_width), 0))
        new_source_height = np.int32(
            round(source_height*(1000/source_width), 0))
        Image1 = Image1.resize((new_source_width, new_source_height),
                               Image.Resampling.LANCZOS)
        source_width, source_height = new_source_width, new_source_height

        Image2 = Image.open(image_details[1])
        width, height = Image2.size
        new_width = np.int32(round(width*ratio, 0))
        new_height = np.int32(round(height*ratio, 0))
        Image2 = Image2.resize((new_width, new_height),
                               Image.Resampling.LANCZOS)
        for i in range(0, source_width, 200):
            for j in range(0, source_height, 200):
                x_min = i
                y_min = j
                x_max = x_min + new_width
                y_max = y_min + new_height
                if (x_max <= source_width) & (y_max <= source_height):
                    print("Processing frame no: ", frame_no)
                    Image1copy = Image1.copy()
                    Image2copy = Image2.copy()
                    if os.path.basename(image_details[1]) == 'multi.png':
                        Image1copy.paste(Image2copy, (x_min, y_min), Image2copy)
                    else:
                        Image1copy.paste(Image2copy, (x_min, y_min))
                    image_array = np.array(Image1copy)
                    file_name = os.path.join('data', 'train', 'frame_' +
                                             str(frame_no) + '.jpg')
                    Image1copy.save(file_name)
                    json_dict = {}
                    json_dict['image_id'] = file_name
                    json_dict['class_name'] = 'Allianz'
                    json_dict['class_id'] = 1
                    json_dict['x_min'] = x_min
                    json_dict['y_min'] = y_min
                    json_dict['x_max'] = x_max
                    json_dict['y_max'] = y_max
                    dataset_list.append(json_dict)
                    if frame_no % 10 == 0:
                        print("Processing frame no: ", frame_no)
                        frame_no += 1
                        file_name = os.path.join('data', 'train', 'frame_' +
                                                 str(frame_no) + '.jpg')
                        json_dict = {}
                        json_dict['image_id'] = file_name
                        json_dict['class_name'] = 'No finding'
                        json_dict['class_id'] = 0
                        json_dict['x_min'] = 0
                        json_dict['y_min'] = 0
                        json_dict['x_max'] = 1
                        json_dict['y_max'] = 1
                        dataset_list.append(json_dict)
                        Image1.save(file_name)
                    frame_no += 1

dataset = pd.DataFrame(dataset_list)
dataset.to_csv(os.path.join('data', 'train', 'train.csv'), index=False)

