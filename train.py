#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 14:55:40 2022

@author: anup
"""

from sklearn.model_selection import train_test_split
from torch import device as device_
import warnings
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
import torch
from albumentations.pytorch.transforms import ToTensorV2
import albumentations
from tqdm import tqdm
import os
import copy
import gc
import cv2
import glob
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
device = device_("cuda" if torch.cuda.is_available() else "cpu")


class DataPrep(Dataset):
    """Class to prepare data."""

    def __init__(self, df, transforms=None):
        self.df = df
        self.image_ids = df['image_id'].unique()
        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        bboxes = self.df[self.df['image_id'] == image_id]
        bboxes = bboxes.reset_index(drop=True)
        image = cv2.imread(self.image_ids[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if bboxes.loc[0, "class_id"] == 0:
            bboxes = bboxes.loc[[0], :]
        boxes = bboxes[['x_min', 'y_min', 'x_max', 'y_max']].values
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)
        labels = torch.tensor(bboxes["class_id"].values, dtype=torch.int64)
        iscrowd = torch.zeros((bboxes.shape[0],), dtype=torch.int64)
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd
        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            target['boxes'] = torch.tensor(sample['bboxes'])
        if target["boxes"].shape[0] == 0:
            target["boxes"] = torch.from_numpy(np.array([[0.0, 0.0, 1.0, 1.0]]))
            target["area"] = torch.tensor([1.0], dtype=torch.float32)
            target["labels"] = torch.tensor([0], dtype=torch.int64)
        return image, target


def get_train_transform():
    return albumentations.Compose([
        albumentations.Flip(0.5),
        albumentations.RandomBrightnessContrast(p=0.2),
        albumentations.ShiftScaleRotate(
            scale_limit=0.1, rotate_limit=45, p=0.25),
        albumentations.LongestMaxSize(max_size=800, p=1.0),
        albumentations.Normalize(mean=(0, 0, 0), std=(
            1, 1, 1), max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def get_valid_transform():
    return albumentations.Compose([
        albumentations.Normalize(mean=(0, 0, 0), std=(
            1, 1, 1), max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def collate_fn(batch):
    return tuple(zip(*batch))


def train_loop_fn(data_loader,
                  model,
                  optimizer,
                  device,
                  t_l,
                  scheduler=None):
    running_loss = 0.0
    for images, labels in tqdm(data_loader):
        images = list(image.to(device) for image in images)
        labels = [{k: v.to(device) for k, v in l.items()} for l in labels]
        optimizer.zero_grad()
        loss_dict = model(images, labels)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        running_loss += losses.item()
        del images, labels
        gc.collect()
        torch.cuda.empty_cache()
    train_loss = running_loss / t_l
    scheduler.step(train_loss)
    return train_loss


def eval_loop_fn(data_loader, model, device, v_l):
    running_loss = 0.0
    for images, labels in tqdm(data_loader):
        images = list(image.to(device) for image in images)
        labels = [{k: v.to(device) for k, v in l.items()} for l in labels]
        loss_dict = model(images, labels)
        losses = sum(loss for loss in loss_dict.values())
        running_loss += losses.item()
        del images, labels
        gc.collect()
        torch.cuda.empty_cache()
    valid_loss = running_loss / v_l
    return valid_loss


def get_model_name(path):
    no_files = len(glob.glob(path + '/*.bin'))
    model_name = path + '/model_' + str(no_files + 1) + '.bin'
    return model_name


def main():
    train_path = os.path.join('data', 'train', 'train.csv')
    train = pd.read_csv(train_path)
    label_dict = {
        0: "No finding",
        1: "Allianz"
    }
    train_list = list(train)
    train['sample'] = copy.deepcopy(train['class_id'])

    df_train, df_valid, y_train, y_test = train_test_split(
        train[train_list],
        train['sample'].values.ravel(),
        test_size=0.20,
        random_state=42,
        stratify=train['sample'].values.ravel())

    df_train = df_train[train_list]
    df_valid = df_valid[train_list]
    train_data = DataPrep(df_train, get_train_transform())
    validation_data = DataPrep(df_valid, get_valid_transform())
    train_data_length = float(len(train_data))
    validation_data_length = float(len(validation_data))

    training_dataloader = DataLoader(
        train_data,
        batch_size=4,
        shuffle=True,
        num_workers=8,
        collate_fn=collate_fn,
        drop_last=True
    )

    val_dataloader = DataLoader(
        validation_data,
        batch_size=2,
        shuffle=False,
        num_workers=6,
        collate_fn=collate_fn,
        drop_last=True
    )

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True)
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.00001, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=4, gamma=0.6)
    num_epochs = 30
    all_losses = []
    for epoch in range(num_epochs):
        print("-" * 50)
        print(f"Epoch --> {epoch+1} / {num_epochs}")
        train_loss = train_loop_fn(
            training_dataloader,
            model,
            optimizer,
            device,
            train_data_length,
            scheduler)
        print('training Loss: {:.4f}'.format(train_loss))
        valid_loss = eval_loop_fn(
            val_dataloader, model, device, validation_data_length)
        print('validation Loss: {:.4f}'.format(valid_loss))
        all_losses.append(valid_loss)
    model_name = get_model_name('models')
    print('='*10, 'Saving the model', '='*10)
    torch.save(model, model_name)


if __name__ == "__main__":
    main()
