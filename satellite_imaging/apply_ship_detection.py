
######################################################################################
# @FileName             apply_ship_detection.py
# @Brief                Applies trained ship segmentation model to satellite images
######################################################################################

import numpy as np
import pandas as pd
from datetime import date, timedelta
import json
import subprocess
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import segmentation_models_pytorch as smp

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.metrics.classification import Accuracy, ConfusionMatrix, F1, ROC, FBeta, Precision, Recall
from pytorch_lightning.metrics import functional as FM

if(torch.cuda.is_available() == True):
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from rasterio.mask import mask as rio_mask
import shapely
import pyproj
from pyproj import Proj, transform
import geopandas as gpd
import folium
import rasterio as rio
from affine import Affine

import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

class IoUBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoUBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = 1 - (intersection + smooth)/(union + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        IoU_BCE = BCE + IoU
                
        return IoU_BCE

class ResNet_Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        pretrained_model = smp.DeepLabV3Plus(encoder_name='resnet152', encoder_depth=5, encoder_weights='imagenet')
        self.model = pretrained_model
        self.sigmoid = nn.Sigmoid()

        self.iou_loss = IoUBCELoss()
        
        self.train_acc = pl.metrics.Accuracy()
        self.train_f1 = pl.metrics.F1()
        self.train_f2 = pl.metrics.FBeta(num_classes = 2, beta = 2)
        self.val_acc = pl.metrics.Accuracy()
        self.val_f1 = pl.metrics.F1()
        self.val_f2 = pl.metrics.FBeta(num_classes = 2, beta = 2)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['targets']
        outputs = self.model(x)
        
        loss = self.iou_loss(outputs, y)
        
        acc = self.train_acc(outputs, y)
        f1 = self.train_f1(outputs, y)
        f2 = self.train_f2(outputs, y)
        self.log('train_acc', acc, prog_bar = True)
        self.log('train_f1', f1, prog_bar = True)
        self.log('train_f2', f2, prog_bar = True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['targets']
        outputs = self.sigmoid(self.model(x))
        acc = self.val_acc(outputs, y)
        f1 = self.val_f1(outputs, y)
        f2 = self.val_f2(outputs, y)
        self.log('val_acc', acc, prog_bar = True)
        self.log('val_f1', f1, prog_bar = True)
        self.log('val_f2', f2, prog_bar = True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        return optimizer

def restrict_range(rgb_img):
    band_min = [110, 135, 150]
    band_max = [4528, 4728, 4992]
    new_rgb_img = np.zeros(rgb_img.shape, dtype = np.uint8)
    def in_range(gray_img_val, min_val, max_val):
        if(gray_img_val > max_val):
            return 255
        elif(gray_img_val < min_val):
            return 0
        else:
            return int((gray_img_val - min_val) / (max_val - min_val) * 255)
    for band in range(3):
        new_rgb_img[:, :, band] = np.vectorize(in_range)(rgb_img[:, :, band], band_min[band], band_max[band])
    return new_rgb_img

class Model_Applicator:
    def __init__(self, sentinel_folder):
        self.model = ResNet_Model()
        self.model.load_state_dict(torch.load('./Pretrained_Weights/trained_model_DeepLabv3Plus_ResNet152_IoUBCELoss.pth'))
        self.model.freeze()
        self.sentinel_folder = sentinel_folder
    
    def extract_satellite_img(self):
        R10 = f'./Scenes/{self.sentinel_folder}'
        b4 = rio.open(R10+f'/B04_10m.jp2', driver = 'JP2OpenJPEG')
        b3 = rio.open(R10+f'/B03_10m.jp2', driver = 'JP2OpenJPEG')
        b2 = rio.open(R10+f'/B02_10m.jp2', driver = 'JP2OpenJPEG')
        with rio.open(f'./Scenes/{self.sentinel_folder}/RGB.tiff', 'w', driver='Gtiff', width=b4.width, height=b4.height, count=3, crs=b4.crs, transform=b4.transform, dtype=b4.dtypes[0]) as rgb:
            rgb.write(b4.read(1), 1)
            rgb.write(b3.read(1), 2)
            rgb.write(b2.read(1), 3)
            rgb.close()
    
    def get_img_coordinates(self):
        with rio.open(f"Scenes/{self.sentinel_folder}/RGB.tiff") as r:
            T0 = r.transform  # upper-left pixel corner affine transform
            p1 = Proj(r.crs)
            tiff_img = r.read().transpose((1, 2, 0))  # pixel values
        
        # Get affine transform for pixel centres
        T1 = T0 * Affine.translation(0.5, 0.5)

        # ul, ur, bl, br
        x_box = [0, tiff_img.shape[1], 0, tiff_img.shape[1]]
        y_box = [0, 0, tiff_img.shape[0], tiff_img.shape[0]]
        coords = [(x_box[idx], y_box[idx]) * T1 for idx in range(len(x_box))]
        x_coords = [x[0] for x in coords]
        y_coords = [x[1] for x in coords]

        p2 = Proj(proj = 'latlong', datum = 'WGS84')
        x_coords, y_coords = transform(p1, p2, x_coords, y_coords)
        coords = list(zip(x_coords, y_coords))
        return coords, tiff_img
    
    def process_img(self, img):
        img = restrict_range(img)
        img = Image.fromarray(img)
        saturation_filter = ImageEnhance.Color(img)
        img = np.asarray(saturation_filter.enhance(2))
        gamma = 3
        img = (((img.astype(np.float32) / 255.0) ** (1 / gamma)) * 255).astype(np.uint8)
        return img
    
    def batch_preprocessor(self, batch):
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]
        batch = (batch - 0) / (255 - 0)
        batch = np.divide(np.subtract(batch, means), stds)
        batch = batch.transpose((0, 3, 1, 2))
        batch = torch.from_numpy(batch).float()
        return batch
    
    def apply_model(self, batch):
        return (nn.Sigmoid()(self.model(batch)).cpu().numpy()[:, 0, :, :] > 0.999).astype(np.uint8)