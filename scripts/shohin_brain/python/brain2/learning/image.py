# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

import os
import json
import numpy as np
import h5py
import brain2.utils.image as img
import cv2
from brain2.utils.info import logerr

import json
import matplotlib.pyplot as plt
from matplotlib import image
import pdb
from tqdm import tqdm

def crop_center(img, cropx, cropy):
    y = img.shape[0]
    x = img.shape[1]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]


def resize_image(rgb):
    rgb = crop_center(rgb, 720, 720)
    rgb = cv2.resize(rgb, dsize=(224, 224))
    return rgb


def resize_images(source, destination):

    if not os.path.exists(destination):
        os.makedirs(destination)

    files = [f for f in os.listdir(source)
            if f[0] != '.' and f.endswith('.h5')]

    for i, f in enumerate(tqdm(files, ncols=50)):
        filename = f
        num = int(filename[4:-3])

        data = h5py.File(os.path.join(source, filename), 'r')
        all_data = {}

        for akey in data.keys():
            all_data[akey] = data[akey][()]

        for akey in ['rgb', 'seg', 'depth']:
            all_data[akey] = []
        try:
            seq_length = data['q'][()].shape[0]
            rgbs = data['rgb'][()]
            segs = data['seg'][()]
            depths = data['depth'][()]
        except KeyError as e:
            logerr("Could not handle file " + str(filename) + ":" + str(e))
            continue

        for j in range(seq_length):
            rgb = img.PNGToNumpy(rgbs[j])
            seg = img.PNGToNumpy(segs[j])
            depth = img.PNGToNumpy(depths[j])
            # crop and resize
            rgb = crop_center(rgb, 720, 720)
            rgb = cv2.resize(rgb, dsize=(224, 224))
            seg = crop_center(seg, 720, 720)
            seg = cv2.resize(seg, dsize=(224, 224))
            depth = crop_center(depth, 720, 720)
            depth = cv2.resize(depth, dsize=(224, 224))

            for k in ['rgb', 'seg', 'depth']:
                all_data[k].append(img.GetPNG(eval(k)))

        with h5py.File(os.path.join(destination, filename), 'w') as f:
            for k in all_data.keys():
                f.create_dataset(k, data=all_data[k])
        f.close()
