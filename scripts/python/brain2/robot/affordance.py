# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

import numpy as np

import brain2.utils.status as status
import brain2.utils.transformations as tra

# Import text colors
from brain2.utils.info import *


class AbstractAffordance(object):
    """ Basic pose implementation """

    def __call__(self, batch_size):
        """ Get batch of possible poses """
        raise NotImplementedError()


class DiscreteAffordance(AbstractAffordance):
    """ Gets the set of poses we need out of a fixed list."""

    def __init__(self, poses, invert=False):
        self.poses = poses
        if invert:
            self.poses = [tra.inverse_matrix(pose) for pose in self.poses]

    def __call__(self, batch_size):
        if batch_size > len(self.poses):
            return self.poses
        else:
            idx = np.random.choice(range(len(self.poses)), batch_size)
            return [self.poses[i] for i in idx]


class GraspsAffordance(AbstractAffordance):
    def __init__(self, filename, correction=None):
        # TODO: this is from OMG (Yu Xiang)
        try:
            nvidia_grasp = np.load(filename, allow_pickle=True)
            grasps = nvidia_grasp.item()['transforms']
        except:
            nvidia_grasp = np.load(filename, allow_pickle=True, fix_imports=True, encoding='bytes')
            grasps = nvidia_grasp.item()[b'transforms']
        
        # TODO why do these poses still seem wrong?
        self.poses = [grasps[i].dot(correction) for i in range(grasps.shape[0])]

    def __call__(self, batch_size):
        if batch_size > len(self.poses):
            return self.poses
        else:
            idx = np.random.choice(range(len(self.poses)), batch_size)
            return [self.poses[i] for i in idx]


class H5Affordance(AbstractAffordance):
    """ May not actually need to be different """

    def __init__(self, dataset):
        n, d = dataset.shape
        assert d == 7
        self.poses = np.zeros((n, 4, 4))
        for i in range(n):
            pose = tra.quaternion_matrix(dataset[i,3:])
            pose[:3, 3] = dataset[i, :3]
            self.poses[i] = pose

    def __call__(self, batch_size):
        if batch_size > len(self.poses):
            return self.poses
        else:
            idx = np.random.choice(range(len(self.poses)), batch_size)
            return [self.poses[i] for i in idx]


class InterpolationAffordance(AbstractAffordance):
    """ Sample values between two poses, e.g. on a handle """
    def __init__(self, pose1, pose2):
        self.pose1 = pose1
        self.pose2 = pose2

        p1 = pose1[:3, 3]
        p2 = pose2[:3, 3]
        q1 = tra.quaternion_from_matrix(pose1)
        q2 = tra.quaternion_from_matrix(pose2)
        self.pq1 = (p1, q1)
        self.pq2 = (p2, q2)

    def __call__(self, batch_size):
        # Choose points between two end poses
        val = np.random.random(batch_size)

        p1, q1 = self.pq1
        p2, q2 = self.pq2
        dp = p2 - p1

        # interpolate between them and return the values
        poses = []
        for v in val:
            p = p1 + (dp * v)
            q = tra.quaternion_slerp(q1, q2, v)
            pose = tra.quaternion_matrix(q)
            pose[:3, 3] = p
            poses.append(pose)

        return poses


