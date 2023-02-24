# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from __future__ import print_function

import numpy as np
import brain2.utils.transformations as tra
import brain2.utils.axis as axis

class BasicCost(object):
    def __init__(self, dist=0.2, cartesian_wt=1., rotation_wt=0.2, verbose=False, dim_wts=None):
        self.dist = dist
        self.cartesian_wt = cartesian_wt
        self.rotation_wt = rotation_wt
        self.verbose = verbose

        if dim_wts is None:
            self.dim_wts = np.array([1, 1, 10])
        elif dim_wts.shape[0] != 3:
            raise RuntimeError(
                'provided invalid dimension weights with shape ' + str(dim_wts.shape))
        else:
            self.dim_wts = dim_wts

        self.offset = np.eye(4)
        self.offset[2,3] = dist

    def __call__(self, current_pose, current_inv_pose, obj_pose, pose=None,
            ref=None, verbose=False):
        # Compute metric for all grasps
        if pose is not None:
            fwd_pose = obj_pose.dot(pose).dot(self.offset)
        else:
            fwd_pose = obj_pose.dot(self.offset)
        fwd_xyz = fwd_pose[:3, axis.POS] * self.dim_wts
        current_xyz = current_pose[:3, axis.POS] * self.dim_wts
        cart_dist = np.linalg.norm(current_xyz - fwd_xyz)
        dT = current_inv_pose.dot(fwd_pose)
        theta, _, _ = tra.rotation_from_matrix(dT)
        metric = self.cartesian_wt * cart_dist + self.rotation_wt * abs(theta)
        if self.verbose:
            print("METRIC", self.cartesian_wt, cart_dist, self.rotation_wt,
                  theta, "=", metric)
        return metric


class PlaceCost(BasicCost):
    def __call__(self, current_pose, current_inv_pose, obj_pose, pose=None,
            ref=None, verbose=False):
        #fwd_pose = obj_pose.dot(pose).dot(self.offset)
        #fwd_pose = obj_pose.dot(pose)
        fwd_pose = pose
        fwd_dim =  (-1 * fwd_pose[axis.Z, axis.POS]) + obj_pose[axis.Z, axis.POS]
        if verbose:
            print("place pose =", pose[:3, 3])
            print("fwd pose =", fwd_pose[:3, 3])
            print("obj pose =", obj_pose[:3, 3])
            print("in heur", fwd_pose[2,3], obj_pose[2,3])
            print("heur z", -1 * fwd_pose[axis.Z, axis.POS], obj_pose[axis.Z,
                  axis.POS])
        # place must be on top - so fwd_pose must be greater than obj pose
        if fwd_dim > -0.0001:
            return None

        dT = current_inv_pose.dot(fwd_pose)
        theta, _, _ = tra.rotation_from_matrix(dT)
        metric = self.cartesian_wt * fwd_dim + self.rotation_wt * abs(theta)
        if self.verbose or verbose:
            print("[PLACE HEURISTIC] Z-METRIC", self.cartesian_wt, fwd_dim,
                  self.rotation_wt, theta, "=", metric)
        return metric
