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
from brain2.robot.lookup_table import BasicLookupTable

class CubeGraspLookupTable(BasicLookupTable):
    """
    This defines the lookup table for picking up cubes.
    """

    def __init__(self, approach_distance=-0.1, cage_distance=-0.03,
                 offset_axis=axis.Z, use_angled=True, *args, **kwargs):
        self.grasps = []

        approach_offset_z, cage_offset_z = np.eye(4), np.eye(4)
        approach_offset_z[axis.Z, 3] = approach_distance
        cage_offset_z[axis.Z, axis.POS] = cage_distance

        for x in range(4):
            for y in range(4):
                for z in range(4):
                    Tx = tra.euler_matrix(np.pi / 2 * x,
                                          np.pi / 2 * y,
                                          np.pi / 2 * z,)

                    self.grasps.append(np.dot(Tx, cage_offset_z))

                    # Add angled grasps
                    if use_angled:
                        Txy = Tx.dot(tra.euler_matrix(0, -1*np.pi/4, 0))
                        self.grasps.append(np.dot(Txy, cage_offset_z))

    def __call__(self, ws, obj):
        return self.grasps

    def has(self, ws, obj):
        if "block" in ws[obj].obj_type or "cube" in ws[obj].obj_type:
            return True


class CubeStackLookupTable(BasicLookupTable):
    """
    This defines the lookup table for picking up cubes.
    """

    def __init__(self, approach_distance=-0.1, cage_distance=-0.03,
                 offset_axis=axis.Z, *args, **kwargs):
        self.grasps = []

        cage_offset_z = np.eye(4)
        cage_offset_z[axis.Z, axis.POS] = cage_distance
        cage_offset_z2 = np.eye(4)
        cage_offset_z2[axis.Z, axis.POS] = -1*cage_distance

        for x in range(4):
            for y in range(4):
                for z in range(4):
                    Tx = tra.euler_matrix(np.pi / 2 * x,
                                          np.pi / 2 * y,
                                          np.pi / 2 * z,)

                    T = np.dot(Tx, cage_offset_z)
                    Tfy = tra.euler_matrix(0, np.pi * y, 0)
                    Tfy = Tx.dot(Tfy).dot(cage_offset_z2)
                    Tfx = tra.euler_matrix(np.pi * x, 0, 0)
                    Tfx = Tx.dot(Tfx).dot(cage_offset_z2)

                    self.grasps += [T, Tfx, Tfy]

                    

    def __call__(self, ws, obj):
        return self.grasps

    def has(self, ws, obj):
        if "block" in ws[obj].obj_type or "cube" in ws[obj].obj_type:
            return True


class CubeAlignLookupTable(BasicLookupTable):
    """
    Defines offsets in each direction from a cube, for finding the right way to put a new cube.
    """
    def __init__(self, side_length=0.05,
                 offset_axis=axis.Z, *args, **kwargs):
        self.grasps = []
        self.side_length = 0.05

        axes = [axis.X, axis.X, axis.Y, axis.Y, axis.Z, axis.Z]
        signs = [1, -1, 1, -1, 1, -1]

        for ax, sign in zip(axes, signs):
            T = np.eye(4)
            T[ax, 3] = sign * self.side_length
            self.grasps.append(T)

    def __call__(self, ws, obj):
        return self.grasps

    def has(self, ws, obj):
        if "block" in ws[obj].obj_type or "cube" in ws[obj].obj_type:
            return True

    def verbs(self):
        return ["align"]


