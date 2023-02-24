# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function, division, absolute_import

import os
import math
import numpy as np
from carbongym import gymapi
from carbongym import gymutil



class SharedData:
    """
    For loading information from the experiment class. Stored data used by all
    possible problems.
    """
    def __init__(self, gym, sim, **kwargs):
        self.gym = gym
        self.sim = sim

        franka = kwargs.get("urdf", "urdf/franka_description/robots/gym_franka_panda.urdf")
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True
        asset_options.armature = 0.01
        self.franka_asset = self._load_asset(franka, asset_options, **kwargs)
        if self.franka_asset is None:
            raise IOError("Failed to load franka asset")
        asset_options.collapse_fixed_joints = True
        asset_options.flip_visual_attachments = True
        mobile_urdf = "urdf/franka_description/robots/franka_carter_d435.urdf"
        self.mobile_asset = self._load_asset(mobile_urdf, asset_options, **kwargs)
        
        # Load the object file
        obj_list_file = kwargs.get("object_list", "object_list.yaml")

        # Load kitchen URDF
        self.kitchen_asset = self._load_asset("urdf/kitchen_description/urdf/kitchen_part_right_gen_convex.urdf")
        self.cabinet_asset = self._load_asset("urdf/sektion_cabinet_description/urdf/sektion_cabinet.urdf")
        self.table_asset = self._load_asset("urdf/simple_table.urdf")

        if self.franka_asset is None:
            raise IOError("Failed to load franka asset")
        
        # Load the object file
        obj_list_file = kwargs.get("object_list", "object_list.yaml")

    def _load_asset(self, asset_file, opts=None, **kwargs):
        asset_root = kwargs.get("data_dir", "./assets")
        if opts is None:
            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            asset_options.flip_visual_attachments = False
            # asset_options.armature = 0.01
            asset_options.thickness = 0.0005
            asset_options.armature = 1e-4
            asset_options.max_angular_velocity = 50.
            asset_options.flip_visual_attachments = False
            asset_options.density = 500
            asset_options.collapse_fixed_joints = True
        else:
            asset_options = opts

        print("... Loading asset '%s' from '%s'" % (asset_file, asset_root))
        return self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
