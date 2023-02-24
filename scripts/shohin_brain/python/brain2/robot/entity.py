# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

import numpy as np
import os

GEOMETRY = -1  # Object geometry in the scene
OBJECT = 0     # Just an object, something with a movable base
JOINT = 1      # Joint with a surface associated
SURFACE = 2    # Surface without a joint associated

class RobotEntity(object):
    """
    This should store the default interface for robot entities and all their
    associated stuff.
    """
    
    def __init__(self, name, object_id, interface, physical=True, verbose=0, mobile=False,
            entity_type=OBJECT):
        
        self.entity_type = entity_type
        self.id = object_id
        self.name = name
        self.interface = interface
        self.is_physical = physical
        self.mobile = mobile

        # Pose information
        self.pose = None

        # Shared data
        self.home_q = None

        # Surfaces 
        self.surfaces = {}
        self.surface_ids = {}

        # Affordances
        self.affordances = {}

        # Inverse kinematics
        self.ik_solver = None
        self.ee_ref = None

    def set_ee_ref(self, ee_ref):
        self.ee_ref = ee_ref

    def set_ik_solver(self, ik_solver):
        self.ik_solver = ik_solver

    def add_affordance(self, name, poses):
        self.affordances[name] = poses

    def get_affordance(self, name):
        return self.affordances[name]

    def set_home_config(self, q):
        self.home_q = q

    # @parameter
    def home_config(self):
        return self.home_q

    def set_mobile(self, mobile):
        """Tracks if this robot has a moving base"""
        self.mobile = mobile

    def get_surface(self, name):
        """ Return a reference to a surface or region that we can place stuff
        on top of"""
        if name in self.surfaces:
            return self.surfaces[name]
        raise RuntimeError('surface ' + str(name) + ' does not exist')

    def add_surface_reference(self, name, reference):
        self.surfaces[name] = reference
        self.surface_ids[name] = reference.id

    # ============================================================================================
    # These things must be implemented!
    def add_surface(self, surface):
        """Attach semantics for placing on top of other objects.
        Must provide:
        - update() to fix surface pose in backend
        - overlap_check() to see if things are in contact
        """
        raise NotImplementedError()
