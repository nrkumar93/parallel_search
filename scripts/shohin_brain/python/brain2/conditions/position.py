# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

import numpy as np
import brain2.utils.axis as axis
import brain2.utils.transformations as tra

from brain2.utils.info import logwarn, logerr


class PoseCondition(object):
    """ Defines a volume around a particular pose """

    def __init__(self, pose,
            pos_tol=5e-3,
            theta_tol=np.radians(5.),
            relative=False,
            verbose=False,
            ):

        self.verbose = verbose
        self.pos_tol = pos_tol
        self.theta_tol = theta_tol
        self.pose = pose
        self.relative = relative


    def __call__(self, world_state, x, y=None):
        """ evaluates if x is at the pose. """

        if self.relative:
            if y is None:
                raise RuntimeError('PoseCondition was expecting an argument')
            goal_pose = world_state[y].pose.dot(self.pose)
        else:
            goal_pose = self.pose
        
        actor_pos = world_state[x].ee_pose[:3, axis.POS]
        frame_pos = goal_pose[:3, axis.POS]
        dr = world_state[x].inv_ee_pose.dot(goal_pose)
        theta, _, _ = tra.rotation_from_matrix(dr)
        dist = np.linalg.norm(frame_pos - actor_pos)

        return dist < self.pos_tol and theta < self.theta_tol


class ObjPoseCondition(object):
    """ Defines a volume around a particular pose.
    TODO remove duplicated code. """

    def __init__(self, pose,
            pos_tol=5e-3,
            theta_tol=np.radians(5.),
            relative=False,
            verbose=False,
            ):

        self.verbose = verbose
        self.pos_tol = pos_tol
        self.theta_tol = theta_tol
        self.pose = pose
        self.relative = relative


    def __call__(self, world_state, x, y=None):
        """ evaluates if x is at the pose. """

        if self.relative:
            if y is None:
                raise RuntimeError('PoseCondition was expecting an argument')
            goal_pose = world_state[y].pose.dot(self.pose)
        else:
            goal_pose = self.pose
        
        actor_pos = world_state[x].pose[:3, axis.POS]
        frame_pos = goal_pose[:3, axis.POS]
        dr = world_state[x].inv_pose.dot(goal_pose)
        theta, _, _ = tra.rotation_from_matrix(dr)
        dist = np.linalg.norm(frame_pos - actor_pos)

        if self.verbose:
            print("=========================")
            print("Comparing", x, y)
            print("dist =", dist)
            print("angle =", theta)

        return dist < self.pos_tol and abs(theta) < self.theta_tol


