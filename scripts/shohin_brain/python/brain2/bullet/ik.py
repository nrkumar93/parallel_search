# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from __future__ import print_function

from brain2.utils.info import logwarn, logerr
from collections import namedtuple

import brain2.utils.axis as axis
import brain2.utils.transformations as tra
import pybullet as pb
import pybullet_data

import numpy as np

from brain2.robot.ik import AbstractIKSolver


class BulletIKSolver(AbstractIKSolver):
    """Simple wrapper for bullet inverse kinematics."""

    def __init__(self, domain=None, iterations=100):
        super(BulletIKSolver, self).__init__(domain)
        self._iter = iterations

    def __call__(self, actor_ref, pose, q0=None, pose_as_matrix=True, tol=1e-4, verbose=False):
        # Set up pose objects
        if pose_as_matrix:
            quat = tra.quaternion_from_matrix(pose)
            pos = pose[:3,3]
        else:
            #pos = pose[:3]
            #quat = pose[3:]
            pos, quat = pose
        
        # Initialize
        if q0 is None:
            q0 = actor_ref.sample_uniform()
            actor_ref.set_joint_positions(q0)
        else:
            actor_ref.set_joint_positions(q0)

        # Main inverse kinematics loop
        for i in range(self._iter):
            res = pb.calculateInverseKinematics(bodyIndex=actor_ref.id, 
                                                endEffectorLinkIndex=actor_ref.ee_idx,
                                                targetPosition=pos,
                                                targetOrientation=quat,
                                                lowerLimits=actor_ref.min,
                                                upperLimits=actor_ref.max,
                                                jointRanges=actor_ref.range,
                                                restPoses=q0,
                                                physicsClientId=actor_ref.client,)

            # Get the correct dimensions
            res = np.array(res[:actor_ref.active_min.shape[0]])
            min_mask = res < actor_ref.active_min
            max_mask = res > actor_ref.active_max
            res[min_mask] = actor_ref.active_min[min_mask]
            res[max_mask] = actor_ref.active_max[max_mask]
            if res is not None:
                q0 = np.array(res[:actor_ref.dof])
            else:
                return None
            actor_ref.set_joint_positions(q0)
            p, _ = actor_ref.get_ee_pose(matrix=False)
            error = np.linalg.norm(pos - p)
            if error < tol:
                return q0

        error = np.linalg.norm(pos - p)
        if error < tol:
            return q0
        elif verbose:
            logwarn("IK failed. err = " + str(error))
        return None

