# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

from brain2.robot.ik import AbstractIKSolver
import brain2.utils.axis as axis
import brain2.utils.transformations as tra

from trac_ik_python.trac_ik import IK

import numpy as np


class TracIKSolver(AbstractIKSolver):
    """Simple wrapper for bullet inverse kinematics."""

    def __init__(self, base_link, ee_link, dof, domain=None, timeout=0.05):
        super(TracIKSolver, self).__init__(domain)
        self._base_link = base_link
        self._ee_link = ee_link
        self._timeout = timeout
        self._dof = dof
        self.ik_solver = IK(self._base_link,
                            self._ee_link,
                            timeout=self._timeout,
                            solve_type="Distance")

    def __call__(self, actor_ref, pose, q0=None, pose_as_matrix=True):
        # Set up pose objects
        if pose_as_matrix:
            rot = tra.quaternion_from_matrix(pose)
            pos = pose[:3,3]
        else:
            # TODO: consistency
            pos, rot = pose

        if q0 is not None:
            q0 = q0[:self._dof]
        else:
            q0 = actor_ref.get_joint_positions()

        # Get a solution
        result = self.ik_solver.get_ik(
            qinit=q0,
            x=pos[0],
            y=pos[1],
            z=pos[2],
            rx=rot[0],
            ry=rot[1],
            rz=rot[2],
            rw=rot[3],)
        if result is not None:
            return np.array(result)
        else:
            return None
