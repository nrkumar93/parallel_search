# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

import numpy as np
import time

import brain2.utils.transformations as tra
import brain2.utils.status as status
from brain2.task.action import Policy
from brain2.utils.info import logerr, logwarn
from brain2.robot.trajectory import retime_trajectory

from brain2.motion_planners.rrt_connect import rrt_connect
from brain2.motion_planners.problem import MotionPlanningProblem
from brain2.utils.extend import simple_extend

# Conditions
from brain2.conditions.position import ObjPoseCondition


def LinearPlan(robot_state, pose1, pose2, ik_solver, step_size=0.01,
                rotation_step_size=0.2, q_init=None, *args, **kwargs):
    """
    Get a linear motion plan connecting pose1 and pose2, where both poses are in SE(3). 
    :param robot_state: reference to current entity state
    :param pose1: first pose in the linear plan
    :param pose2: second pose, final pose.
    """
    # Get positions and orientations as quaternions
    p1 = pose1[:3, 3]
    p2 = pose2[:3, 3]
    q1 = tra.quaternion_from_matrix(pose1)
    q2 = tra.quaternion_from_matrix(pose2)
    if q_init is None:
        config = robot_state.q
    else:
        config = q_init

    theta, _, _, = tra.rotation_from_matrix(tra.inverse_matrix(pose1).dot(pose2))

    # Compute a reasonable number of steps for the motion
    steps = np.ceil(max(np.linalg.norm(p2 - p1) / step_size,
                    abs(theta) / rotation_step_size))
                
    if steps <= 1:
        config = ik_solver(robot_state.ref, (p2, q2), q0=config, pose_as_matrix=False)
        if config is None:
            return None
        else:
            return [config], np.zeros(1)

    # then interpolate between positions and final quaternion positions
    dp = p2 - p1
    trajectory = []
    for t in np.linspace(0, 1, steps):
        # Interpolate in position space
        p = p1 + (dp * t)
        # Smooth interpolation in quaternion space from start to end
        q = tra.quaternion_slerp(q1, q2, t)
        config = ik_solver(robot_state.ref, (p, q), q0=config, pose_as_matrix=False)
        if config is None or not robot_state.ref.validate(config, *args, **kwargs):
            # linear planning failed; do something more clever
            return None
        else:
            trajectory.append(np.array(config))

    return retime_trajectory(robot_state, trajectory)

def JointSpaceLinearPlan(robot_state, q1, q2, step_size=0.05, *args, **kwargs):
    """ Get a plan in joint space, for getting to a preferred waypoint or a 
    nice home position."""

    # Compute a reasonable number of steps for the motion
    steps = np.ceil(np.abs(np.max(q2 - q1)) / step_size)
    if steps <= 1:
        # return retime_trajectory(robot_state, [q1, q2])
        return [q2], np.zeros(1)

    # then interpolate between positions and final quaternion positions
    dq = q2 - q1
    trajectory = []
    for t in np.linspace(0, 1, steps):
        # Interpolate in position space
        config = q1 + (dq * t)
        if config is None or not robot_state.ref.validate(config, *args, **kwargs):
            # linear planning failed; do something more clever
            return None
        else:
            trajectory.append(np.array(config))

    trajectory, timings = retime_trajectory(robot_state, trajectory)
    return trajectory, timings


