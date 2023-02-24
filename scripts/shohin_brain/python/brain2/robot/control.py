# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import print_function

class RobotControl(object):
    """ Controls robot arm. This is a pretty abstract interface. """

    def __init__(self, arm, gripper, base=None):
        """ Create object to contain arm + gripper stuff """
        self.arm = arm
        self.gripper = gripper
        self.base = base

        self.close_gripper = self.gripper.close
        self.open_gripper = self.gripper.open
        self.execute_joint_trajectory = self.arm.execute_joint_trajectory
        self.go_local = self.arm.go_local
        self.forward_kinematics = self.arm.forward_kinematics
        self.visualize_trajectory = self.arm.visualize_trajectory
        self.go_config = self.arm.go_config
        
        self.reset()

    def reset(self):
        """ Clear plans for control. """
        self.approach_pose = None
        self.approach_obj = None

    def set_approach(self, obj, pose):
        """ Not sure if this is really how we want to do it. But this will
        tell us how we were planning to grasp an object. """
        self.approach_pose = pose
        self.approach_obj = obj

    def get_approach(self, obj):
        """
        When planning a motion we can set a pose for the next policy to follow.
        """
        if obj == self.approach_obj:
            return self.approach_pose
        else:
            return None

    def forward(self, q):
        """ forward kinematics """
        raise NotImplementedError('must provide forward kinematics function')

    def close_gripper(self, *args, **kwargs):
        raise NotImplementedError()

    def open_gripper(self, *args, **kwargs):
        raise NotImplementedError()

    def execute_joint_trajectory(self, path, timings=None):
        raise NotImplementedError()

    def visualize_trajectory(self, robot_state, path, timings):
        raise NotImplementedError()

    def go_local(self, T, q=None, wait=False):
        """
        Move to six-dof pose in base coordinates given by T
        """
        raise NotImplementedError()

    def go_config(self, q, wait=False):
        raise NotImplementedError()


