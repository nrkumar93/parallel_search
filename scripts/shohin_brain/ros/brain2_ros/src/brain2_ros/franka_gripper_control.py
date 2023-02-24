# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import print_function

import rospy
import actionlib
import numpy as np
from franka_gripper.msg import (GraspAction, GraspGoal, GraspEpsilon,
                                MoveAction, MoveGoal)
from sensor_msgs.msg import JointState

from brain2.utils.pose import make_pose
from brain2.utils.info import logwarn, logerr
import brain2.utils.transformations as tra
from actionlib_msgs.msg import GoalStatus

class FrankaGripperControl(object):
    """Real robot gripper control"""
    def __init__(self):
        action_client = "/franka_gripper/grasp"
        logwarn("Creating gripper action client " + action_client)
        self.grasp_client = actionlib.SimpleActionClient(action_client, GraspAction)
        self.grasp_client.wait_for_server()
        action_client = "/franka_gripper/move"
        logwarn("Creating gripper move client " + action_client)
        self.move_client = actionlib.SimpleActionClient(action_client, MoveAction)
        self.move_client.wait_for_server()
        # self.index = 8
        self.last_client = None

    def get_gripper_offsets(self):
        """ gets information about the gripper -- basically, how it should be
        oriented with respect to a canonical orientation frame (aka, grasping
        along positive z-axis, with the gripper z axis)."""
        t0 = tra.euler_matrix(np.pi*-1, 0, np.pi)
        t1 = tra.euler_matrix(np.pi*-1, 0, np.pi).dot(tra.euler_matrix(0, 0, np.pi))
        return [t0, t1]

    def update_gripper_state(self, entity_state):
        g = entity_state.gripper_state
        entity_state.gripper_fully_closed = g < 0.0004 #1e-3
        entity_state.gripper_fully_open = g > 0.039
        #entity_state.gripper_moving = (self.last_client.get_result() is None
        entity_state.gripper_moving = (self.last_client.get_state() <= GoalStatus.ACTIVE)

    def close(self, width=0., speed=0.1, force=10., eps=(0.2, 0.2), wait=True):
        """Send a close command"""

        goal = GraspGoal()
        goal.width = width
        goal.speed = speed
        goal.force = force
        goal.epsilon.inner = eps[0]
        goal.epsilon.outer = eps[1]
        self.grasp_client.send_goal(goal)

        if wait:
            self.grasp_client.wait_for_result()

        self.last_client = self.grasp_client
        res = self.grasp_client.get_result()
        return res is not None and res.success

    def move(self, width, speed=0.1, wait=True):
        """Move to a specific pose"""
        goal = MoveGoal()
        goal.width = width
        goal.speed = speed
        self.move_client.send_goal(goal)

        if wait:
            self.move_client.wait_for_result()

        res = self.move_client.get_result()
        self.last_client = self.move_client
        return res is not None and res.success

    def open(self, speed=0.1, wait=True):
        """Send a predefined Franka open command"""
        return self.move(width=0.08, speed=speed, wait=wait)

class LulaFrankaGripperControl(FrankaGripperControl):
    """ version of the interface designed to work with lula """

    def __init__(self, franka):
        # Save Lula end effector interface
        self.ee = franka.end_effector

    def update_gripper_state(self, entity_state):
        g = entity_state.gripper_state
        entity_state.gripper_fully_closed = g < 1e-3
        entity_state.gripper_fully_open = g > 0.039
        entity_state.gripper_moving = False

    def close(self, width=0., speed=0.1, force=10., eps=(0.2, 0.2), wait=True):
        controllable_object = None
        self.ee.gripper.close(
            controllable_object,
            wait=wait,
            speed=speed,
            actuate_gripper=True,
            force=force,)

    def open(self, speed=0.1, wait=True):
        self.ee.gripper.open(speed=speed, wait=wait)

    def move(self, *args, **kwargs):
        raise NotImplementedError()


class SimFrankaGripperControl(FrankaGripperControl):
    """ Create interface to a simulated gripper for use with or without lula. """

    joints = ["panda_finger_joint1", "panda_finger_joint2"]
    open_positions = [0.04, 0.04]
    closed_positions = [0., 0.]

    def __init__(self):
        self.gripper_cmd_pub = rospy.Publisher("interp/gripper", JointState,
                                               queue_size=1)

    def close(self, width=0., speed=0.1, force=10., eps=(0.2, 0.2), wait=True):
            self.gripper_cmd_pub.publish(JointState(name=self.joints,
                                                    position=self.closed_positions))

    def open(self, speed=0.1, wait=True):
        self.gripper_cmd_pub.publish(JointState(name=self.joints,
                                                position=self.open_positions))

    def move(self, width, speed=0.1, wait=True):
        self.gripper_cmd_pub.publish(JointState(name=selfjoints,
                                                position=[width] * len(self.joints)))

    def update_gripper_state(self, entity_state):
        g = entity_state.gripper_state
        entity_state.gripper_fully_closed = g < 1e-3
        entity_state.gripper_fully_open = g > 0.039
        entity_state.gripper_moving = False


if __name__ == '__main__':
    rospy.init_node('open_franka_gripper')
    gripper = FrankaGripperControl()
    #gripper.open(wait=False)
    gripper.close(wait=False)
    rate = rospy.Rate(10)

    while True:
        ok = gripper.last_client.get_state()
        print("Moving =", ok <= GoalStatus.ACTIVE)

        res = gripper.last_client.get_result()
        print("Result =", res)

        if res is not None:
            break

        rate.sleep()
