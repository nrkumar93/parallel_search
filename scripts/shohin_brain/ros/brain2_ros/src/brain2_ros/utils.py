# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from __future__ import print_function

import copy
import numpy as np
import rospy

from brain2.utils.pose import make_pose


def make_pose_from_msg(msg):
    trans = (msg.transform.translation.x,
             msg.transform.translation.y,
             msg.transform.translation.z,)
    rot = (msg.transform.rotation.x,
           msg.transform.rotation.y,
           msg.transform.rotation.z,
           msg.transform.rotation.w,)
    return make_pose(trans, rot)


def make_pose_from_pose_msg(msg):
    trans = (msg.pose.position.x,
             msg.pose.position.y, msg.pose.position.z,)
    rot = (msg.pose.orientation.x,
           msg.pose.orientation.y,
           msg.pose.orientation.z,
           msg.pose.orientation.w,)
    return make_pose(trans, rot)


def make_pose_from_unstamped_pose_msg(msg):
    trans = (msg.position.x,
             msg.position.y,
             msg.position.z,)
    rot = (msg.orientation.x,
           msg.orientation.y,
           msg.orientation.z,
           msg.orientation.w,)
    return make_pose(trans, rot)

from moveit_msgs.msg import RobotState
from moveit_msgs.msg import DisplayTrajectory
from moveit_msgs.msg import RobotTrajectory

def make_display_trajectory_pub():
    display_pub = rospy.Publisher('/display_planned_path', DisplayTrajectory, queue_size=1)
    return display_pub

def show_trajectory(publisher, joint_trajectory, q0):

    # create messages
    base_frame = "base_link"
    display_trajectory = DisplayTrajectory()
    display_trajectory.trajectory_start = RobotState()
    display_trajectory.trajectory_start.joint_state.name = joint_trajectory.joint_names
    display_trajectory.trajectory_start.joint_state.position = q0
    robot_state_trajectory = RobotTrajectory(joint_trajectory=joint_trajectory)

    robot_state_trajectory.joint_trajectory.header.frame_id = base_frame
    display_trajectory.trajectory.append(robot_state_trajectory)
    publisher.publish(display_trajectory)


