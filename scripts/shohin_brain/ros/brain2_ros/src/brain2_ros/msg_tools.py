# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

from moveit_msgs.msg import RobotState

def makeRobotStateMsg(robot_state):
    msg = RobotState()
    msg.joint_state.name = robot_state.ref.get_joint_names()
    msg.joint_state.position = robot_state.q
    return msg
