# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


#!/usr/bin/env python

import rospy
from brain2_ros.franka_gripper_control import FrankaGripperControl

if __name__ == '__main__':
    rospy.init_node('open_franka_gripper')
    gripper = FrankaGripperControl()
    gripper.open()
