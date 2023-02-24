# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


#!/usr/bin/env python

import rospy
from brain2_ros.simulation import Simulation

if __name__ == '__main__':
    rospy.init_node('test_sim_node')
    sim = Simulation()
    sim.random_scene(["potted_meat_can", "tomato_soup_can"])
    raw_input('Press enter')
    sim.random_scene(["cracker_box"])
    raw_input('Press enter')
    sim.random_scene(["potted_meat_can", "tomato_soup_can", "mustard_bottle"])
