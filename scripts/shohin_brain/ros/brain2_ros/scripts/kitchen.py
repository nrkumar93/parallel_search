#!/usr/bin/env python
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import rospy
import signal


# import simulation stuff and command line tools
import brain2_ros.parse as parse
from brain2.robot.kitchen import *
from brain2_ros.sim_manager import SimManager

# Domain definiiton

# Create domain definition through the backend and load assets
import brain2.bullet.problems as problems

if __name__ == '__main__':
    rospy.init_node("test_grasping")
    args = parse.parse_kitchen_args(sim=1, lula=0, lula_opt=0)
    # Create backend for planning
    env = problems.franka_kitchen_right(problems.default_assets_path, args.gui,
            "d435", add_ground_plane=False, load_ycb=True)

    # Create simulation manager
    man = SimManager(args, env, get_task=TaskSortIndigo)

    if args.iter <= 0:
        while not rospy.is_shutdown():
            man.do_random_trial(reset=False)
            rospy.sleep(0.5)
    else:
        for _ in range(args.iter):
            man.do_random_trial(reset=False)
            rospy.sleep(0.5)

    # Kill Lula
    pid = os.getpid()
    os.kill(pid, signal.SIGKILL)
