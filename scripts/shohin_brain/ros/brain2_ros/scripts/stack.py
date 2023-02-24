#!/usr/bin/env python

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

import argparse
import numpy as np
import os
import pdb
import rospy
import sys
import time
import timeit
import pybullet as pb

from datetime import datetime

# ---------------
# ROS imports -- sensor measurement from brain2 ros
from brain2_ros.hand_sensor import HandSensor
from brain2_ros.grasp_sensor import GraspSensor
from brain2_ros.multi_pose_sensor import MultiPoseSensor
from brain2_ros.tf_sensor import TfSensor

# Other ROS tools
from brain2_ros.utils import *
from brain2_ros.robot import *
from brain2_ros.video import VideoCapture

import brain2.utils.color as color

from brain2.robot.domain import CartBlocksDomainDefinition
from brain2.task.search import TaskPlanner
from brain2.utils.info import logwarn, logerr, say, log, start_log, end_log
import brain2.utils.parser as parsing

# Task and motion planning
import brain2.motion_planners.task as tamp

# Leonardo setup
from brain2.domains.leonardo import TaskLeonardo
from brain2.domains.leonardo import GetLeonardoProblems

# FOR DEBUGGING
from brain2.bullet.ik import BulletIKSolver


def parse_args():
    """ Get command-line arguments """
    parser = argparse.ArgumentParser()
    parsing.addPlanningArgs(parser)
    parsing.addRobotArgs(parser)
    parser.add_argument("--use_human", action="store_true", help="should "
                            "we have a human in the scene?")
    parser.add_argument("--test_input", action="store_true", help="run in a "
                            "loop to test inputs; no planning")
    parser.add_argument("--problem", type=str, help="problem to try",
                        choices=GetLeonardoProblems() + [""])
    parser.add_argument("--open_loop", action="store_true",
                        help="move to poses open loop")
    parser.add_argument("--joint_limit_padding", type=float, default=0.1,
                        help="padding to add to joint limits to make sure "
                        "positions do not get too hard for Lula.")
    return parser.parse_args()


def run(args):
    """ Run the stacking test """
    rospy.init_node('brain2_robot_test')

    if args.seed is not None:
        np.random.seed(args.seed)
    visualize = args.visualize > 0
    assets_path = args.assets_path
    ik = get_ik_solver(args)
    domain = CartBlocksDomainDefinition(ik_solver=ik,
                                        visualize=args.visualize,
                                        assets_path=args.assets_path,
                                        padding=args.joint_limit_padding,
                                        hands=args.use_human).compile()
    ctrl, observer = setup_robot_control(domain, ik)
    
    # Add sensors
    # Register sensors
    if args.use_human:
        observer.add_sensor(HandSensor, "right", "k4a/right_hand_crop_center",
                            max_obs_age=args.max_grasp_age)
        observer.add_sensor(HandSensor, "left", "k4a/left_hand_crop_center",
                            max_obs_age=args.max_grasp_age)
        #observer.add_sensor(GraspSensor, "obj", "/k4a/grasps_sampling_%s" % args.hand,
        #                    "/k4a/object_crop_center",
        #                    domain.grasps,
        #                    max_obs_age=args.max_grasp_age)
        # Checks gripper connection
    tf_cfg = {
            "red_block": "posecnn/mean/01_block_red_median_01",
            "blue_block": "posecnn/mean/01_block_blue_median_01",
            "green_block": "posecnn/mean/01_block_green_median_01",
            "yellow_block": "posecnn/mean/01_block_yellow_median_01",
            }
    observer.add_sensor(TfSensor, tf_cfg, reset_orientation=True)
    observer.validate(domain.iface)

    if args.test_input:
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            ws = observer.observe()
            domain.iface.update(ws)
            rate.sleep()
    else:
        # Sample and create world for planning
        planner = tamp.TaskAndMotionPlanner(domain, verbose=0)
        code, (objs, goal, lang) = TaskLeonardo(domain.iface, args.seed,
                                                problem=args.problem)
        
        # Execution 
        rate = rospy.Rate(10)
        ws = observer.observe(blocking=False)
        domain.verbose = True

        # Plan and execute
        print("GOAL =", goal)
        policy = planner.get_policy(ws, goal, max_time=999.)
        if policy is None:
            raise RuntimeError('Planning failed')

        if args.open_loop:
            policy.execute_open_loop(ws)
        else:
            # If we found a plan, try to execute it now
            policy.enter(ws)
            t0 = timeit.default_timer()
            while not rospy.is_shutdown():
                ws = observer.observe(blocking=False)
                # t, grasps, scores = domain.grasps.objs["obj"]
                domain.iface.update(ws)
                print(">>> Left hand seen?", ws["left"].observed)
                print(">>> Right hand seen?", ws["right"].observed)
                # print(">>> Object in hand seen?", ws["obj"].observed)

                # Record some information
                res = policy.step(ws)
                op, op_args = policy.get_current_op()
                current_op = op.name
                log("Switched to op", current_op, "at", timeit.default_timer() - t0)
                if res < 0:
                    logwarn("Getting new plan. Current one is no longer valid.")
                    # No validaction
                    policy.exit()
                    log("Replanned at", timeit.default_timer() - t0)
                    replan_attempts += 1
                    policy = task_planner.get_policy(ws, goal)
                    if policy is None:
                        raise RuntimeError('planning failed; you have a problem!')
                    policy.enter(ws)
                elif res > 0:
                    print("--------------------------------")
                    print("|    EXITING EXECUTION LOOP    |")
                    print("--------------------------------")
                    break
                rate.sleep()


if __name__ == '__main__':
    run(parse_args())

    import signal
    pid = os.getpid()
    os.kill(pid, signal.SIGKILL)
