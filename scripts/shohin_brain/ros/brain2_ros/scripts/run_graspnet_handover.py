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
import time
import timeit
import rospy
import pybullet as pb

from datetime import datetime

# ---------------
# ROS imports -- sensor measurement from brain2 ros
from brain2_ros.hand_sensor import HandSensor
from brain2_ros.grasp_sensor import GraspSensor
from brain2_ros.multi_pose_sensor import MultiPoseSensor
from brain2_ros.robot import *
from brain2_ros.utils import *

import brain2.utils.color as color
from brain2_ros.video import VideoCapture

from brain2.task.search import TaskPlanner
from brain2.domains.handover import *
from brain2.utils.info import logwarn, logerr, say, log, start_log, end_log


def parse_args():
    parser = argparse.ArgumentParser()
    if "HOME" in os.environ:
        default_path = os.path.join(os.environ["HOME"], 'src/brain_gym/assets/urdf')
    else:
        default_path = '../../../assets/urdf',
    parser.add_argument('--assets_path',
        default=default_path,
        help='assets path')
    parser.add_argument('--trial', type=int, default=1,
            help='True if thsi is a trial, false if use task planning')
    parser.add_argument('--visualize', type=int, default=0,
            help='should we also visualize the environment if supported? Note'
                  ' that this will also slow down planning.')
    parser.add_argument('--user', default='NA')
    parser.add_argument('--name', default='NA')
    parser.add_argument("--hand", default="left", choices=["left", "right"])
    parser.add_argument("--max_grasp_age", default=1.5, type=float,
                        help="maximum age grasps are allowed to persist")
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--parallel_ik", action="store_true",
                        help="WARNING still work in progress")
    parser.add_argument("--num_ik_threads", type=int, default=10)
    return parser.parse_args()

def run(args):

    rospy.init_node('brain2_robot_test')
    visualize = args.visualize > 0
    assets_path = args.assets_path
    ik = get_ik_solver(args)

    # Create planning domain
    domain = HandoverDomainDefinition(assets_path, gui=visualize, ik_solver=ik,
                                      hand=args.hand)
    ctrl, observer = setup_robot_control(domain, ik)

    # Create task planner
    task_planner = TaskPlanner(domain)

    # Add sensors
    # Register sensors
    observer.add_sensor(HandSensor, "right", "k4a/right_hand_crop_center",
                        max_obs_age=args.max_grasp_age)
    observer.add_sensor(HandSensor, "left", "k4a/left_hand_crop_center",
                        max_obs_age=args.max_grasp_age)
    observer.add_sensor(GraspSensor, "obj", "/k4a/grasps_sampling_%s" % args.hand,
                        "/k4a/object_crop_center",
                        domain.grasps,
                        max_obs_age=args.max_grasp_age)
    observer.validate(domain.iface) # Checks gripper connection

    # Set the goal for task planning
    goal = [
            ("has_obj(%s, obj)" % (domain.robot), True),
            # ("at_home(%s)" % domain.robot, True),
            ]

    filename = "handover_" + datetime.now().strftime("%Y-%m-%d") + ".mp4"
    video = VideoCapture(filename=filename)
    palette = color.UmberToScarlet

    for trial in range(args.num_trials):

        filename = "handover_" + datetime.now().strftime("%Y-%m-%d")
        filename += "_user=" + str(args.user)
        filename += "_experiment=" + str(args.name)
        filename += "_trial=%02d" % trial
        filename += ".txt"
        start_log(filename)

        # Get world state and run loop
        rate = rospy.Rate(10)
        ws = observer.observe(blocking=False)
        domain.verbose = True
        t0 = rospy.Time.now().to_sec()
        # Plan and execute
        policy = task_planner.get_policy(ws, goal)
        policy.enter(ws)
        t0 = timeit.default_timer()
        grasp_attempts = 0
        approach_attempts = 0
        replan_attempts = 0
        approach_times = []
        approach_t0 = None
        current_op = None
        while not rospy.is_shutdown():
            print("==============================")
            print(">>> goal obj =", ws["robot"].goal_obj)
            ws = observer.observe(blocking=False)
            t, grasps, scores = domain.grasps.objs["obj"]
            domain.iface.update(ws)

            print(">>> Left hand seen?", ws["left"].observed)
            print(">>> Right hand seen?", ws["right"].observed)
            print(">>> Object in hand seen?", ws["obj"].observed)

            # Record some information
            res = policy.step(ws)
            op, op_args = policy.get_current_op()
            if op is not None and op.name != current_op:
                if "approach" in op.name:
                    name = "Approach object"
                    rcol, mcol = palette[4]
                    approach_attempts += 1
                    approach_t0 = timeit.default_timer()
                elif "grasp" in op.name:
                    name = "Grasp object"
                    rcol, mcol = palette[3]
                    grasp_attempts += 1
                    if approach_t0 is not None:
                        approach_times.append(timeit.default_timer() - approach_t0)
                    approach_t0 = None
                else:
                    rcol, mcol = palette[0]
                    name = "Wait for object"
                    approach_t0 = None

                current_op = op.name
                log("Switched to op", current_op, "at", timeit.default_timer() - t0)

            video.annotate(msg=name, rectangle_color=rcol,
                           msg_color=mcol)

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

        rcol, mcol = palette[1]
        video.annotate(msg="Placing object", rectangle_color=rcol,
                       msg_color=mcol)

        policy.exit()
        log("=============")
        log("# approach:", approach_attempts)
        log("# grasp:", grasp_attempts)
        log("# replan:", replan_attempts)
        avg = np.mean(approach_times) if len(approach_times) > 0 else 0
        log("avg approach time:", avg)
        log("approach times:", approach_times)

        # TODO: change this waypoint, or make it not wait all the way
        waypoint_q = [-0.5220034506758843, -0.04347318905426682,
            -0.0535705365407051, -1.8353953491109916, 0.30912330485218104,
            2.0664310570021445, 0.082300612132490]
        ctrl.go_config(q=waypoint_q, speed="med")
        wait_for_q(observer, "robot", waypoint_q, max_t=5.)
        # TODO: Wei can change this to figure out where the robot will put the
        # object
        # - Translation: [0.497, -0.277, 0.162]
        # - Rotation: in Quaternion [-0.006, 0.979, 0.028, 0.200]
        #        in RPY (radian) [3.084, 0.402, 3.141]
        #        in RPY (degree) [176.709, 23.031, 179.992]
        drop_q = [-0.46102238867134443, 0.25313198158197237, -0.02884049947359917,
                -2.3114114993500414, 0.7889157780156415, 2.785062871153136,
                -0.432817044027759]
        ctrl.go_local(q=drop_q, wait=True, speed="med")
        ctrl.open_gripper()
        t_final = timeit.default_timer() - t0
        log("Final time:", t_final)

        rospy.sleep(1.)
        ctrl.go_local(q=ws[domain.robot].ref.home_q)

        end_log()

        rcol, mcol = palette[2]
        video.annotate(msg=None,
                       rectangle_color=rcol,
                       msg_color=mcol)

        # Wait here until we are ready for the next one
        rospy.sleep(5.)

    video.close()

if __name__ == '__main__':
    run(parse_args())

    import signal
    pid = os.getpid()
    os.kill(pid, signal.SIGKILL)
