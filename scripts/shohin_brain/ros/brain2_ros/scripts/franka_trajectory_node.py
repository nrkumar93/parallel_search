# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


#!/usr/bin/env python

from __future__ import print_function

import argparse
import numpy as np
import os
import rospy

from brain2_ros.franka_gripper_control import FrankaGripperControl
from brain2.robot.domain import CartObjectsDomainDefinition
import brain2.bullet.problems as problems
from brain2_ros.trac_ik_solver import TracIKSolver
from brain2_ros.robot import RosWorldStateObserver

rospy.init_node('relay_commands_to_franka_node')

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
    return parser.parse_known_args()[0]

# Create interfaces with real robot
gripper = FrankaGripperControl()
gripper.open()

# Create world model, IK solver, and control interfaces
args = parse_args()
env = problems.franka_cart(args.assets_path, args.visualize, "d435",
        padding=0.0)
# WE really should not be using this
base_link = "panda_link0"
ee_link = "panda_hand"
ik = TracIKSolver(base_link=base_link,
                  ee_link=ee_link,
                  dof=7)

# ============================================
# DEFINE SOME SYMBOLS
# Set up planning domain
domain = CartObjectsDomainDefinition(env)
domain.compile()
prefix = "measured/"
prefix = ""
observer = RosWorldStateObserver(domain,
                                 root=prefix + base_link,
                                 base_link=prefix + base_link,
                                 ee_link=prefix + ee_link,
                                 camera_link="depth_camera_link",
                                 gripper=gripper)

#def _cb(msg):
#    print(msg.position)
#    arm.go_local(q=msg.position)

from sensor_msgs.msg import JointState

from brain2.robot.trajectory import retime_trajectory
from brain2.robot.trajectory import compute_derivatives
from brain2.motion_planners.linear import LinearPlan
from brain2.motion_planners.linear import JointSpaceLinearPlan
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import actionlib
from control_msgs.msg import FollowJointTrajectoryAction
from control_msgs.msg import FollowJointTrajectoryActionGoal
from control_msgs.msg import FollowJointTrajectoryGoal

client = actionlib.SimpleActionClient("/position_joint_trajectory_controller"
                                      "/follow_joint_trajectory",
                                      FollowJointTrajectoryAction)

#home_q = np.array([0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4])
#initial_configuration:
#  joint_state:
#    - { name: panda_joint1,       position: 0.0 }
#    - { name: panda_joint2,   position: 0.0 }
#    - { name: panda_joint3,  position: 0.0 }
#    - { name: panda_joint4, position: -1.0 }
#    - { name: panda_joint5,     position: 0.0 } #-0.2 }
#    - { name: panda_joint6,   position: 1.5 }
#    - { name: panda_joint7,     position: 0.0 } #-0.2 }
#home_q = np.array([0, 0, 0, -1., 0., 1.5, 0])
#home_q = np.array([0., -0.785, 0., -2.356, 0., 1.571, 0.785])
#home_q = np.array([0.1560555699455867,
#                   -1.306638358404831,
#                   -1.11498810450236,
#                   -2.2451363150613366,
#                   -0.3784684199359682,
#                   1.5615516125625295,
#                   -0.5023804819997814])
#home_q = np.array([0.20480990034029004,
#                    -1.5462639688859905,
#                    -1.0964815305036872,
#                    -2.2834671904246013, -0.43910167417918655,
#                    1.5600739196274014, -0.2640529223543448])
#home_q = np.array([0.19943905853615498, -0.6206339396677519,
#    -0.6428357117217883, -1.794293103268272, -0.014381015713785401,
#    1.488322779682169, 0.3773888988519837])
home_q = np.array([0.19861893887283372, -0.6224963690888742, -0.6457779602524669,
        -1.7994142279373306, -0.017759000219624198, 1.491777432536225,
        0.37158449923994646])

print("Connecting to action server...")
client.wait_for_server()


def follow_trajectory(robot, times, traj, pos, vel, acc):
    msg = JointTrajectory()
    msg.joint_names = joint_names
    # Add safety limit to the robot's range
    # APPLY PADDING?
    qmin, qmax = (robot.ref.active_min,
                  robot.ref.active_max)
    # For debugging purposes
    # print(qmin, qmax)
    for opos, t in zip(traj, times):
        pt = JointTrajectoryPoint()
        q = pos(t)
        #q = opos
        if (np.any(q < qmin) or np.any(q > qmax)):
            print(q)
            print(qmin)
            print(qmax)
            raise RuntimeError()
        q[:7] = np.clip(q[:7], qmin, qmax)
        pt.positions = q
        pt.velocities = vel(t)
        pt.accelerations = np.clip(acc(t), -1*np.pi, np.pi)
        pt.time_from_start = rospy.Duration(t)
        print("---", t, "---")
        print(pt.positions)
        print(pt.velocities)
        print(pt.accelerations)
        msg.points.append(pt)

    # raw_input("[ READY ]")
    goal = FollowJointTrajectoryGoal(trajectory=msg)
    client.send_goal(goal)
    client.wait_for_result()
    return client.get_result()

joint_names = ["panda_joint%d" % i for i in range(1,8)]

speed_factor = 0.8
acc_fract = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.25, 0.5])

def go_home():
    world_state = observer.observe(blocking=False)
    q0 = world_state["robot"].q
    qg = home_q
    res = JointSpaceLinearPlan(world_state["robot"], q0, qg,
                               step_size=0.1,
                               acceleration_fraction=acc_fract)
    if res is None:
        raise RuntimeError('could not go home directly for some reason')
    traj, times = res
    if len(traj) == 1:
        # No motion necessary
        return None
    pos, vel, acc = compute_derivatives(traj, times, scale=speed_factor)
    return follow_trajectory(world_state["robot"], times, traj, pos, vel, acc)

reprofile = False
def exec_trajectory(msg):
    global reprofile
    robot = env.get_object("robot")
    if reprofile:
        world_state = observer.observe(blocking=False)
        q0 = world_state["robot"].q
        qg = home_q
        traj = [np.array(pt.positions) for pt in msg.points]
        traj, times = retime_trajectory(world_state["robot"], traj,
                acceleration_fraction=acc_fract)
        if len(traj) == 1:
            # No motion necessary
            return None
        pos, vel, acc = compute_derivatives(traj, times, scale=speed_factor)
        return follow_trajectory(world_state["robot"], times, traj, pos, vel, acc)
    else:
        goal = FollowJointTrajectoryGoal(trajectory=msg)
        for pt in msg.points:
            print("check", pt.positions, "=", robot.validate(q=pt.positions))
        # raw_input('----')
        client.send_goal(goal)
        client.wait_for_result()
        return client.get_result()

if __name__ == '__main__':

    import threading
    lock = threading.Lock()
    trajectory = None

    def callback(msg):
        global trajectory
        trajectory = msg

    sub = rospy.Subscriber('joint_trajectory_franka', JointTrajectory, callback)

    moving = False
    gone_home = False

    from std_srvs.srv import Empty, EmptyResponse
    def reset_go_home(req=None):
        global gone_home
        gone_home = False
        return EmptyResponse()
    s = rospy.Service('go_home', Empty, reset_go_home)

    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        # Get a world state
        world_state = observer.observe(blocking=False)
        # print("Current config:", world_state['robot'].q)
        if world_state['robot'].q is None:
            r.sleep(); continue
        elif not gone_home:
            # Check if we've done home yet
            res = go_home()
            print("Result:", res)
            if res is not None and res.error_code != 0:
                break
            gone_home = True
            r.sleep(); continue
        else:
            lock.acquire()
            if trajectory is not None:
                exec_trajectory(trajectory)
                trajectory = None
            lock.release()

        r.sleep()
