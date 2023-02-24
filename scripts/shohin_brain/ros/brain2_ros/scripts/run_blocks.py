# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


#!/usr/bin/env python

import numpy as np
import rospy

from brain2.task.world_state import WorldStateObserver

# Bullet-specific stuff
import brain2.bullet.problems as problems
from brain2.bullet.interface import BulletInterface
from brain2.bullet.ik import BulletIKSolver
import brain2.utils.axis as axis
from brain2.utils.info import logwarn, logerr
import brain2.utils.status as status

# Motion tools; policies and stuff
from brain2.policies.planned import SimplePlannedMotion
from brain2.motion_planners.rrt import rrt
from brain2.motion_planners.rrt_connect import rrt_connect
from brain2.policies.linear import linear_plan

# General domain tools
from brain2.utils.pose import make_pose
from brain2.robot.cube import CubeGraspLookupTable
from brain2.robot.domain import CartObjectsDomainDefinition
from brain2.policies.grasping import get_relative_goal_discrete_sampler

# --------------
# ROS imports
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
import tf
import tf2_ros

import household_poses as household

from hand_sensor import HandSensor
from multi_pose_sensor import MultiPoseSensor
from franka_gripper_control import FrankaGripperControl
from lula_franka_control import LulaFrankaControl

# -------------------
# Brain2 robot imports
from trajectory import spline_parameterization
from robot import *

assets_path = "/home/pittsburgh/src/brain_gym/assets/urdf"
if __name__ == '__main__':
    rospy.init_node('brain2_robot_test')
    visualize = True

    # Create interfaces with real robot
    gripper = FrankaGripperControl()
    franka = LulaFrankaControl(base_link="base_link",
                               ee_link="right_gripper",
                               dof=7,
                               sim=False)
    gripper.open()

    # Create world model
    env = problems.franka_cart_objects(assets_path, visualize, "d435", pb_obj_config)
    # Get IK solver
    ik = BulletIKSolver()

    domain = CartBlocksDomainDefinition(env, ik, domain_obj_config)

    rate = rospy.Rate(10)
    observer = RosWorldStateObserver(domain,
                                     root="base_link",
                                     base_link="measured/base_link",
                                     ee_link="measured/right_gripper")

    posecnn_config = {}
    instance = ["00"]
    size = "median"
    for color in ["red", "green", "blue", "yellow"]:
        for instance in instances:
            center_tf = "01_block_{color}_{size}".format()
            accurate_tf = "poserbpf/00_block_green_median_{instance}".format()
            name = "{size}_block_{color}".format()
            posecnn_config[name] = {}
            posecnn_config[name]["center_tf"] = center_tf
            posecnn_config[name]["accurate_tf"] = accurate_tf


