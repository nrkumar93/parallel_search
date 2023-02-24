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

from dope_sensor import DopeSensor
from hand_sensor import HandSensor
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

    pb_obj_config = {
            'bbq_sauce': 'BBQSauce',
            'mac_and_cheese': 'MacaroniAndCheese',
            'raisins': 'Raisins',
            'milk': 'Milk',
            }

    # Create interfaces with real robot
    gripper = FrankaGripperControl()
    franka = LulaFrankaControl(base_link="base_link",
                               ee_link="right_gripper",
                               dof=7,
                               sim=False)
    gripper.open()

    # Create world model
    env = problems.franka_cart_objects(assets_path, visualize, "d435",
            pb_obj_config, hands=True)
    # Get IK solver
    ik = BulletIKSolver()
    # Get planning domain
    domain_obj_config = {
            'bbq_sauce': {"obj_type": 'BBQSauce'},
            'mac_and_cheese': {"obj_type": 'MacaroniAndCheese'},
            'raisins': {"obj_type": 'Raisins'},
            'milk': {"obj_type": 'Milk'},
            }
    domain = CartObjectsDomainDefinition(env, ik, domain_obj_config)
    # Get observation function --- and add sensors
    observer = RosWorldStateObserver(domain,
                                     root="base_link",
                                     base_link="measured/base_link",
                                     ee_link="measured/right_gripper")
    observer.add_sensor(DopeSensor, pb_obj_config)
    observer.add_sensor(HandSensor, "right", "k4a/right_hand_status")
    observer.add_sensor(HandSensor, "left", "k4a/left_hand_status")

    # TODO figure out how best to store this information
    gripper_offsets = gripper.get_gripper_offsets()
    grasp_table = household.get_household_poses(gripper_offsets)
    hand = env.get_object("robot_hand")


    goto_home = True
    goto_look = True
    goto_approach = True
    goto_grasp = True
    goto_backoff = True
    drop = True
    standoff_dist = 0.1


    rate = rospy.Rate(10)
    t0 = rospy.Time.now()
    # actor, goal = "robot", "mac_and_cheese"
    # actor, goal = "robot", "raisins"
    # actor, goal = "robot", "bbq_sauce"
    actor, goal = "robot", "milk"

    # ------------
    while not rospy.is_shutdown():
        # Get information about our world
        world_state = observer.observe(blocking=False)
        domain.update_logical(world_state)
        goal_type = world_state[goal].type

        # Visualize what we think the current world looks like
        env.update(world_state)
        if world_state[actor].config is None:
            rate.sleep()
            continue

        print(goal, "obs =", world_state[goal].observed,
              "stable =", world_state[goal].stable)

        # Check time
        t = (rospy.Time.now() - t0).to_sec()
        if t > 0.5:
            if goto_home:
                # First go home
                world_state = observer.observe(blocking=False)
                _sample_goal = lambda: home_q
                problem = domain.get_default_planning_problem(actor, _sample_goal)
                trajectory, tree = rrt_connect(world_state[actor].config, problem)
                msg = spline_parameterization(world_state[actor], trajectory)
                franka.publish_display_trajectory(world_state[actor], msg)
                franka.execute_joint_trajectory(trajectory)
                gripper.open()
                goto_home = False
            elif goto_look:
                _sample_goal = lambda: obs_right_q
                problem = domain.get_default_planning_problem(actor,
                                                              _sample_goal)
                trajectory, tree = rrt_connect(world_state[actor].config, problem)
                if trajectory is None:
                    raise RuntimeError('bad trajectory')
                msg = spline_parameterization(world_state[actor], trajectory)
                franka.publish_display_trajectory(world_state[actor], msg)
                franka.execute_joint_trajectory(trajectory)
                goto_look = False
                t0 = rospy.Time.now()
            elif (goto_approach and world_state[goal].observed
                  and world_state[goal].stable):

                _sample_goal = get_relative_goal_discrete_sampler(ik,
                                                   world_state[actor].ref,
                                                   world_state[goal].pose,
                                                   grasp_table[goal_type],
                                                   debug_entity=hand,
                                                   config=world_state[actor].config,
                                                   standoff=standoff_dist)
                problem = domain.get_default_planning_problem(actor, _sample_goal)
                problem.iterations = 10
                trajectory, tree = rrt_connect(world_state[actor].config, problem)
                if trajectory is None:
                    print("BAD TRAJECTORY -- aborting")
                    continue

                # - ROS display
                msg = spline_parameterization(world_state[actor], trajectory)
                franka.publish_display_trajectory(world_state[actor], msg)

                raw_input("Enter to execute APPROACH")
                franka.execute_joint_trajectory(trajectory)
                goto_approach = False

                # ----------------------------------
                for _ in range(10):
                    world_state = observer.observe(blocking=False,
                                                   entities=[actor])
                    rate.sleep()

                T = np.eye(4)
                T[2,3] = standoff_dist
                final_pose = world_state[actor].ee_pose.dot(T)
                trajectory = linear_plan(world_state[actor],
                                         world_state[actor].ee_pose,
                                         final_pose,
                                         ik,
                                         step_size=0.01)
                if trajectory is None:
                    print("BAD TRAJECTORY -- aborting")
                    continue

                msg = spline_parameterization(world_state[actor], trajectory)
                franka.publish_display_trajectory(world_state[actor], msg)
                raw_input("Enter to execute GRASP")
                franka.execute_joint_trajectory(trajectory)
                goto_grasp = False
                gripper.close()

                standoff_pose = np.copy(final_pose)
                standoff_pose[2,3] += 0.1
                trajectory = linear_plan(world_state[actor],
                                         final_pose,
                                         standoff_pose,
                                         ik,
                                         step_size=0.01)

                msg = spline_parameterization(world_state[actor], trajectory)
                franka.publish_display_trajectory(world_state[actor], msg)
                franka.execute_joint_trajectory(trajectory)
                goto_home = True

        # and done
        rate.sleep()

    # import sys
    # from select import select
    # print("Waiting for keypress...")
    # timeout_s = 30
    # select([sys.stdin], [], [], timeout_s)

