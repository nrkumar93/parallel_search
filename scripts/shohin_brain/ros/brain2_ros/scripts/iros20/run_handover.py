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
import timeit
from datetime import datetime

# Bullet-specific stuff
import brain2.bullet.problems as problems
import brain2.utils.axis as axis
from brain2.utils.info import logwarn, logerr, say, log, start_log, end_log
import brain2.utils.status as status
import brain2.utils.transformations as tra

# Task planning stuff
from brain2.task.search import TaskPlanner

# Motion tools; policies and stuff
from brain2.motion_planners.rrt import rrt
from brain2.motion_planners.rrt_connect import rrt_connect

# General domain tools
from brain2.utils.pose import make_pose
from brain2.robot.cube import CubeGraspLookupTable
from brain2.robot.domain import CartBlocksDomainDefinition
from brain2.policies.grasping import get_relative_goal_discrete_sampler

# --------------
# ROS imports
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
import tf
import tf2_ros

# ---------------
# ROS imports -- sensor measurement from brain2 ros
from brain2_ros.hand_sensor import HandSensor
from brain2_ros.multi_pose_sensor import MultiPoseSensor
from brain2_ros.control import CreateLulaControlInterface
from brain2_ros.robot import *

# -----------
# inverse kinematics options
from brain2_ros.trac_ik_solver import TracIKSolver
# About half the sleep of the Track-IK option
# from brain2.bullet.ik import BulletIKSolver

# -------------------
# Brain2 policies and conditions
from brain2.policies.gripper import BlockingOpenGripper, BlockingCloseGripper
from brain2.policies.planned import RelativeGoPlanned
from brain2.policies.planned import WaypointGoPlanned
from brain2.policies.planned import BlockingGraspObject
from brain2.conditions.approach import ApproachRegionCondition

import cProfile

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
    return parser.parse_args()

def run(args):

    filename = "handover_" + datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    filename += "_user=" + str(args.user)
    filename += "_experiment=" + str(args.name)
    filename += ".txt"
    start_log(filename)

    rospy.init_node('brain2_robot_test')
    visualize = True

    # Create world model, IK solver, and control interfaces
    env = problems.franka_cart_blocks(args.assets_path, args.visualize, "d435", True)
    ik = TracIKSolver(base_link="base_link",
                      ee_link="right_gripper",
                      dof=7)
    ctrl = CreateLulaControlInterface(ik)

    # ============================================
    # DEFINE SOME SYMBOLS
    # Set up planning domain
    domain = CartBlocksDomainDefinition(env, ctrl, hands=True)
    hand_appr_dist = -0.2 # How far away we go to stand-off from the hand
    approach_region = ApproachRegionCondition(approach_distance=2*hand_appr_dist,
                                              approach_direction=axis.Z,
                                              verbose=False,
                                              slope=10.,
                                              pos_tol=2e-2,
                                              max_pos_tol=5e-2,
                                              theta_tol=np.radians(30))
    domain.add_relation("in_approach_region", approach_region, domain.robot, domain.hands)

    drop_q = np.array([-0.5148817479066682, 0.5074429247002853,
        0.28036785539827846, -2.1822740914544294, 0.019065026521645986,
        3.13, 0.35])
    def at_drop(ws, x):
            x = ws[x]
            return None if x.q is False else np.all(np.abs(x.q[:7] - drop_q[:7]) < 0.1)
    domain.add_property("at_drop", at_drop, domain.robot)
    # In case we get too close to a hand, withdraw
    def too_close(ws, x, hand):
        if not ws[x].observed or not ws[hand].observed: return False
        pt1 = ws[x].ee_pose
        pt2 = ws[hand].pose
        if pt1 is None or pt2 is None: return False
        # Check for a distance slightly smaller than approach distance
        dist = np.linalg.norm(pt1[:3, axis.POS] - pt2[:3, axis.POS])
        limit = (abs(hand_appr_dist) - 0.03)
        return dist < limit
    domain.add_relation("too_close_to_hand", too_close, domain.robot, domain.hands)

    # ----------------
    # Go to a drop position with an object.
    actor = domain.robot
    go_to_drop = WaypointGoPlanned(config=drop_q, step_size=0.25)
    domain.add_operator("go_to_drop",
            policy=go_to_drop,
            preconditions=[
                    ("has_anything(%s)" % actor, True),
                    ("at_drop(%s)" % actor, False)
                    ],
            effects=[("at_drop(%s)" % actor, True)],
            task_planning=True,)
    # Drop the object at the drop position.
    drop_object = BlockingOpenGripper()
    domain.add_operator("drop_object",
            policy=go_to_drop,
            preconditions=[
                  ("at_drop(%s)" % actor, True),
                  ("has_anything(%s)" % actor, True)],
            effects=[("has_anything(%s)" % actor, False)],
            task_planning=True,)
    # Open gripper if it was empty.
    open_gripper = BlockingOpenGripper()
    domain.add_operator("open_gripper_grasp_failed",
                        policy=open_gripper,
                        preconditions=[
                            ("has_anything(%s)" % actor, True),
                            ("gripper_fully_closed(%s)" % actor, True)],
                        effects=[
                            ("has_anything(%s)" % actor, False),
                            ("gripper_fully_closed(%s)" % actor, False)],
                        task_planning=True,)
    # Wait for the human to present a block with a particular hand.
    # Just go home. Can be interrupted.
    go_home = WaypointGoPlanned(step_size=0.25)
    # NOTE: this operator doesn't make sense.
    domain.add_operator("wait_for_object",
                        policy=go_home,
                        preconditions=[("observed({})", False)],
                        effects=[("observed({})", True)],
                        task_planning=True,
                        to_entities=domain.manipulable_objs)
    domain.add_operator("wait_for_human_with_object",
                        policy=go_home,
                        preconditions=[("hand_over_table({})", False)],
                        effects=[("hand_over_table({})", True),
                                 ("stable({})", True),
                                 ("has_anything(%s)" % actor, False),
                                 ("hand_has_obj({})", True)],
                        to_entities=domain.hands,
                        task_planning=True,)
    # Withdraw if we are too close to a hand
    domain.add_operator("retreat_from_hand",
                        policy=go_home,
                        preconditions=[
                            # ("has_anything(%s)" % actor, False),
                            ("in_approach_region(%s, {})" % actor, False),
                            ("too_close_to_hand(%s, {})" % actor, True)],
                        effects=[
                            ("too_close_to_hand(%s, {})" % actor, False),
                            ("at_home(%s)" % actor, True)],
                        to_entities=domain.hands,
                        task_planning=True,)

    # Get a plan and go to offset. Planning can't be interrupted but the motion
    # actually can be, if you leave the space.
    approach_offset = np.eye(4)
    approach_offset[2, 3] = hand_appr_dist
    t0 = np.eye(4) # Default grasp
    t1 = tra.euler_matrix(0, 0, np.pi) # grasp, flipped around z axis
    ty1 = tra.euler_matrix(0, np.pi/4, 0) # grasp, shift around y axis
    ty2 = tra.euler_matrix(0, -1*np.pi/4, 0) # grasp, shift around y axis
    ty3 = tra.euler_matrix(0, np.pi/8, 0) # grasp, shift around y axis
    ty4 = tra.euler_matrix(0, -1*np.pi/8, 0) # grasp, shift around y axis
    home_q = np.array([0., -0.9, -0.12, -2.5, 0., 2.0, 0.68,])
    dz = [t0, t1] # Adding t1 will let it flip the gripper over
    dy = [t0, ty1, ty2, ty3, ty4]
    grasps = []
    for a in dz:
        for b in dy:
            grasps.append(a.dot(b))
    q_weights = np.array([0.25, 0.5, 0.5, 1., 1., 2., 5.])
    #weighted_dist_from_home = lambda q: ((home_q - q) * (home_q - q)).dot(q_weights)
    def weighted_distance(q0, q):
        return ((0.1 * (home_q - q) * (home_q - q)).dot(q_weights)
                + (1.0 * (q0 - q) * (q0 - q)).dot(q_weights))
    plan_approach = RelativeGoPlanned(approach_offset,
                                     goal_offsets=grasps,
                                     step_size=0.05,
                                     #q_metric=weighted_dist_from_home)
                                     q_metric=weighted_distance)
    # Take the object. Blocking; no new observations within state.
    grasp_from_hand = BlockingGraspObject(step_size=0.05)

    # DONE DEFINING SYMBOLS
    # ============================================
    domain.compile() # Create operators and conditions for all objects

    rate = rospy.Rate(10)
    observer = RosWorldStateObserver(domain,
                                     root="base_link",
                                     base_link="measured/base_link",
                                     ee_link="measured/right_gripper",
                                     camera_link="depth_camera_link",
                                     gripper=ctrl.gripper)

    posecnn_config = {}
    instances = ["00"]
    size = "median"
    all_colors = ["red", "green", "blue", "yellow"]
    for color in all_colors:
        for instance in instances:
            center_tf = "01_block_%s_%s" % (color, size)
            accurate_tf = "poserbpf/00_block_green_median_%s" % instance
            name = "%s_block" % (color)
            posecnn_config[name] = {}
            posecnn_config[name]["center_tf"] = center_tf
            posecnn_config[name]["accurate_tf"] = accurate_tf

    # Register sensors
    observer.add_sensor(MultiPoseSensor, posecnn_config) # TODO
    observer.add_sensor(HandSensor, "right", "k4a/right_hand_status")
    observer.add_sensor(HandSensor, "left", "k4a/left_hand_status")
    observer.validate(env) # Checks gripper connection

    # Reference to observer's world state fork
    world_state = observer.observe(blocking=False)
    def check(conditions, verbose=False):
        return domain.check(world_state, conditions, verbose)

    if not args.trial:

        task_planner = TaskPlanner(domain)
        plan = None

        # Goal is to believe all objects are on the table.
        #goal_conditions = [("on_table_top(%s_block)" % color, True) for color in all_colors]
        goal_conditions = [("on_table_top(%s_block)" % color, True)
                for color in ["red"]]

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():

            t0 = timeit.default_timer()
            # Get information about our world
            world_state = observer.observe(blocking=False)
            t1 = timeit.default_timer()

            # Visualize current world state in rospy
            env.update(world_state)

            # If plan is none... get a new plan.
            if plan is None:
                plan = task_planner.solve(world_state, goal_conditions)

            rate.sleep()
 

    elif args.trial:
        # -------------------------
        # Section for collecting data, running the user study
        actor = "robot"
        goal = hand = "right"
        other_hand = "left"
        rate = rospy.Rate(10)
        prev_policy = None
        policy = None
        i = 0
        num_approach_attempts = 0
        num_grasp_attempts = 0
        num_drops = 0
        approach_t, first_approach_t, human_t = None, None, None
        durations = []
        first_durations = []
        human_durations = []
        cur_hand_shape = None
        count_shapes = {}
        count_shapes_successful = {}
        while not rospy.is_shutdown():
            i += 1

            log("Num approach attempts:", num_approach_attempts)
            log("Num grasp attempts:", num_grasp_attempts)
            log("Num drops:", num_drops)
            log("Time since last approach:", durations)
            log("Time since first approach in cycle:", first_durations)
            log("Time since human first seen over table:", human_durations)
            log("Hand shapes seen:")
            for k, v in count_shapes.items():
                vs = count_shapes_successful[k] if k in count_shapes_successful else 0
                log("-", k, "seen =", v, "successful =", vs)

            # Get information about our world
            t0 = timeit.default_timer()
            world_state = observer.observe(blocking=False)
            t1 = timeit.default_timer()

            if human_t is None:
                if check([("hand_over_table(%s)" % hand, True)]):
                    human_t = rospy.Time.now()
            # Get hand shape if relevant
            hand_shape = world_state[hand].hand_shape
            log("CURRENT HAND SHAPE:", hand_shape)

            # Visualize current world state in rospy
            env.update(world_state)

            #print("--------------------------------")
            #print("home =", check([("at_home(%s)" % actor, True)]))
            #print("observed =", check([("observed(%s)" % hand, True)]))
            #print("stable =", check([("stable(%s)" % hand, True)]))
            #print("over table =", check([("hand_over_table(%s)" % hand, True)]))
            #print("has a plan  =", check([("has_plan(%s, %s)" % (actor, hand), True)]))
            #print("in appr region =", check([("in_approach_region(%s, %s)" % (actor, hand), True)]))
            #print("hand has obj =", check([("hand_has_obj(%s)" % hand, True)]))
            #print("has anything =", check([("has_anything(robot)", True)]))
            #print("has obj =", check([("has_obj(robot, blue_block)", True)]))
            #continue

            # Simple RLDS
            trajectory = None
            close_gripper = False
            msg = None

            print("avoiding =", check([("too_close_to_hand(%s, %s)" % (actor, hand),
                True)]))

            # --------------------------------------------------------------
            # This block lists preconditions for all the different actions, and
            # is in reverse priority order.
            if check([ # ("hand_over_table(%s)" % hand, False),
                      ("at_drop(%s)" % actor, True),
                      ("has_anything(%s)" % actor, True)]):
                msg = "Drop the block."
                policy = drop_object
            elif check([("has_anything(%s)" % actor, False),
                        ("gripper_fully_closed(%s)" % actor, True)]):
                msg = "Grasp attempt failed."
                policy = open_gripper
            elif check([("in_approach_region(%s, %s)" % (actor, hand), False),
                        ("too_close_to_hand(%s, %s)" % (actor, hand), True)]):
                msg = "Avoiding human hand for safety!"
                policy = go_home
            elif check([("too_close_to_hand(%s, %s)" % (actor, other_hand), True)]):
                msg = "Avoiding other human hand for safety!"
                policy = go_home
            elif check([("has_anything(%s)" % actor, True),
                        ("at_drop(%s)" % actor, False)]):
                msg = "go to drop position"
                policy = go_to_drop
            elif check([("hand_over_table(%s)" % hand, False),
                      ("at_home(%s)" % actor, False)]):
                # go to home
                msg = "Going home."
                policy = go_home
            elif check([("hand_over_table(%s)" % hand, False),
                      ("at_home(%s)" % actor, True)]):
                msg = "Hand not over table, waiting at home..."
                policy = go_home
            elif check([("hand_over_table(%s)" % hand, True)]):
                if check([("stable(%s)" % hand, False)]):
                    msg = "WAITING FOR HAND POSITION TO STABILIZE"
                    policy = None
                elif check([("has_plan(%s, %s)" % (actor, hand), True),
                            ("in_approach_region(%s, %s)" % (actor, hand), True),
                            ("has_anything(%s)" % actor, False)]):
                    msg = "Try to grasp (blocking)"
                    policy = grasp_from_hand
                elif check([("hand_has_obj(%s)" % hand, False)]):
                    msg = "WAITING FOR HAND WITH OBJECT"
                    policy = None
                elif check([("hand_over_table(%s)" % hand, True),
                            ("stable(%s)" % hand, True),
                            ("has_anything(%s)" % actor, False),
                            ("hand_has_obj(%s)" % hand, True)]):
                    msg = "Move to approach"
                    policy = plan_approach
            # --------------------------------------------------------------

            # Check that we found something to do
            if msg is None:
                logerr("Not sure what to do!")
                raise RuntimeError("INVALID LOGICAL STATE")

            # Tell the user what's going on
            say(msg)

            # Core RLDS logic:
            # enter or exit as necessary to "clean up" policy objects and cache
            # things needed for execution
            # execute one step of the policy
            if policy is not None:
                # Core RLDS logic
                if policy is not prev_policy:
                    if prev_policy is not None:
                        prev_policy.exit(world_state, actor, goal)
                    policy.enter(world_state, actor, goal)

                    # -------------------------------------
                    # Data collection logic
                    # Get time and collect some metrics
                    t = rospy.Time.now()
                    if policy == drop_object:
                        num_drops += 1
                        durations.append((t - approach_t).to_sec())
                        first_durations.append((t - first_approach_t).to_sec())
                        human_durations.append((t - human_t).to_sec())
                        approach_t = None
                        first_approach_t = None
                        # On enter drop, count successful shape
                        if hand_shape not in count_shapes_successful:
                            count_shapes_successful[hand_shape] = 1
                        else:
                            count_shapes_successful[hand_shape] += 1
                        human_t = None
                    elif policy == grasp_from_hand:
                        num_grasp_attempts += 1
                    elif policy == plan_approach:
                        num_approach_attempts += 1
                        approach_t = t
                        if first_approach_t is None:
                            first_approach_t = t
                        # On enter approach, count shape
                        if hand_shape not in count_shapes:
                            count_shapes[hand_shape] = 1
                        else:
                            count_shapes[hand_shape] += 1
                    # -------------------------------------

                    prev_policy = policy
                policy.step(world_state, actor, goal)
            else:
                if prev_policy is not None:
                    prev_policy.exit(world_state, actor, goal)
                prev_policy = None

            # Sleep and wait for next step
            rate.sleep()

    # Clean up log at end
    end_log()

if __name__ == '__main__':
    #cProfile.run('run()')
    run(parse_args())
