# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from __future__ import print_function

import math
import copy
import numpy as np
import rospy
import tf2_ros
import timeit
import std_msgs

# Brain2 tools and things
import brain2.utils.axis as axis
import brain2.utils.status as status
import brain2.utils.pose as pose
import brain2.utils.transformations as tra
from brain2.utils.rotation import euler_to_quaternion
from brain2.utils.rotation import quaternion_to_euler

# Presets
import brain2.robot.entity as entity

# Planners
from brain2.task.search import TaskPlanner
from brain2.motion_planners.task import TaskAndMotionPlanner

# Domain definitions and tools
from brain2.robot.domain import KitchenDomainDefinition
from brain2.robot.kitchen import DefineYcbObjects

# Brain2 core ROS utilities
from brain2_ros.robot import RosWorldStateObserver
from brain2_ros.control import CreateLulaControlInterface
from brain2_ros.tf_sensor import TfSensor
from brain2_ros.articulated_obj_sensor import ArticulatedObjInfo
from brain2_ros.articulated_obj_sensor import ArticulatedObjSensor

# Inverse kinematics solvers
from brain2_ros.trac_ik_solver import TracIKSolver
from brain2.bullet.ik import BulletIKSolver

# Basic execution stuff
from brain.execution import PlanExecutionPolicy
from brain_msgs.msg import Predicate, PredicateList
from isaac_bridge.manager import SimulationManager
from isaac_bridge.manager import ros_camera_pose_correction
from lula_control.object import ControllableObject

from sensor_msgs.msg import JointState


class SimManager(object):

    home_q = np.array([0., -0.9, -0.12,
                       -2.5, 0., 2.0, 0.68,])

    def get_robot(self):
        return self.move_arm

    def _create_domain(self):
        """ Create the domain. This will need to be done many times to get different problems for
        planning in. """
        if self._backend is None:
            raise RuntimeError('missing physics backend for planning')

        # TODO: choose objects to use in this planning domain here if need be
        # Default behavior: load kitchen YCB objects
        obj_config = DefineYcbObjects()
        ik = TracIKSolver(base_link="base_link",
                          ee_link="right_gripper",
                          dof=7)
        ik = BulletIKSolver()
        
        ctrl = CreateLulaControlInterface(ik, self._backend.get_object("robot"), sim=self.use_sim, lula=self.use_lula)
        self.domain = KitchenDomainDefinition(self._backend, robot_control=ctrl, objs=obj_config,
                add_blocks=False, ik_solver=ik).compile()
        print(self.domain.root["robot"].ctrl)
        print(self.domain.root["robot"].ref)
        print(self.domain.root["robot"].ref.ik_solver)
        self.move_arm = self.domain.get_robot().get_control_interface()
        self.observer = RosWorldStateObserver(self.domain,
                                     #root="base_link",
                                     root="ue_world",
                                     #base_link="measured/base_link",
                                     base_link="00_chassis_link",
                                     ee_link="measured/right_gripper",
                                     camera_link="rgb_camera_link",
                                     gripper=self.move_arm.gripper)

        self.observer.add_sensor(TfSensor, config=self.all_obj_configs)
        self.observer.add_sensor(ArticulatedObjSensor,
                tracker_topic='/tracker/kitchen/joint_states',
                config=self.art_obj_config)

    def _js_cb(self, msg):
        self.q = msg.position
        self.dq = msg.velocity


    def __init__(self, args, backend, get_task, sim=None, use_sim=True):

        # Set up basic params
        self.use_lula = args.lula > 0
        self.use_sim = use_sim

        # Used to sample environments
        self.get_task = get_task

        if sim is None and self.use_sim:
            self.sim = SimulationManager(lula=self.use_lula)
        else:
            self.sim = sim

        self.all_objs = ["spam", "tomato_soup", "sugar", "cracker_box", "mustard" ]
        self.all_task_objs = ["spam", "tomato_soup", "sugar"]
        self.all_obj_configs = {
                "spam": "00_potted_meat_can",
                "sugar": "00_sugar_box",
                "cracker_box": "00_cracker_box",
                "tomato_soup": "00_tomato_soup_can",
                "mustard": "00_mustard_bottle"}
        self.obj_ue4_handle = {
                "spam": "potted_meat_can_1",
                "sugar": "sugar_box_1",
                "cracker_box": "cracker_box_1",
                "tomato_soup": "tomato_soup_can_1",
                "mustard": "mustard_bottle_1",
                }

        self.art_obj_config = {
                'indigo_top_drawer': ArticulatedObjInfo(
                    name='indigo_top_drawer',
                    tf_frame='indigo_drawer_top',
                    joint_name='indigo_drawer_top_joint'),
                'indigo_bottom_drawer': ArticulatedObjInfo(
                    name='indigo_bottom_drawer',
                    tf_frame='indigo_drawer_bottom',
                    joint_name='indigo_drawer_bottom_joint'),
                'hitman_top_drawer': ArticulatedObjInfo(
                    name='hitman_top_drawer',
                    tf_frame='hitman_drawer_top',
                    joint_name='hitman_drawer_top_joint'),
                }

        self.q = None
        self.dq = None
        self._js_sub = rospy.Subscriber("joint_states", JointState, self._js_cb)

        self._backend = backend
        self._create_domain()
        self.drawers = self.domain.drawers[:2]

        self.args = args

        self.var_theta = {
            "spam": 0.1,
            "tomato_soup": 0.01,
            "sugar": 0.1,
            "cracker_box": 0.5,
            "mustard": 0.5,
        }
        self.obj_offset = {
            "spam": 0.06,
            "tomato_soup": 0.15,
            "sugar": 0.15,
            "cracker_box": 0.15,
            "mustard": 0.15,
        }
        self.region_x_bounds = {
            "indigo_front": [4.3, 4.6],
            "indigo": [4.25, 4.75],
            "dismiss_indigo": [3.7, 4.2],
            "mid_table": [1.6, 2.3],
            "mid_table_far": [1.6, 2.3],
        }
        self.region_y_bounds = {
            "indigo_front": [-8.2, -8.05],
            "indigo": [-8.1, -7.85],
            "dismiss_indigo": [-10.5, -10.],
            "mid_table": [-11.31, -11.01],
            "mid_table_far": [-11.01, -10.51],
        }

        self.pub_label = rospy.Publisher("brain/label", std_msgs.msg.String,
                                         queue_size=1)
        self.pub_task = rospy.Publisher("brain/task", std_msgs.msg.String,
                                        queue_size=1)
        self.pub_status = rospy.Publisher("brain/status", std_msgs.msg.Int32,
                                          queue_size=1)
        self.pub_predicates = rospy.Publisher(
            "brain/predicates", PredicateList, queue_size=1)
        self.pub_seed = rospy.Publisher(
            "brain/seed", std_msgs.msg.Int64, queue_size=1)

        if self.args.linear:
            self.domain.linear = True

        if len(self.args.image) > 0 and len(self.args.image_topic) > 0:
            from image_capture import ImageCapture
            self.img_capture = ImageCapture(topic=self.args.image_topic)
        else:
            self.img_capture = None

        self.ticks_per_sec = 10
        self.rate = rospy.Rate(self.ticks_per_sec)
        self.trials_done = 0

        if self.args.seed is not None:
            self.seed = int(self.args.seed)
        else:
            self.seed = np.random.randint(1000, 4294967295)

        self.target_obj_order = None

    def check(self, poses, distance):
        """
        Determine if configuration is ok
        """
        for k1, pose in poses.items():
            p1 = pose[:2, axis.POS]
            for k2, pose2 in poses.items():
                if k2 == k1:
                    continue
                p2 = pose2[:2, axis.POS]
                dist = np.linalg.norm(p2 - p1)
                # print("CHECKING", k1, k2, dist, p1, p2)
                if dist < distance:
                    # print("collision:", k1, "at", p1, k2, "at", p2)
                    return False
        return True

    def set_camera(self, trans_magnitude=0.0075, rot_magnitude=0.0075,
                   randomize=False):
        """
        Sets camera pose. If randomize is true, will choose a random position
        and orientation near the "correct" one
        """

        # TODO: remove this text
        # tmp default pos at 0611
        # default_camera_trans=np.array([4.83, -8.6, 1.8])
        # default_camera_rot=np.array([0.6293311, 0.66770539, 0.04096442,
        # -0.39551712])

        default_camera_trans = np.array([4.95, -9.03, 2.03])
        default_camera_rot = np.array(
            [0.58246231, 0.72918614, 0.08003926, -0.35016988])

        if randomize:
            camera_trans = default_camera_trans + \
                np.random.randn(3) * trans_magnitude
            camera_rot = default_camera_rot + \
                np.random.randn(4) * rot_magnitude
            camera_rot /= np.linalg.norm(camera_rot)
        else:
            camera_rot = default_camera_rot
            camera_trans = default_camera_trans
        camera_pose = pose.make_pose(camera_trans, camera_rot)
        camera_pose = ros_camera_pose_correction(camera_pose, "zed_left")
        self.sim.set_pose("zed_left", camera_pose, do_correction=False)

    def test_real(self, task):
        while not self.observer.update():
            self.rate.sleep()
        objs, goal = self.get_task(self._backend)
        print("==================================")
        print("OBJS:", objs)
        print("GOAL:\n", goal)
        print("PLAN:\n", plan)
        rospy.sleep(1.)
        start_t = timeit.default_timer()
        res, tries = self.test(goal, plan, ["arm", objs[0]], task)
        print("==================================")
        print("TOOK TIME:", timeit.default_timer() - start_t)
        print("PLANNING TRIES:", tries)
        print("SUCCESS =", res)
        self.trials_done += 1
        return res

    def do_random_trial(self, reset=True, task_name="test"):
        self.move_arm.open_gripper(wait=False)

        # Do not do a smooth, pretty Lula retract, even if that functionality is
        # enabled right now.
        self.move_arm.go_local(q=self.home_q, wait=True)
        if reset and self.use_sim:
            self.sim.reset()

        # WAIT UNTIL OBSERVER SUCCEEDS
        while not self.observer.update():
            print('waiting for success...')
            self.rate.sleep()

        # Vsiaulize the environment
        world_state = self.observer.observe()
        self._backend.update(world_state)
        self.observer.update_camera_from_tf(self._backend)

        # Random sleep to make sure services reconnected properly
        # rospy.sleep(1.)
        if self.use_sim:
            self.sim.wait_for_services()
            rospy.sleep(3.)

        # ----------------------------------------------
        # Generate a random seed
        # Sample a task
        # Configure the environment
        # Apply changes to the world via sim interface
        trial_seed = self.seed + self.trials_done
        print("==================================")
        print("TRIAL SEED =", trial_seed, "starting from",
              self.seed, "+", self.trials_done)

        # Move to a default position that we can use
        self.move_arm.go_local(q=self.home_q, wait=True)
        if trial_seed is not None:
            np.random.seed(trial_seed)
            self.pub_seed.publish(data=trial_seed)

        # Sample a task
        objs, goal = self.get_task(self._backend, trial_seed)

        # Set the camera position
        self.set_camera(randomize=self.args.randomize_camera > 0)
        rospy.sleep(0.1)
        if np.random.randint(2) == 0:
            self.move_arm.close_gripper(wait=False)
        else:
            self.move_arm.open_gripper()

        self.observer.update_camera_from_tf(self._backend)

        # Set object configurations that have been sampled earlier
        for obj_name in self.all_objs:
            obj = self._backend.get_object(obj_name)
            ue_name = self.obj_ue4_handle[obj_name]
            T = obj.get_pose(matrix=True)
            print("setting", obj_name, "ue sim id =", ue_name, "xyz =", T[:3, 3])
            self.sim.set_pose(ue_name, T, do_correction=False)

        # Randomize textures and move to random start position
        self.go_to_random_start(world_state)
        rospy.sleep(0.1)
        if np.random.randint(0, 100) != 0:
            if self.args.randomize_textures > 0:
                self.sim.dr()
            rospy.sleep(0.1)
        # Done randomizing texture and applying to the environment
        # ----------------------------------------------

        print("==================================")
        print("OBJS:", objs)
        print("GOAL:\n", goal)
        start_t = timeit.default_timer()
        res = self.test(goal, ["arm", objs[0]], task_name)
        print("==================================")
        print("TOOK TIME:", timeit.default_timer() - start_t)
        print("SUCCESS =", res)
        self.trials_done += 1
        return res

    def get_current_robot_config(self):
        """ Get the current joint states """
        print("Getting current joint state...")
        while not rospy.is_shutdown():
            if self.q is not None:
                return self.q
            self.rate.sleep()

    def go_to_random_start(self, world_state):
        print("+" * 10 + " ROBOT START STATE RANDOMIZATION " + "+" * 10)
        robot = world_state[self.domain.robot]

        # set up the camera
        #self.observer.update_camera_from_tf(self._backend)

        # Lookup kitchen pose
        #print("Looking up kitchen location")
        #ue4_to_kitchen = self.observer.get_relative_tf("ue_world", "sektion")
        #ue4_to_kitchen = self.observer.get_relative_tf("sektion", "ue_world")
        # kitchen_pose = self.observer.lookup_base_relative_to_frame("sektion")
        #kitchen_pose = world_state["kitchen"].ref.get_pose()
        robot_pose = world_state[self.domain.robot].ref.get_pose()
        #kitchen_to_robot = tra.inverse_matrix(kitchen_pose).dot(robot_pose)

        #print(ue4_to_kitchen)
        #robot_pose = ue4_to_kitchen.dot(kitchen_to_robot)
        self.sim.set_pose("franka_1", robot_pose, do_correction=False)

        # effector or another object
        # trans = np.array([4.2,-8.7,0.235])
        # quat = np.array([0.707,0,0,0.707])

        quat = tra.quaternion_from_matrix(robot.pose)
        trans = robot.ee_pose[:3, axis.POS]
        while True:
            _trans = trans + np.random.randn(3) * 0.2
            print(_trans)
            if ((_trans[0] > 0.45) and (_trans[1] < 0.20) and (_trans[2] < 0.50)
                    and (_trans[1] > -0.40)) is not True:
                break
            if ((_trans[0] > 0.45) and (_trans[2] < 0.25)) is not True:
                break

        quat = quat + np.random.randn(4) * 0.2
        quat = quat / np.linalg.norm(quat)

        T = tra.quaternion_matrix(quat)
        T[:3, axis.POS] = _trans
        self.move_arm.go_local(T, q=self.get_current_robot_config())

    def get_plan(self, goal, plan_args, plan=None):
        """ Find a task and motion plan to achieve the given goal. """
        # problem = TaskPlanner(self.domain, verbose=0)
        problem = TaskAndMotionPlanner(self.domain, verbose=0)
        instantiated_goal = [
            (self.domain.format(g, plan_args[0], plan_args[1]), value)
            for g, value in goal]
        world_state = self.observer.observe()
        self._backend.update(world_state)

        start_t = timeit.default_timer()
        print("\n\n >>> GOAL =")
        print(instantiated_goal)

        if self.args.pause:
            print("About to start planning...")
            raw_input()

        actions, subgoals, trajs = problem.solve(world_state, instantiated_goal)
        t = timeit.default_timer() - start_t
        print([action.opname for action in actions])
        print(">>> Planning took %f seconds" % t)
        if self.args.pause:
            raw_input()
        return actions, subgoals, trajs

    def publish_predicates(self, domain, world_state):
        msg = PredicateList()
        for pred_name, pred_idx in domain.predicate_idx.items():
            val = world_state.logical_state[pred_idx]
            pred = Predicate(name=pred_name, value=val)
            msg.predicates.append(pred)
        self.pub_predicates.publish(msg)

    def time_check(self, world_state):
        """
        Use this to determine if the timing is ok, and the `/tf` topic isn't
        being processed too slowly.
        """
        t0 = timeit.default_timer()
        world_state = self.domain.update_logical(world_state)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        print("t =", timeit.default_timer() - t0)
        try:
            t = self.observer.tf_listener.getLatestCommonTime(
                'panda_link7',
                '00_potted_meat_can')
        except Exception as e:
            print(e)
            return False
        print(rospy.Time.now().to_sec(), t.to_sec(),
              rospy.Time.now().to_sec() - t.to_sec())
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # Time check went alright
        return True

    def test(self, goal, plan_args, task=""):
        """
        Simple loop to try reaching the goal. It uses the execution policy.
        """
        res = None
        counter = 0

        # Get the first world state and update our current plan
        # self.domain.reset()
        world_state = self.observer.observe()
        robot = world_state[self.domain.robot]
        robot.reset()

        # Find a new plan
        plan = self.get_plan(goal, plan_args)
        if plan is None:
            return status.FAILED, 1
            
        # If we still have issues do this
        return self.execute_open_loop(plan, goal)
    
    def execute_open_loop(self, plan, goal):
        """ Execute the plan, as planned. See what happens. """
        actions, subgoals, connections = plan
        for action, subgoal, connection in zip(actions, subgoals, connections):
            # Execute plan parameterized by this particular action. Shouldn't be too hard.
            print("Executing:", action.opname)
            print(subgoal)
            print(connection)

            raw_input()

        # At the end, check to see if we were successful
        return True

    def set_joints_for_indigo(self, qs):
        self.sim.set_joints(
            ["indigo_drawer_top_joint", "indigo_drawer_bottom_joint"],
            [qs[0], qs[1]])

