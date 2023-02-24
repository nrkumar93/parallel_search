# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import print_function

import numpy as np
import rospy

from brain2.task.world_state import WorldStateObserver

# Bullet-specific stuff
import brain2.bullet.problems as problems
from brain2.bullet.interface import BulletInterface
from brain2.bullet.ik import BulletIKSolver
import brain2.utils.axis as axis
from brain2.utils.info import logwarn, logerr, say
import brain2.utils.status as status

# Motion tools; policies and stuff
from brain2.motion_planners.rrt import rrt
from brain2.motion_planners.rrt_connect import rrt_connect

# General domain tools
from brain2.utils.pose import make_pose
from brain2.robot.cube import CubeGraspLookupTable
from brain2.robot.domain import CartObjectsDomainDefinition
from brain2.policies.grasping import get_relative_goal_discrete_sampler

# Tools and inverse kinematics
from brain2.robot.ik import ParallelIKSolver
from brain2.bullet.ik import BulletIKSolver

# ROS tools and stuff
from brain2_ros.trac_ik_solver import TracIKSolver

# --------------
# ROS imports
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
import tf
import tf2_ros

import household_poses as household

from dope_sensor import DopeSensor
from franka_gripper_control import FrankaGripperControl
from franka_gripper_control import SimFrankaGripperControl
from franka_gripper_control import LulaFrankaGripperControl
from lula_franka_control import LulaFrankaControl
from simple_franka_control import SimpleFrankaControl
from brain2.robot.control import RobotControl



# -------------------
from ros_trajectory import spline_parameterization

home_q = np.array([0.041381039132674526, -0.9077992521863245,
    -0.1437450179690571, -2.4953988227174992, -0.04077561163042027,
    1.9712251559377367, 0.7069003548925157, 0.03954381123185158,
    0.03954381123185158])

obs_right_q_01 = np.array([-0.08372005704515857, -1.3871465824194122,
    -0.7403369805891125, -2.714278107354826,
    -0.3100849666529232, 2.0643488490051687,
    0.601301422494936, 0.039627864956855774,
    0.039627864956855774])
obs_right_q_02 = np.array([-0.5457770673878247, -1.1790879560139675,
                            -0.6771668123622974, -2.8217309585016483,
                            -0.028044707424291996, 2.0337970816559263,
                            0.35724591793968896, 0.039578285068273544,
                            0.039578285068273544])
obs_right_q_03 = np.array([0.6702612993931504, -1.5109475905267815,
                           -1.4701904053437065, -2.2544051126401863,
                           -0.9428737315991393, 1.9370799481338923,
                           -0.9984565831516905, 0.03953789919614792,
                           0.03953789919614792])
obs_right_q = obs_right_q_02

class RosWorldStateObserver(WorldStateObserver):
    """reads in the ROS version of the world state and shows it."""

    def __init__(self, domain, root, base_link, ee_link, camera_link, arm=None, gripper=None):
        super(RosWorldStateObserver, self).__init__(domain)

        # ROS objects created here
        # look up base link or whaterver
        self.tf_listener = tf.TransformListener()
        self.tf_broadcaster = tf.TransformBroadcaster()

        # Basic variables for holding info about the world
        self.sensors = []
        self.root = root
        self.ee_link = ee_link
        self.base_link = base_link
        self.camera_link = camera_link
        self.verbose = False

        # Set gripper and arm interfaces
        # These are jointly managing control + data I/O from the arms
        self.arm = arm
        self.gripper = gripper

    def validate(self, env):
        """ make sure the gripper is working """
        logwarn("Ensure that the gripper open + close works.")
        self.gripper.open()
        actor = self.domain.robot
        self.gripper.close()
        ws = self.observe(blocking=False)
        c1 = ws[actor].gripper_state
        print("GRIPPER TEST: c1 =", c1)
        self.gripper.open()
        rospy.sleep(0.1)
        ws = self.observe(blocking=False)
        c2 = ws[actor].gripper_state
        print("GRIPPER TEST: c2 =", c2)
        if abs(c1 - c2) < 0.01:
            print("config when closed:", c1)
            print("config when opened:", c2)
            raise RuntimeError('gripper control failed: ' + str(c1) + ', '
                               + str(c2))
        say("Gripper test finished. Updating camera position...")
        env.update(ws) # Update the scene, just in case.
        self.update_camera_from_tf(env)

    def get_relative_tf(self, frame1, frame2):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            try:
                trans, rot = self.tf_listener.lookupTransform(self.base_link,
                        self.camera_link, rospy.Time(0))
                break
            except (tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                logwarn(str(e))
            rate.sleep()
        return make_pose(trans, rot)

    def lookup_base_relative_to_frame(self, frame):
        return self.get_relative_tf(frame, self.base_link)

    def update_camera_from_tf(self, env):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            try:
                trans, rot = self.tf_listener.lookupTransform(self.root,
                        self.camera_link, rospy.Time(0))
                env.set_camera((trans, rot), matrix=False)
                break
            except (tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                logwarn(str(e))
            rate.sleep()

    def add_sensor(self, SensorType, *args, **kwargs):
        # Sensors get hooks into the observer
        # And we do some registration work here
        self.sensors.append(SensorType(self, *args, **kwargs))
        # TODO: register hooks here instead
        #for name, entity in self.current_state.entities():
        # ...
        sensor = self.sensors[-1]
        for name, entity in self.current_state.entities.items():
            if sensor.has(name):
                # Run update at time t
                entity.update = lambda t, _e=entity: sensor.update(_e, t)

    def update(self, entities=None, verbose=False):
        """ Update world state of the objects we need updated """
        success = True
        t = rospy.Time.now().to_sec()
        self.current_state.time = t 
        if entities is None:
            entities = self.current_state.entities.keys()
        for name in entities:
            if verbose:
                print(t, "UPDATING", name)
            entity = self.current_state.entities[name]
            if name == "robot":

                if self.arm.q_latest is None:
                    success = False
                    continue

                # TF updates it
                try:
                    (trans, rot) = self.tf_listener.lookupTransform(
                            self.root,
                            self.base_link,
                            rospy.Time(0))

                    pose = make_pose(trans, rot)
                    entity.set_base_pose(pose)
                    entity.observed = entity.stable = True

                    # -----------
                    # Get end effector
                    (trans, rot) = self.tf_listener.lookupTransform(
                            self.root,
                            self.ee_link,
                            rospy.Time(0))

                    pose = make_pose(trans, rot)
                    entity.set_ee_pose(pose)
                    entity.set_config(self.arm.q_latest, self.arm.dq_latest)
                    entity.gripper_state = self.arm.g_latest
                    
                    if self.gripper is not None:
                        # Use the gripper object to set gripper-specific stuff
                        # These use hard coded values for now but could be read in
                        # from configuration files
                        self.gripper.update_gripper_state(entity)
                    if entity.gripper_fully_open or entity.gripper_fully_closed:
                        # Grasping clearly failed
                        # NOTE: detach should NOT clear goals, we need something
                        # else for that. Probably an action. But really goal
                        # state needs to be tracked elsewhere
                        entity.detach()
                
                except (tf2_ros.LookupException,
                        tf2_ros.ConnectivityException,
                        tf2_ros.ExtrapolationException) as e:
                    logwarn(str(e))
                    success = False
                    continue
            elif name == "table":
                # For now, there is nothing really to handle with the table or the kitchen.
                # ... or is there?
                try:
                    (trans, rot) = self.tf_listener.lookupTransform(
                            self.root,
                            "00_table",
                            rospy.Time(0))

                    pose = make_pose(trans, rot)
                    entity.pose = np.copy(pose)
                    entity.observed = True
                except (tf2_ros.LookupException,
                        tf2_ros.ConnectivityException,
                        tf2_ros.ExtrapolationException) as e:
                    logwarn(str(e))
                    success = False
                    continue
            elif name == "kitchen": # or name == "kitchen":
                # For now, there is nothing really to handle with the table or the kitchen.
                # ... or is there?
                try:
                    (trans, rot) = self.tf_listener.lookupTransform(
                            self.root,
                            "sektion",
                            rospy.Time(0))

                    pose = make_pose(trans, rot)
                    entity.pose = np.copy(pose)
                    entity.observed = True
                except (tf2_ros.LookupException,
                        tf2_ros.ConnectivityException,
                        tf2_ros.ExtrapolationException) as e:
                    logwarn(str(e))
                    success = False
                    continue
            else:
                try:
                    res = entity.update(t)
                except AttributeError as e:
                    logerr("No state update function registered for " +
                            str(name) + " --- "+ str(entity))
                    raise e

                if entity.pose is None:
                    continue
                elif res < 0 and self.verbose:
                    logerr("Updating " + str(entity.name)
                           + " failed with code " + str(res))
        if verbose:
            print("==== DONE UPDATING ====")
        return success


def get_ik_solver(args):
    """
    Get inverse kinematics solver - this should just be current best
    """
    ik = ParallelIKSolver()
    for i in range(args.num_ik_threads):
        _ik = TracIKSolver(base_link="base_link",
                      ee_link="right_gripper",
                      dof=7)
        ik.add_ik_solver(_ik)
    if args.parallel_ik:
        ik.start()
    return ik


def CreateLulaControlInterface(ik_solver, robot_ref=None, sim=False, lula=True):

    # Create interfaces with real robot
    if not lula:
        # Create robot first and pull out lula control interface
        franka = SimpleFrankaControl(
                                   base_link="base_link",
                                   ee_link="right_gripper",
                                   dof=7,
                                   sim=True,
                                   ik_solver=ik_solver,
                                   robot_ref=robot_ref)
        franka_gripper = SimFrankaGripperControl()
    elif sim:
        # Create robot first and pull out lula control interface
        franka = LulaFrankaControl(base_link="base_link",
                                   ee_link="right_gripper",
                                   dof=7,
                                   sim=True)
        franka_gripper = LulaFrankaGripperControl(franka.franka)
    else:
        # Create the arm controller
        franka = LulaFrankaControl(
                                   base_link="base_link",
                                   ee_link="right_gripper",
                                   dof=7,
                                   sim=False)
        # Create gripper interface
        franka_gripper = FrankaGripperControl()

    return RobotControl(gripper=franka_gripper, arm=franka)


def setup_robot_control(domain, ik):
    """ Create lula interfaces and make sure we can talk to the robot """

    # Create robot control interface and add to robot
    ctrl = CreateLulaControlInterface(ik)
    domain.set_robot_control(ctrl)
    # Create observer to get scene information
    observer = RosWorldStateObserver(domain,
                                     root="base_link",
                                     base_link="measured/base_link",
                                     ee_link="measured/right_gripper",
                                     camera_link="depth_camera_link",
                                     arm=ctrl.arm,
                                     gripper=ctrl.gripper)
    return ctrl, observer
    

def wait_for_q(observer, robot, q, tol=1e-3, max_t=5.):
    """ Wait until ROS has the configuration of the robot """
    rate = rospy.Rate(30)
    t0 = rospy.Time.now()
    while not rospy.is_shutdown():
        ws = observer.observe(blocking=False)
        if np.linalg.norm(ws[robot].q - q) < tol:
            break
        elif (rospy.Time.now() - t0).to_sec() > max_t:
            break

