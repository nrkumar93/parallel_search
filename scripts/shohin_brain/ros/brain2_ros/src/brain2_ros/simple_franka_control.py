# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import print_function

import numpy as np
import pickle
import rospy
import time

import brain2.utils.axis as axis

try:
    import PyKDL as kdl
    from kdl_parser_py import urdf
except ImportError as e:
    rospy.logwarn("Could not load kdl parser. Try: `sudo apt install "
                  "ros-kinetic-kdl-*`")
    raise e

# --- Brain
from brain2.utils.info import logerr, logwarn

# --- Trajectory tools
from ros_trajectory import spline_parameterization
from ros_trajectory import spline_trajectory
from control_visualizer import ControlVisualizer

# Display trajectories
from moveit_msgs.msg import DisplayTrajectory
from moveit_msgs.msg import DisplayRobotState
from moveit_msgs.msg import RobotTrajectory
import msg_tools # our message authoring tools

# Conditions
from brain2.conditions.position import PoseCondition

# Set up goal
from brain2_msgs.msg import Goal
from sensor_msgs.msg import JointState


class SimpleFrankaControl(object):

    joint_names = ["panda_joint%d" % i for i in range(1, 8)]

    def _js_cb(self, msg):
        self.q_latest = np.array(msg.position[:7])
        self.dq_latest = np.array(msg.velocity[:7])
        self.g_latest = np.array(msg.position[8])

    def __init__(self, base_link, ee_link, dof, ik_solver, robot_ref, sim=False):
        """Set this up. dof is the number of dof that THIS INTERFACE IS
        RESPONSIBLE FOR."""
        self.dof = dof
        self.base_link = base_link
        self.ee_link = ee_link
        self.sim = sim
        self.viz = ControlVisualizer()
        self.ik_solver = ik_solver
        self.robot_ref = robot_ref  # Contains information about the robot and a reference 

        # Joint states
        self._js_sub = rospy.Subscriber("joint_states", JointState,
                                       self._js_cb, 
                                       queue_size=100)

        success, kdl_tree = urdf.treeFromParam('/robot_description')
        if not success:
            raise RuntimeError(
                "Could not create kinematic tree from /robot_description.")

        self.kdl_chain = kdl_tree.getChain(self.base_link, ee_link)
        print("[FRANKA] Number of joints in KDL chain:",
              self.kdl_chain.getNrOfJoints())
        assert self.dof == self.kdl_chain.getNrOfJoints()
        self.kdl_fk = kdl.ChainFkSolverPos_recursive(self.kdl_chain)

        # Messages for showing trajectories
        self.robot_state_state_pub = rospy.Publisher('/display_robot_state_state', DisplayRobotState, queue_size=1)
        self.display_trajectory_pub = rospy.Publisher('/display_planned_path', DisplayTrajectory, queue_size=1)
        self.goal_joint_cmd_pub = rospy.Publisher('/interp/joint_states',
                                                  Goal, queue_size=1)

        # Robot things
        self.q_latest = None
        self.dq_latest = None
        self.g_latest = None


    def joint_list_to_kdl(self, q):
        """Get in KDL format"""
        if q is None:
            return None
        if isinstance(q, np.matrix) and q.shape[1] == 0:
            q = q.T.tolist()[0]
        q_kdl = kdl.JntArray(len(q))
        for i, q_i in enumerate(q):
            q_kdl[i] = q_i
        return q_kdl

    def forward_kinematics(self, q):
        ee_frame = kdl.Frame()
        kinematics_status = self.kdl_fk.JntToCart(self.joint_list_to_kdl(q[:self.dof]),
                                                  ee_frame)
        if kinematics_status >= 0:
            p = ee_frame.p
            M = ee_frame.M
            return np.array([[M[0, 0], M[0, 1], M[0, 2], p.x()],
                             [M[1, 0], M[1, 1], M[1, 2], p.y()],
                             [M[2, 0], M[2, 1], M[2, 2], p.z()],
                             [0, 0, 0, 1]])
        else:
            return None

    def visualize_trajectory(self, robot_state, trajectory, timings):
        """ Send trajectory to ROS to visualize """
        msg = spline_trajectory(robot_state, trajectory, timings)
        self.publish_display_trajectory(robot_state, msg)

    def execute_joint_trajectory(self, trajectory, timings=None, entity_state=None, timestep=0.2,
                                 wait_at_end=True):
        """ Slowly execute a ROS joint state trajectory """

        if entity_state is not None:
            msg = spline_parameterization(entity_state, trajectory)
            self.publish_display_trajectory(entity_state, msg)

        if timings is None:
            timings = np.arange(len(trajectory)) * timestep

        final_q = trajectory[-1] # bias to final point, don't move in config space
        self.franka.set_speed('med_slow') # For safety!
        self.config_modulator.send_config(final_q[:self.dof])
        prev_t = timings[0]
        for i, (q, t) in enumerate(zip(trajectory, timings)):
            T = self.forward_kinematics(q[:self.dof])
            if T is None:
                logerr("forward kinematics failed for franka! q = " + str(q))
                continue
            if wait_at_end and i >= len(trajectory) - 1:
                wait = True
            else:
                wait = False

            # Send command and visualization
            self.viz.send(T)
            lula_go_local(self.franka.end_effector, T, wait_for_target=wait)
            
            # only sleep if not sending waypoint
            if not wait:
                rospy.sleep(t - prev_t)
                prev_t = t
        if not wait:
            rospy.sleep(0.5)

        # Finish
        # self.viz.stop()

    def go_local(self, T=None, q=None, wait=False, time=100., err_threshold=0.003):
        """ Control end effector position and orientation to target """
        if q is not None and T is None:
            T = self.forward_kinematics(q[:self.dof])
        elif q is None and T is not None:
            q = self.ik_solver(self.robot_ref, T, q)
        elif q is None and T is None:
            raise RuntimeError('must provide q or T')

        # Send visualization ASAP
        self.viz.send(T)

        # Publish
        self.goal_joint_cmd_pub.publish(
            joint_state=JointState(
                name=self.joint_names,
                position=list(q),
            ),
            time=time,
        )
        if T is None:
            raise RuntimeError('go_local: T or q required')

    def retract(self, speed=None, lula=True, wait=True):
        """
        Take the arm back to its home position.
        """
        if self.use_lula:
            if speed is not None:
                self.franka.set_speed(speed)
            self.franka.end_effector.go_local(orig=[], axis_x=[], axis_y=[], axis_z=[])
            self.go_local(q=self.home_q, wait=wait)
        else:
            # Send an ordinary move to this q
            self.go_local(q=self.home_q, wait=wait)

    def lula_go_local_no_orientation(self, T, q=None):
        """ Control end effector position but not orientation """
        if q is not None:
            self.config_modulator.send_config(q[:self.dof])
        lula_go_local_no_orientation(self.franka.end_effector, T, wait_for_target=False)
        self.viz.send(T)

    def publish_display_trajectory(self, robot_state, joint_trajectory):

        # create messages
        display_trajectory = DisplayTrajectory()
        display_trajectory.trajectory_start = msg_tools.makeRobotStateMsg(robot_state)
        robot_state_trajectory = RobotTrajectory(joint_trajectory=joint_trajectory)

        robot_state_trajectory.joint_trajectory.header.frame_id = self.base_link
        display_trajectory.trajectory.append(robot_state_trajectory)
        self.display_trajectory_pub.publish(display_trajectory)

        display_state = DisplayRobotState()
        display_state.state = display_trajectory.trajectory_start
        last_conf = joint_trajectory.points[-1].positions
        joint_state = display_state.state.joint_state
        joint_state.position = list(joint_state.position)
        for joint_name, position in zip(joint_trajectory.joint_names, last_conf):
            joint_index = joint_state.name.index(joint_name)
            joint_state.position[joint_index] = position

        self.robot_state_state_pub.publish(display_state)
        self.viz.send(self.forward_kinematics(last_conf[:self.dof]))

