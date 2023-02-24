#!/usr/bin/env python
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from __future__ import print_function

from brain2_msgs.msg import Attach
from lula_franka.franka import Franka
from lula_control.frame_commander import RobotConfigModulator
from lula_tools import lula_go_local
from lula_tools import lula_go_local_y_axis
from lula_tools import lula_go_local_no_orientation
from moveit_msgs.msg import CollisionObject
from visualization_msgs.msg import Marker, MarkerArray
from shape_msgs.msg import SolidPrimitive
from sensor_msgs.msg import JointState
from lula_control.frame_commander import FrameConvergenceCriteria, TimeoutMonitor
from lula_control.message_listener import MessageListener
from lula_control.trajectory_client import TrajectoryClient
from lula_controller_msgs.msg import ApproachableTarget, AffineRmp
from lula_dartpy.fixed_base_suppressor import FixedBaseSuppressor
from threading import Lock

import brain2.utils.axis as axis
import copy
import os
import signal
import rospy
import sys
import numpy as np
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from brain2_msgs.msg import Grasp
from brain2_msgs.msg import Goal
from brain2_ros.control_visualizer import ControlVisualizer
import tf.transformations as tra
import parse

try:
    import PyKDL as kdl
    from kdl_parser_py import urdf
except ImportError as e:
    rospy.logwarn("Could not load kdl parser. Try: `sudo apt install "
                  "ros-kinetic-kdl-*`")
    raise e

from trac_ik_python.trac_ik import IK

from collections import namedtuple
GripperInfo = namedtuple("GripperInfo", ["joints", "open_positions",
                                         "closed_positions"], verbose=False)

USE_MOVEIT = True


class MoveitBridge(object):

    # TODO make this more portable
    joint_names = ["panda_joint%d" % i for i in range(1, 8)]

    default_home_q = [0.01200158428400755, -0.5697816014289856,
                      5.6801487517077476e-05,
                      -2.8105969429016113, -0.00025768374325707555, 3.0363450050354004,
                      0.7410701513290405]
    default_gripper = GripperInfo(
        joints=["panda_finger_joint1", "panda_finger_joint2"],
        open_positions=[0.04, 0.04],
        closed_positions=[0., 0.],
    )

    def print_moveit_info(self):
        """
        Just prints out some debug information about the MoveIt setup.
        """

        print("============ Reference frame: %s" % self.planning_frame)
        group_names = self.robot.get_group_names()
        print("============ Robot Groups:", group_names)
        print("============ Printing robot state")
        print(self.robot.get_current_state())
        print("")

    def reset_attachments(self):
        """
        Clear out attachments.
        """
        self.attach_cmd_pub.publish(actor=self.actor, goal="")

    def attach(self, goal_obj):
        """
        Send out an attachment message.
        """
        self.attach_cmd_pub.publish(actor=self.actor, goal=goal_obj)

    def __init__(self, group_name, robot_interface, dilation=1.,
                 lula_world_objects="/world/objects/rviz", verbose=0,
                 franka=None, ee_link='right_gripper', home_q=None,
                 gripper=None,
                 use_lula=True,
                 view_tags={},
                 use_gripper_server=False,):
        """
        Create the MoveIt bridge, with reasonable default values.
        """
        self._update_world = False
        self.verbose = verbose
        self.use_gripper_server = use_gripper_server
        self.tracked_objs = set()
        self.ignored_objs = set()
        moveit_commander.roscpp_initialize(sys.argv)
        self.visualizer = ControlVisualizer()
        self.obj_lock = Lock()
        self.actor = "arm"
        self.attach_cmd_pub = rospy.Publisher(
            '/brain/attach_state', Attach, queue_size=1)

        self.view_tags = view_tags

        if USE_MOVEIT:
            self.robot = moveit_commander.RobotCommander()
            self.scene = moveit_commander.PlanningSceneInterface()
            self.group = moveit_commander.MoveGroupCommander(group_name)
            self.planning_frame = self.group.get_planning_frame()
            # Get collision object publisher
            self.collision_object_pub = self.scene._pub_co

        self.home_q = home_q
        if self.home_q is None:
            self.home_q = self.default_home_q

        self.gripper = gripper
        if self.gripper is None:
            self.gripper = self.default_gripper

        if robot_interface is None:
            self.use_lula = False
        else:
            self.use_lula = use_lula

        if lula_world_objects is not None:
            self.lula_world_sub = rospy.Subscriber(
                lula_world_objects, MarkerArray, self._lula_world_cb, queue_size=1000)

        self.fixed_base_suppressors = {}
        if self.use_lula:
            self.franka = robot_interface
            self.config_modulator = RobotConfigModulator()
            for s, s_tag in self.view_tags.items():
                self.fixed_base_suppressors[s] = FixedBaseSuppressor(
                    s_tag,
                    wait_for_connection=False)

            if self.verbose:
                print('Setting up trajectory client...')
            self.trajectory_client = TrajectoryClient(
                trajectory_msg_topic='/robot/right/cspace_trajectory/msg',
                trajectory_commander_topic='/robot/right/cspace_trajectory/command')

            if self.verbose:
                print("Setting up approachable target publisher...")
            self.approachable_target_publisher = rospy.Publisher(
                '/robot/right/approachable_target',
                ApproachableTarget, queue_size=10)

            if self.verbose:
                print('Setting up affine rmp message listener...')
            self.affine_rmp_listener = MessageListener(
                topic='/robot/right/affine_rmp',
                msg_type=AffineRmp,
                clear_on_timeout=True)

        else:
            self.franka = None
            self.goal_joint_cmd_pub = rospy.Publisher('/interp/joint_states',
                                                      Goal, queue_size=1)
            self.gripper_cmd_pub = rospy.Publisher("interp/gripper", JointState,
                                                   queue_size=1)

        self.display_trajectory_publisher = rospy.Publisher(
            '/move_group/display_planned_path',
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20)

        self.grasp_pub = rospy.Publisher('robot/gripper', Grasp, queue_size=10)
        self.dilation = dilation

        # initialize forward kinematics
        self.base_link = 'base_link'
        self.ee_link = ee_link

        success, kdl_tree = urdf.treeFromParam('/robot_description')
        if not success:
            raise RuntimeError(
                "Could not create kinematic tree from /robot_description.")

        self.kdl_chain = kdl_tree.getChain(self.base_link, ee_link)
        if self.verbose:
            print("Number of joints in KDL chain:",
                  self.kdl_chain.getNrOfJoints())
        self.kdl_fk = kdl.ChainFkSolverPos_recursive(self.kdl_chain)

        # initialize inverse kinematics
        # uses robot_description from rosparam server by default
        #
        # There's apparently also a moveit kinematics plugin for trac_ik
        # that could be used through the moveit_commander interface
        # but it doesn't expose all arguments:
        # http://docs.ros.org/kinetic/api/moveit_tutorials/html/doc/trac_ik/trac_ik_tutorial.html
        self.ik_solver = IK(self.base_link, ee_link,
                            timeout=0.05,
                            solve_type="Distance")
        self.last_ik = None
        self.affine_rmp_id = 0

    def get_fixed_base_suppressor(self, view):
        """
        This is designed to communicate with dart -- unfix base when we want
        to lock on.
        """
        return self.fixed_base_suppressors[view]

    def get_current_fixed_base_suppressor(self):
        return self.fixed_base_suppressors[self.current_view]

    def unfix_bases(self):
        for suppressor in self.fixed_base_suppressors.values():
            suppressor.deactivate()

    def fix_bases(self):
        for suppressor in self.fixed_base_suppressors.values():
            suppressor.activate()

    def go_approachable_target(self, pose, q=None, wait=False):
        """
        Use approachable_target and motion optimization to grab something
        from a particular approach direction. "pose" is a 4x4 matrix in SE(3),
        "q" is estimated final configuration at pose (or None), and direction
        is the approach direction. Set wait to true for blocking motions.
        """

        if not self.use_lula:
            raise RuntimeError(
                'go_approachable_target not supported without lula')

        print("TARGET =", pose[:3, 3])
        target = ApproachableTarget()
        self.affine_rmp_id += 1
        target.id = self.affine_rmp_id
        target.frame = self.ee_link
        target.orig = pose[:3, axis.POS]
        target.axis_x = pose[:3, axis.X]
        target.axis_y = pose[:3, axis.Y]
        target.axis_z = pose[:3, axis.Z]
        target.approach_direction = pose[:3, axis.Z]
        self.approachable_target_publisher.publish(target)

        # This is all for the blocking version.
        if wait:
            init_id = self.affine_rmp_id
            rate = rospy.Rate(50)
            final_sent = False
            prev_prog = -1
            while not rospy.is_shutdown():
                progress = self.get_affine_rmp_progress()
                if self.verbose or True:
                    print("Affine RMP:", self.affine_rmp_id,
                          str(progress * 100) + "%")
                if progress > 0.1 and self.affine_rmp_id != init_id:
                    # Preempted -- only checking progress to make sure we do
                    # not hand over control too early.
                    return False
                elif not final_sent and progress > 0.5:
                    # Send final position
                    self.go_local(pose, q=q)
                    final_sent = True
                elif progress >= 1. - 1e-2:
                    # Done
                    return True
                if progress < prev_prog:
                    raise RuntimeError('progress went backwards')
                prev_prog = progress
                rate.sleep()
            return False

    def get_affine_rmp_progress(self):
        """
        Return progress through currently tracked affine RMP.
        """

        frac = 0
        if self.affine_rmp_listener.has_msg():
            affine_rmp = self.affine_rmp_listener.latest_msg
            if affine_rmp.id == self.affine_rmp_id:
                t = affine_rmp.time_from_start.to_sec()
                T = affine_rmp.total_trajectory_duration.to_sec()
                frac = t / T

        return frac

    def open_gripper(self, force=40., sleep=0.,
                     speed=0.1, wait=True):
        if self.use_gripper_server:
            grasp = Grasp(gripper=self.ee_link,
                          obj_name="", obj_base_frame="")
            self.grasp_pub.publish(grasp)
        elif self.use_lula:
            self.franka.end_effector.gripper.open(speed=speed, wait=wait)
        else:
            self.gripper_cmd_pub.publish(JointState(name=self.gripper.joints,
                                                    position=self.gripper.open_positions))

        self.reset_attachments()
        if sleep > 0.:
            rospy.sleep(sleep)

    def interpolate_go_local(self, start, goal, duration, timeout, err_threshold=0.003, q=None):
        if self.use_lula:
            ee = self.franka.end_effector
            q = self.ik(goal, q)
            self.config_modulator.send_config(q)
        else:
            ee = None
        steps = 10
        dt = duration / steps
        start_quat = tra.quaternion_from_matrix(start)
        goal_quat = tra.quaternion_from_matrix(goal)
        for t in np.linspace(0, 1, steps):
            quat = tra.quaternion_slerp(start_quat, goal_quat, t)
            tf = tra.quaternion_matrix(quat)
            tf[:3, 3] = (1.0 - t) * start[:3, 3] + t * goal[:3, 3]

            high_precision = True
            self.visualizer.send(tf)
            if ee is not None:
                ee.go_local(
                    orig=tf[:3, 3],
                    axis_x=tf[:3, 0],
                    axis_z=tf[:3, 2],
                    use_target_weight_override=high_precision,
                    use_default_config=False,
                    wait_for_target=False
                )
            else:
                self.go_local(tf)
            rospy.sleep(dt)

        pose = goal
        conv = FrameConvergenceCriteria(
            target_orig=pose[:3, 3],
            target_axis_x=pose[:3, 0],
            target_axis_z=pose[:3, 2],
            required_orig_err=err_threshold,
            timeout_monitor=TimeoutMonitor(timeout)
        )

        if self.use_lula:
            rate = rospy.Rate(30)
            while not rospy.is_shutdown() and not self.franka.end_effector.is_preempted:
                if conv.update(
                    self.franka.end_effector.frame_status.orig,
                    self.franka.end_effector.frame_status.axis_x,
                    self.franka.end_effector.frame_status.axis_y,
                    self.franka.end_effector.frame_status.axis_z,
                    verbose=True
                ):
                    break

                rate.sleep()
        self.visualizer.stop()

    def close_gripper(self, goal_obj="", goal_frame="", controllable_object=None, force=40.,
                      sleep=0., speed=0.1, wait=True):
        """
        Properly close the gripper.
        """
        print("--------")
        print(self.use_gripper_server, self.use_lula)
        if self.use_gripper_server:
            grasp = Grasp(gripper="right_gripper",
                          obj_name=goal_obj, obj_base_frame=goal_frame)
            self.grasp_pub.publish(grasp)
        elif self.use_lula:
            self.franka.end_effector.gripper.close(
                controllable_object,
                wait=wait,
                speed=speed,
                actuate_gripper=True,
                force=force,)
        else:
            self.gripper_cmd_pub.publish(JointState(name=self.gripper.joints,
                                                    position=self.gripper.closed_positions))
        if sleep > 0.:
            rospy.sleep(sleep)

    def _lula_world_cb(self, msg):
        """
        Create MoveIt collision objects corresponding to the ones in lula.
        """
        if self._update_world:
            if self.verbose > 1:
                print("===========================")
            for marker in msg.markers:
                if self.verbose > 1:
                    print(marker.ns, marker.header.frame_id)
                    print(marker.type == Marker.CUBE)
                if (marker.ns not in self.tracked_objs and marker.ns not in
                        self.ignored_objs):
                    # Add it
                    if marker.type == Marker.CUBE:
                        dims = marker.scale.x, marker.scale.y, marker.scale.z
                        self.add_box(marker.ns, dims, frame=marker.header.frame_id,
                                     pose_msg=marker.pose)
                        if self.verbose > 0:
                            print("... adding box", marker.ns,
                                  "with dims", dims)
                    elif marker.type == Marker.CYLINDER:
                        dims = marker.scale.z, 0.5 * marker.scale.x, 0
                        self.add_cylinder(marker.ns, dims,
                                          frame=marker.header.frame_id,
                                          pose_msg=marker.pose)
                        print("... adding cylinder",
                              marker.ns, "with dims", dims)
                elif marker.ns in self.tracked_objs:
                    self.update(marker.ns, marker.header.frame_id, marker.pose)
            self._update_world = False

    def update(self, name, frame, pose_msg):
        """
        Send updates to planning scene for all tracked objects
        """
        if not USE_MOVEIT:
            return
        co = CollisionObject()
        co.id = name
        co.operation = CollisionObject.MOVE
        co.header.frame_id = frame
        co.primitive_poses = [pose_msg]
        self.collision_object_pub.publish(co)

    def __del__(self):
        moveit_commander.roscpp_shutdown()

    def retract(self, speed=None, lula=True, wait=True):
        """
        Take the arm back to its home position.
        """
        if self.use_lula and lula:
            if speed is not None:
                self.franka.set_speed(speed)
            self.franka.end_effector.go_local(orig=[], axis_x=[], axis_y=[], axis_z=[])
            self.go_local(q=self.home_q, wait=wait)
        else:
            # Send an ordinary move to this q
            self.go_local(q=self.home_q, wait=wait)

    def go_local_y_axis(self, T, q=None, controlled_frame=None):
        if q is not None:
            self.config_modulator.send_config(q)
        if controlled_frame is None:
            controlled_frame = self.franka.end_effector
        lula_go_local_y_axis(controlled_frame, T)

    def go_local_no_orientation(self, T, q=None, controlled_frame=None):
        if not self.use_lula:
            self.go_local(T, controlled_frame=controlled_frame)
        else:
            if q is not None:
                self.config_modulator.send_config(q)
            if controlled_frame is None:
                controlled_frame = self.franka.end_effector
            print('I SURVIVED THE HARDSHIPS OF MAKING HERE IN THE FUNCTION')
            lula_go_local_no_orientation(controlled_frame, T)
            print('IF THIS WORKS, SOMETHING IS PROBABLY WRONG ELSEWHERE')

    def go_local(self, T=None, q=None, wait=False, controlled_frame=None,
                 time=100., err_threshold=0.003):
        """
        Move locally to transform or joint configuration q.
        """
        if T is None and q is not None:
            T = self.forward_kinematics(q)
        self.visualizer.send(T)
        if self.use_lula:
            if controlled_frame is None:
                controlled_frame = self.franka.end_effector
            if q is not None:
                self.config_modulator.send_config(q)
            lula_go_local(self.franka.end_effector, T, wait_for_target=False)

            if wait:
                conv = FrameConvergenceCriteria(
                    target_orig=T[:3, 3],
                    target_axis_x=T[:3, 0],
                    target_axis_z=T[:3, 2],
                    required_orig_err=err_threshold,
                    timeout_monitor=TimeoutMonitor(time)
                )
                rate = rospy.Rate(30)
                while not rospy.is_shutdown() and not self.franka.end_effector.is_preempted:
                    if conv.update(
                            self.franka.end_effector.frame_status.orig,
                            self.franka.end_effector.frame_status.axis_x,
                            self.franka.end_effector.frame_status.axis_y,
                            self.franka.end_effector.frame_status.axis_z,
                            verbose=True):
                        break
                    rate.sleep()

        else:
            if q is None and T is not None:
                q = self.ik(T, q)
            self.goal_joint_cmd_pub.publish(
                joint_state=JointState(
                    name=self.joint_names,
                    position=q,
                ),
                time=time,
            )

    def joint_list_to_kdl(self, q):
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
        kinematics_status = self.kdl_fk.JntToCart(self.joint_list_to_kdl(q),
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

    def get_pose(self, pos, rot):
        """
        Helper function to create a pose.

        pos: position (x, y, z)
        rot: orientation as a quaternion (x, y, z, w)
        """
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation.x = rot[0]
        pose_goal.orientation.y = rot[1]
        pose_goal.orientation.z = rot[2]
        pose_goal.orientation.w = rot[3]
        pose_goal.position.x = pos[0]
        pose_goal.position.y = pos[1]
        pose_goal.position.z = pos[2]
        return pose_goal

    def publish_display_trajectory(
        self,
        plan,
        trajectory_start,
        base_link=None,
    ):
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = trajectory_start
        display_trajectory.trajectory.append(plan)
        if base_link is None:
            base_link = self.base_link
        display_trajectory.trajectory[0].joint_trajectory.header.frame_id = \
            base_link
        rospy.sleep(0.1)
        self.display_trajectory_publisher.publish(display_trajectory)

    def plan_q(self, q):
        self.group.set_joint_value_target(q)
        plan = self.group.plan()
        if self.verbose > 1:
            print("goal =", q)
            print(plan)
            print(type(plan))
        if len(plan.joint_trajectory.points) == 0:
            return None
        return plan

    def ik(self, T, q0=None):
        if q0 is None:
            q0 = self.group.get_current_joint_values()
        rot = tra.quaternion_from_matrix(T)
        pos = T[:3, 3]
        result = self.ik_solver.get_ik(
            qinit=q0,
            x=pos[0],
            y=pos[1],
            z=pos[2],
            rx=rot[0],
            ry=rot[1],
            rz=rot[2],
            rw=rot[3],)
        return result

    def get_plan(self, T, q0=None):
        self.visualizer.send(T)
        rot = tra.quaternion_from_matrix(T)
        pos = T[:3, axis.POS]
        return self.plan(pos, rot, q0)

    def plan(self, pos, rot, q0=None, use_trac_ik=True):
        pose_goal = self.get_pose(pos, rot)

        if q0 is None:
            q0 = self.group.get_current_joint_values()

        self.last_ik = None
        if use_trac_ik:
            result = self.ik_solver.get_ik(
                qinit=self.home_q,
                x=pos[0],
                y=pos[1],
                z=pos[2],
                rx=rot[0],
                ry=rot[1],
                rz=rot[2],
                rw=rot[3],
            )
            if result is not None:
                print("IK solution: ", result)
                self.last_ik = result
                self.group.set_joint_value_target(result)

                ee_frame = self.forward_kinematics(result)
                self.visualizer.send(ee_frame)
            else:
                print("No IK solution found!")
                return None
        else:
            self.last_ik = pose_goal
            self.group.set_pose_target(pose_goal)

        plan = self.group.plan()
        if self.verbose > 1:
            print(pose_goal)
            print(plan)
            print(type(plan))
        if len(plan.joint_trajectory.points) == 0:
            return None
        return plan

    def step_plan(self, goal_pose, plan, t, publish_display_trajectory=True, dilation=1):
        if publish_display_trajectory:
            self.publish_display_trajectory(
                plan, trajectory_start=self.robot.get_current_state())

        points = plan.joint_trajectory.points
        idx = 0
        for i in range(len(points)):
            if i > 0:
                # time_from_start is relative to trajectory.header.stamp; each
                # trajectory point's time_from_start must be greater than the
                # last.
                t_cur = points[i].time_from_start.to_sec()
                time = t_cur * dilation
                t_prev = points[i - 1].time_from_start.to_sec()
                step_time = (t_cur - t_prev) * dilation
            else:
                time = 0

            if self.verbose > 0:
                print('Check current time', t, "< plan time", time)
            if time > t:
                break
            else:
                idx += 1

        # Go locally to the final position at the very end -- make sure we get
        # there eventually.
        if t > points[-1].time_from_start.to_sec():
            # We're at the end, just move there.
            self.go_local(goal_pose, q=points[-1].positions, time=step_time)
            return True
        else:
            if idx >= len(points):
                idx = -1
            # names = plan.joint_trajectory.joint_names
            positions = points[idx].positions
            T0 = self.forward_kinematics(positions)
            self.visualizer.send(T0)
            self.go_local(T0, q=points[-1].positions)
            return False

    def execute(self, plan,
                required_orig_err=0.005,
                timeout=5.0,
                publish_display_trajectory=False,
                override_timing=True,
                override_vel=1.):
        """
        Execute a trajectory.
        """
        if publish_display_trajectory:
            self.publish_display_trajectory(
                plan, trajectory_start=self.robot.get_current_state())

        #names = plan.joint_trajectory.joint_names
        #points = plan.joint_trajectory.points
        names = plan.joint_names
        points = plan.points
        if not self.use_lula or True:
            prev_positions = np.array(points[0].positions)
            t_prev = points[0].time_from_start.to_sec()
            for i in range(len(points)):
                positions = np.array(points[i].positions)
                t_cur = points[i].time_from_start.to_sec()

                if self.verbose > 0:
                    print(zip(names, positions))

                # Command frame
                ee_frame = self.forward_kinematics(positions)
                self.visualizer.send(ee_frame)

                # time_from_start is relative to trajectory.header.stamp; each
                # trajectory point's time_from_start must be greater than the
                # last.
                dq = positions - prev_positions

                if override_timing:
                    max_dq = np.max(np.abs(dq))
                    override_timing_step = max_dq / override_vel
                    print(i, max_dq, override_timing_step)
                    t_cur = t_prev + override_timing_step

                #q = points[-1].positions
                q = positions
                time = (t_cur - t_prev) * self.dilation
                self.go_local(T=ee_frame, q=q, time=time)
                if i > 0:
                    # first time_from_start is usually zero - so we can ignore it
                    print("wait for t =", time, t_cur, t_prev)
                    rospy.sleep(time)

                prev_positions = positions
                t_prev = t_cur

            q = points[-1].positions
            self.go_local(T=ee_frame, q=q, wait=True)
        else:
            self.trajectory_client.send(plan.joint_trajectory)
            self.trajectory_client.wait_for_finish()

        self.visualizer.stop()

    def goto(self, pos=None, rot=None, q=None, use_trac_ik=True, timeout=5.0):
        if pos is not None:
            plan = self.plan(pos, rot, use_trac_ik=use_trac_ik)
        elif q is not None:
            plan = self.plan_q(q)
        else:
            raise RuntimeError('must give either pos, rot or q to goto')

        if plan is None:
            return False

        self.execute(plan, timeout=timeout)
        self.freeze()

        return True

    def goto_last_q_fk(self, timeout=None, required_orig_err=.01):
        if self.last_ik is not None:
            ee_frame = self.forward_kinematics(self.last_ik)
            self.go_local(ee_frame, self.last_ik, wait=(timeout is None))
        else:
            print("last_ik not set")
            return

        if timeout:
            conv = FrameConvergenceCriteria(
                target_orig=ee_frame[:3, 3],
                target_axis_x=ee_frame[:3, 0],
                target_axis_z=ee_frame[:3, 2],
                required_orig_err=required_orig_err,
                timeout_monitor=TimeoutMonitor(timeout))
            rate = rospy.Rate(30)
            while not rospy.is_shutdown() and not self.franka.end_effector.is_preempted:
                if conv.update(
                        self.franka.end_effector.frame_status.orig,
                        self.franka.end_effector.frame_status.axis_x,
                        self.franka.end_effector.frame_status.axis_y,
                        self.franka.end_effector.frame_status.axis_z,
                        verbose=True):
                    break
                rate.sleep()

    def freeze(self, with_lula=True):
        """
        Just completely stop motion.
        """
        if self.use_lula and with_lula:
            self.franka.end_effector.freeze()
        else:
            self.go_local(q=self.group.get_current_joint_values())

    def add_box(
        self, name, dims, pos=(
            0, 0, 0), rot=(
            0, 0, 0, 1), frame="world", pose_msg=None):
        self.tracked_objs.add(name)
        if not USE_MOVEIT:
            return
        self.add_obstacle(name, dims, pos, rot, frame, 'box', pose_msg)

    def add_cylinder(self, name, dims, pos=(0, 0, 0), rot=(0, 0, 0, 1),
                     frame="world", pose_msg=None):
        self.tracked_objs.add(name)
        if not USE_MOVEIT:
            return
        # Create a cylinder CO
        # publish it
        co = CollisionObject()
        co.id = name
        co.operation = CollisionObject.ADD
        can = SolidPrimitive()
        can.type = SolidPrimitive.CYLINDER
        can.dimensions = dims
        co.primitives = [can]
        if pose_msg is None:
            co.primitive_poses = [self.get_pose(pos, rot)]
        else:
            co.primitive_poses = [copy.copy(pose_msg)]
        co.header.frame_id = frame
        self.collision_object_pub.publish(co)

    def ignore(self, name):
        """
        Call this on objects that you have picked up
        TODO: add something to attach the collision object instead of just
        ignoring it.
        """
        self.ignored_objs.add(name)
        if name in self.tracked_objs:
            print("Started ignoring", name)
        self.remove(name)

    def track(self, name):
        """
        Call this on objects that you want to start avoiding again.
        """
        if name in self.ignored_objs:
            print("Stopped ignoring", name)
            self.ignored_objs.remove(name)

    def remove(self, name=None):
        """
        Remove an object -- or remove all. This is based on the similar function
        in MoveIt's Scene object.
        """

        self.obj_lock.acquire()

        # Delete from tracked list
        if name in self.tracked_objs:
            self.tracked_objs.remove(name)
        if name is None:
            # clear the whole set
            self.tracked_objs = set()

        # Send ignore request
        if USE_MOVEIT:
            co = CollisionObject()
            co.operation = CollisionObject.REMOVE
            if name is not None:
                co.id = name
            self.collision_object_pub.publish(co)

        self.obj_lock.release()

    def add_obstacle(
        self, name, dims, pos=(
            0, 0, 0), rot=(
            0, 0, 0, 1), frame="world", shape="box", pose_msg=None):
        pose_stamped = geometry_msgs.msg.PoseStamped()
        pose_stamped.header.frame_id = frame
        if pose_msg is None:
            pose_stamped.pose.position.x = pos[0]
            pose_stamped.pose.position.y = pos[1]
            pose_stamped.pose.position.z = pos[2]
            pose_stamped.pose.orientation.x = rot[0]
            pose_stamped.pose.orientation.y = rot[1]
            pose_stamped.pose.orientation.z = rot[2]
            pose_stamped.pose.orientation.w = rot[3]
        else:
            pose_stamped.pose = copy.copy(pose_msg)
        if shape == "box":
            self.scene.add_box(name, pose_stamped, dims)
        elif shape == "sphere":
            self.scene.add_sphere(name, pose_stamped, dims)
        else:
            raise NotImplementedError("Shape {} not supported.".format(shape))

    def remove_obstacle(self, name=None):
        # Removes everything if name is None
        self.scene.remove_world_object(name)

    def update_from_lula(self, max_wait_t=0.5):
        """
        Wait until we get a world message from the lula world server for
        object geometry.
        """
        self._update_world = True
        if max_wait_t > 0:
            start_t = rospy.Time.now().to_sec()
            while self._update_world:
                rospy.sleep(0.1)
                cur_t = rospy.Time.now().to_sec()
                if cur_t - start_t > max_wait_t:
                    return not self._update_world
        else:
            return True

# For testing
from brain2.utils.pose import make_pose

if __name__ == '__main__':
    rospy.init_node('moveit_lula_bridge')
    args = parse.parse_kitchen_args(sim=1, lula=0)
    if args.lula:
        franka = Franka(is_physical_robot=(not args.sim))
    else:
        franka = None
    bridge = MoveitBridge(
        group_name='panda_arm',
        robot_interface=franka,
        dilation=1.,
        verbose=0)
    bridge.remove()
    bridge.update_from_lula()
    bridge.add_box("table", frame="base_plane", dims=(2.2, 2.2, 0.08))
    bridge.add_box("back", frame="world",
                   pos=(-0.3, 0, 0), dims=(0.04, 2.2, 2.2))
    bridge.retract()

    rospy.sleep(1)
    bridge.close_gripper()
    rospy.sleep(1)
    bridge.open_gripper()
    rospy.sleep(2.)
    bridge.go_local(make_pose(([0.317, -0.250, 0.650], [-0.136, 0.899,
                                                            -0.397, 0.127])))
    bridge.close_gripper()
    bridge.goto([0.317, -0.250, 0.650], [-0.136, 0.899, -0.397, 0.127])
    bridge.freeze()
    rospy.sleep(2.)
    bridge.open_gripper()
    bridge.goto([0.500, 0.233, 0.445], [-0.169, 0.938, 0.301, 0.042])
    bridge.freeze()
    bridge.goto([0.443, -0.001, 0.2], [-0.169, 0.938, 0.301, 0.042])
    bridge.dilation = 5.
    bridge.goto([0.443, -0.001, 0.1], [-0.169, 0.938, 0.301, 0.042])

    # Kill lula
    pid = os.getpid()
    os.kill(pid, signal.SIGKILL)
