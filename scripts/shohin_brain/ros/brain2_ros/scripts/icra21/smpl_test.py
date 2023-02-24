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
import tf
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from std_srvs.srv import Empty
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker

from smpl_test_msgs.srv import GetPath
from std_srvs.srv import *

def parse_args():
    parser = argparse.ArgumentParser("TestAPP")
    parser.add_argument("--path", type=int, default=0, help="0 for full, >0 for test")
    if "HOME" in os.environ:
        default_path = os.path.join(os.environ["HOME"], 'src/brain_gym/assets/urdf')
    else:
        default_path = '../../../assets/urdf',
    parser.add_argument('--assets_path',
        default=default_path,
        help='assets path')
    parser.add_argument('--visualize', type=int, default=0,
            help='should we also visualize the environment if supported? Note'
                  ' that this will also slow down planning.')
    args, _ = parser.parse_known_args()
    return args

def get_test_path_params(path):
    """ Get path from the default set up """
    if path == 1:
        goal = 0.40, 0.09, 0
        obs1 = 0.560487, 0.249063, 0
        obs2 = 0.212562, -0.041682, 0
    elif path == 2:
        goal = 0.40, 0.09, 0
        obs1 = 0.223469, -0.105595, 0
        obs2 = 0.296317, 0.280846, 0
    elif path == 3:
        goal = 0.40, 0.09, 0
        obs1 = 0.547114, 0.233310, 0
        obs2 = 0.341370, -0.107356, 0
    elif path == 4:
        goal = 0.3, -0.15, 0.1
        obs1 = 0.446718, -0.193827, 0
        obs2 = 0.456240, -0.095517, 0
    elif path == 5:
        goal = 0.5, -0, 0
        obs1 = 0.685, -0.03, 0
        obs2 = 0.345, -0.01, 0
    else:
        raise RuntimeError('could not parse path with id ' + str(path))

    return goal, obs1, obs2

def _to_pose(pos):
    pose1 = Pose()
    pose1.position.x = pos[0]
    pose1.position.y = pos[1]
    pose1.position.z = pos[2]
    pose1.orientation.w = 1.
    return pose1

def _to_obs_msg(*all_obs):
    msg = PoseArray()
    msg.header.stamp = rospy.Time.now()
    for obs in all_obs:
        pose = _to_pose(obs)
        msg.poses.append(pose)
    return msg

def _to_goal_msg(pos):
    msg = PoseStamped()
    msg.header.stamp = rospy.Time.now()
    msg.pose = _to_pose(pos)
    return msg

def query_test_path(get_path, goal, obs1, obs2):
    g = _to_goal_msg(goal)
    o = _to_obs_msg(obs1, obs2)
    return get_path(goal_pose=g, obstacle_poses=o)

def _send_goal_pos(pos):
    print("Sending goal =", pos)
    msg = _to_goal_msg(pos)
    goal_pub.publish(msg)

def _send_obs(pos1, pos2):
    print("sending obs =", pos1, pos2)
    msg = _to_obs_msg(pos1, pos2)
    obs_pub.publish(msg)

def _sample_goal():
    r = np.random.random() * (np.array([0.3, 0.3, 0]))
    r += np.array([0.3, -0.15, 0])
    return r

def _sample_obs(goal):
    r1 = np.random.random(2) * 2 - 1
    r2 = np.random.random(2) * 2 - 1
    r1 = r1 / np.linalg.norm(r1)
    r2 = r2 / np.linalg.norm(r2)
    r1 *= 0.201
    r2 *= 0.201
    assert(np.linalg.norm(r1) > 0.2)
    r1 = goal + np.array([r1[0], r1[1], 0])
    r2 = goal + np.array([r2[0], r2[1], 0])
    return r1, r2

from moveit_msgs.msg import RobotState
from moveit_msgs.msg import DisplayTrajectory
from moveit_msgs.msg import RobotTrajectory
def show_trajectory(publisher, joint_trajectory, q0):

    # create messages
    base_frame = "base_link"
    display_trajectory = DisplayTrajectory()
    display_trajectory.trajectory_start = RobotState()
    display_trajectory.trajectory_start.joint_state.name = joint_trajectory.joint_names
    display_trajectory.trajectory_start.joint_state.position = q0
    robot_state_trajectory = RobotTrajectory(joint_trajectory=joint_trajectory)

    robot_state_trajectory.joint_trajectory.header.frame_id = base_frame
    display_trajectory.trajectory.append(robot_state_trajectory)
    publisher.publish(display_trajectory)

  
def get_objects(publisher, listener, goal="bowl", poserbpf=False):
    rospy.sleep(0.1)

    if poserbpf:
        objects = {
                "bowl": "poserbpf/024_bowl_0",
                "pitcher1": "poserbpf/019_pitcher_base_0",
                "pitcher2": "poserbpf/019_pitcher_base_1",
                }
    else:
        objects = {
                "bowl": "024_bowl_01",
                "pitcher1": "019_pitcher_base_01",
                "pitcher2": "019_pitcher_base_02",
                #"mustard": "poserbpf/006_mustard_bottle_0",
                #"soup": "poserbpf/005_tomato_soup_can_0",
                #"mug1": "poserbpf/025_mug_0",
                #"mug2": "poserbpf/025_mug_1",
                #"mug3": "poserbpf/025_mug_2",
                }

    base_frame = "base_link"
    obs = []
    goal_pos = None
    markers = MarkerArray()
    for i, (obj, tf_frame) in enumerate(objects.items()):
        try:
            pos, rot = listener.lookupTransform(base_frame, tf_frame,
                    rospy.Time(0))
            if pos[2] < 0:
                print("ERROR: all object poses should be above the table")
                return None
            marker = Marker()
            marker.pose = _to_pose(pos)
            marker.color.r = 1.
            marker.color.b = 0.
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.id = i
            marker.type = Marker.SPHERE
            if obj == goal:
                print("[GOAL]", obj, pos, rot)
                marker.color.g = 1.
                goal_pos = pos
            else:
                print("[obs]", obj, pos, rot)
                marker.color.g = 0.
                obs.append(pos)
            marker.color.a = 0.5
            marker.header.stamp = rospy.Time.now()
            marker.header.frame_id = base_frame
            markers.markers.append(marker)
        except Exception as e:
            rospy.logwarn(str(e))
            return None
    publisher.publish(markers)

    for pos in obs:
        a = np.array(pos)
        b = np.array(goal_pos)
        dist = np.linalg.norm(a - b)
        if dist < 0.24:
            print("!!!! TOO CLOSE TOGETHER !!!!!")
            return None

    return _to_goal_msg(goal_pos), _to_obs_msg(*obs)

# ---------------------------------------------------------
from sensor_msgs.msg import JointState
dq = None
q = None
def _cb(msg):
    global q; global dq
    q = msg.position
    dq = np.max(msg.velocity)

def _is_moving():
    global dq
    return dq is not None and dq > 1e-3


# -------------
# Trajectory code
from brain2_ros.franka_gripper_control import FrankaGripperControl
from brain2.robot.domain import CartObjectsDomainDefinition
import brain2.bullet.problems as problems
from brain2_ros.trac_ik_solver import TracIKSolver
from brain2.bullet.ik import BulletIKSolver
from brain2_ros.robot import RosWorldStateObserver
from brain2_ros.video import VideoCapture

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

joint_names = ["panda_joint%d" % i for i in range(1,8)]
speed_factor = 0.8
acc_fract = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.25, 0.5])
reprofile = False

def follow_trajectory(client, robot, times, traj, pos, vel, acc):
    msg = JointTrajectory()
    msg.joint_names = joint_names
    # Add safety limit to the robot's range
    # APPLY PADDING?
    # ar = robot.ref.active_range * 0.01
    #qmin, qmax = (robot.ref.active_min #+ ar,
    #              robot.ref.active_max) #- ar)
    qmin = robot.ref.active_min
    qmax = robot.ref.active_max
    print("=== FOLLOWING TRAJECTORY ===")
    for i, (opos, t) in enumerate(zip(traj, times)):
        pt = JointTrajectoryPoint()
        q = pos(t)
        #q = opos
        if np.any(q <= qmin) or np.any(q >= qmax):
            raise RuntimeError("joint " + str(t) + " pos = " + str(q) + " was " +
                               "out of bounds.")
        #q[:7] = np.clip(q[:7], qmin, qmax)
        pt.positions = q
        pt.velocities = vel(t)
        pt.accelerations = np.clip(acc(t), -1*np.pi, np.pi)
        pt.time_from_start = rospy.Duration(t)
        #print("---", t, "---")
        print(i, "pos =", pt.positions)
        #print(pt.velocities)
        #print(pt.accelerations)
        msg.points.append(pt)
    print("===========================")

    # raw_input("[ READY ]")
    goal = FollowJointTrajectoryGoal(trajectory=msg)
    client.send_goal(goal)
    client.wait_for_result()
    return client.get_result()

def go_cfg(observer, client, qg):
    world_state = observer.observe(blocking=False)
    q0 = world_state["robot"].q
    print("curr =", q0)
    print("goal =", qg)
    res = JointSpaceLinearPlan(world_state["robot"], q0, qg,
                               step_size=0.2,
                               acceleration_fraction=acc_fract)
    if res is None:
        raise RuntimeError('could not go home directly for some reason')
    traj, times = res
    if len(traj) == 1:
        # No motion necessary
        print("[GONE HOME] Already at home")
        return None
    pos, vel, acc = compute_derivatives(traj, times, scale=speed_factor)
    res = follow_trajectory(client, world_state["robot"], times, traj, pos, vel, acc)
    print("[GONE HOME] Result = res")
    return res

def go_home(observer, client):
    return go_cfg(observer, client, home_q)

def exec_trajectory(observer, client, msg, reprofile=False):
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
        return follow_trajectory(client, world_state["robot"], times, traj, pos, vel, acc)
    else:
        goal = FollowJointTrajectoryGoal(trajectory=msg)
        qmin = robot.active_min
        qmax = robot.active_max
        for pt in msg.points:
            ok = robot.validate(q=pt.positions, verbose=True)
            ok = not (np.any(pt.positions <= qmin)
                    or np.any(pt.positions >= qmax))
            print("check", pt.positions, "=", ok)
            if not ok:
                raw_input('FAILED - PRESS ENTER')
                raise RuntimeError('tried to execute invalid trajectory')
        # raw_input('----')
        client.send_goal(goal)
        client.wait_for_result()
        return client.get_result()

def reverse_trajectory(traj):
    traj2 = JointTrajectory()
    traj2.points = traj.points[::-1]
    for pt in traj2.points:
        pt.velocities = [-1*v for v in pt.velocities]
        pt.accelerations = [-1*v for v in pt.accelerations]
    return traj2

def wait_for_config(observer):
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        world_state = observer.observe(blocking=False)
        q0 = world_state["robot"].q
        if q0 is None:
            rate.sleep()
        else:
            break

import datetime
def write_query(query):
    now = datetime.datetime.now()
    filename = "query_app_%04d-%02d-%02d_%02d-%02d-%02d.txt" % (
            int(now.year),
            int(now.month),
            int(now.day),
            int(now.hour),
            int(now.minute),
            int(now.second))
    with file(filename, 'w') as f:
        f.write(str(query[0]) + "\n")
        f.write(str(query[1]) + "\n")

import brain2.utils.color as color

if __name__ == '__main__':
    rospy.init_node('smpl_geom_test_node')

    print("Wait for get_path service")
    rospy.wait_for_service('get_path')
    print("Service is active; start moving the robot")

    get_path = rospy.ServiceProxy('get_path', GetPath)
    js_sub = rospy.Subscriber("joint_states", JointState, _cb, queue_size=100)
    marker_pub = rospy.Publisher("tabletop_objects", MarkerArray, queue_size=1)
    display_pub = rospy.Publisher('/display_planned_path', DisplayTrajectory, queue_size=1)
    listener = tf.TransformListener()

    # --------------------------
    # Trajectory execution 
    # Create interfaces with real robot
    gripper = FrankaGripperControl()
    gripper.open()
    # Create world model, IK solver, and control interfaces
    args = parse_args()
    env = problems.franka_cart(args.assets_path, args.visualize, "d435",
                               padding=0.01)
    # WE really should not be using this
    base_link = "panda_link0"
    ee_link = "panda_hand"
    ik = BulletIKSolver()

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
    #home_q = np.array([0.1560555699455867,
    #                  -1.306638358404831,
    #                   -1.11498810450236,
    #                   -2.2451363150613366,
    #                   -0.3784684199359682,
    #                   1.5615516125625295,
    #                   -0.5023804819997814])
    home_q = np.array([0.19943905853615498, -0.6206339396677519,
        -0.6428357117217883, -1.794293103268272, -0.014381015713785401,
        1.488322779682169, 0.3773888988519837])
    handover_q = np.array([1.4447848188584311,
                           -1.2720416672839971,
                           -1.2444491554561414,
                           -1.8008493739549942,
                           -1.4544149992883777,
                           2.167397186305788,
                           1.0375492069097332,])

    client = actionlib.SimpleActionClient("/position_joint_trajectory_controller"
                                  "/follow_joint_trajectory",
                                  FollowJointTrajectoryAction)
    print("Connecting to action server...")
    client.wait_for_server()
    wait_for_config(observer)
    go_home(observer, client)
    at_home = True

    import timeit
    from timeit import default_timer
    from brain2.utils.info import logwarn, logerr, say, log, start_log, end_log
    root_filename = "app_" + datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    filename = root_filename + ".txt"
    start_log(filename)
    perception_ts = []
    planning_ts = []
    
    verbose = False
    poserbpf = False
    if poserbpf:
        reset_poserbpf = rospy.ServiceProxy('reset_poserbpf', Empty)
        pause_poserbpf = rospy.ServiceProxy('pause_poserbpf', Empty)
        unpause_poserbpf = rospy.ServiceProxy('unpause_poserbpf', Empty)
    else:
        reset_poserbpf = lambda: None
        pause_poserbpf = lambda: None
        unpause_poserbpf = lambda: None

    t_perception = None
    raw_input('ENTER TO START')

    video = None
    if video is None:
        filename = root_filename + ".mp4"
        video = VideoCapture(filename=filename)
    palette = color.PaleContrastToRed

    # Get things to do
    args = parse_args()
    t0 = timeit.default_timer()
    if args.path > 0:
        goal, obs1, obs2 = get_test_path_params(args.path)
        res = query_test_path(get_path, goal, obs1, obs2)
        print("Path test result = ", res)
    else:
        rate = rospy.Rate(10)
        have_obj = False
        while not rospy.is_shutdown():
            rc, mc = palette[1]
            video.annotate(msg="Updating...", rectangle_color=rc, msg_color=mc)

            world_state = observer.observe(blocking=False)
            q0 = world_state["robot"].q
            if q0 is None:
                continue
            elif not at_home:
                rc, mc = palette[2]
                video.annotate(msg="Going home...", rectangle_color=rc,
                               msg_color=mc)
                log("===============================")
                log("going home:", default_timer() - t0)
                go_home(observer, client)
                at_home = True
                log("at home:", default_timer() - t0)
                unpause_poserbpf()
                reset_poserbpf()
                if poserbpf:
                    rospy.sleep(2.)
                    reset_poserbpf()
                continue

            # Query twice to make sure the objects are stabilized
            rc, mc = palette[1]
            video.annotate(msg="Detecting objects...", rectangle_color=rc,
                    msg_color=mc)
            unpause_poserbpf()
            log("perception:", default_timer() - t0)
            if t_perception is None:
                t_perception = default_timer()
            query = get_objects(marker_pub, listener, "bowl", poserbpf)
            if query is None:
                rate.sleep()
                continue

            # Record perception information
            log("done perception:", default_timer() - t0)
            t_perception = default_timer() - t_perception
            perception_ts.append(t_perception)
            log("perception took:", t_perception)
            log("avg perception time:", np.mean(perception_ts), np.std(perception_ts))
            t_perception = None

            #rospy.sleep(1.)
            #query = get_objects(marker_pub, listener, "bowl", poserbpf)
            #if query is None:
            #    rate.sleep()
            #    continue

            log("goal =", query[0].pose.position.x, query[0].pose.position.y)
            log("obj x =", [pose.position.x for pose in query[1].poses])
            log("obj y =", [pose.position.y for pose in query[1].poses])
            # write_query(query)

            assert(len(query[1].poses) >= 2)
            tp = default_timer()
            log("planning:", default_timer() - t0)
            rc, mc = palette[0]
            video.annotate(msg="Planning...", rectangle_color=rc, msg_color=mc)
            res = get_path(goal_pose=query[0], obstacle_poses=query[1])
            rc, mc = palette[3]
            video.annotate(msg="Following trajectory...", rectangle_color=rc,
                           msg_color=mc)
            tp = default_timer() - tp
            log("done planning:", default_timer() - t0)
            planning_ts.append(tp)
            log("planning took:", tp)
            log("avg planning time:", np.mean(planning_ts), np.std(planning_ts))
            if len(res.trajectory_fwd.points) < 1:
                print("!!! CANNOT DO THIS !!!")
                rospy.sleep(1.)
                continue

            pause_poserbpf()
            show_trajectory(display_pub, res.trajectory_fwd, q0)
            if verbose:
                print("Trajectory of length:", len(res.trajectory_fwd.points))
                print("SHOW BEFORE MOVING")
                for i, pt in enumerate(res.trajectory_fwd.points):
                    print(pt.positions)
            #raw_input('wait to execute')
            log("execute fwd:", default_timer() - t0)
            exec_trajectory(observer, client, res.trajectory_fwd,
                            reprofile=False)
            
            # Close the gripper
            rc, mc = palette[4]
            video.annotate(msg="Closing gripper...", rectangle_color=rc,
                           msg_color=mc)
            log("close gripper:", default_timer() - t0)
            gripper.close()

            # Reverse the trajectory
            rc, mc = palette[3]
            video.annotate(msg="Following trajectory...", rectangle_color=rc,
                           msg_color=mc)
            log("displaying bwd:", default_timer() - t0)
            world_state = observer.observe(blocking=False)
            q0 = world_state["robot"].q
            #traj2 = reverse_trajectory(res.trajectory)
            traj2 = res.trajectory_bwd
            show_trajectory(display_pub, traj2, q0)
            log("execute bwd:", default_timer() - t0)
            exec_trajectory(observer, client, traj2, reprofile=False)

            log("done bwd:", default_timer() - t0)

            rc, mc = palette[5]
            video.annotate(msg="Handing to human...", rectangle_color=rc,
                           msg_color=mc)
            rospy.sleep(0.2)
            go_cfg(observer, client, handover_q)
            log("open gripper:", default_timer() - t0)
            gripper.open()
            at_home = False
            log("DONE CYCLE:", default_timer() - t0)

        end_log()

    if video is not None:
        video.close()
