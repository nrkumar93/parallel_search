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

from brain2_msgs.srv import GetPath
from brain2_ros.utils import make_display_trajectory_pub
from brain2_ros.utils import show_trajectory
from brain2.utils.pose import make_pose

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
  
objects = {
    #"mustard": "poserbpf/006_mustard_bottle_0",
    #"soup": "poserbpf/005_tomato_soup_can_0",
    #"bowl": "poserbpf/024_bowl_0",
    #"mug": "poserbpf/025_mug_0",
    #"mug": "poserbpf/025_mug_1",
    #"mug": "poserbpf/025_mug_2",
    "bowl": "00_bowl",
    "mug": "00_mug",
    "mug": "00_mug_2",
        }
def get_objects(publisher, listener, goal="bowl"):
    base_frame = "base_link"
    obs = []
    goal_pos = None
    markers = MarkerArray()
    for i, (obj, tf_frame) in enumerate(objects.items()):
        try:
            pos, rot = listener.lookupTransform(base_frame, tf_frame,
                    rospy.Time(0))
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
    return _to_goal_msg(goal_pos), _to_obs_msg(*obs)

def get_base(listener):
    base_frame = "ue_world"
    tf_frame = "base_link"
    rate = rospy.Rate(10)
    while True:
        try:
            pos, rot = listener.lookupTransform(base_frame, tf_frame,
                    rospy.Time(0))
            return make_pose(pos, rot)
        except Exception as e:
            rospy.logwarn(str(e))
        rate.sleep()

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
from brain2_ros.lula_franka_control import LulaFrankaControl
from brain2.robot.domain import CartObjectsDomainDefinition
import brain2.bullet.problems as problems
from brain2_ros.trac_ik_solver import TracIKSolver
from brain2.bullet.ik import BulletIKSolver
from brain2_ros.robot import RosWorldStateObserver

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

# Temporary thing here
from brain2_ros.moveit import MoveitBridge

def follow_trajectory(client, robot, times, traj, pos, vel, acc, no_unsafe=False):
    msg = JointTrajectory()
    msg.joint_names = joint_names
    # Add safety limit to the robot's range
    # APPLY PADDING?
    ar = robot.ref.active_range * 0.01
    qmin, qmax = (robot.ref.active_min + ar,
                  robot.ref.active_max - ar)
    # For debugging purposes
    # print(qmin, qmax)
    for opos, t in zip(traj, times):
        pt = JointTrajectoryPoint()
        q = pos(t)
        #q = opos
        if (np.any(q < qmin) or np.any(q > qmax)) and no_unsafe:
            print(q)
            print(qmin)
            print(qmax)
            raise RuntimeError()
        #q[:7] = np.clip(q[:7], qmin, qmax)
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
    #client.execute_joint_trajectory([pt.positions for pt in msg.points])
    client.execute(msg)
    return True

def go_cfg(observer, client, qg):
    world_state = observer.observe(blocking=False)
    q0 = world_state["robot"].q
    print("curr =", q0)
    print("goal =", qg)
    res = JointSpaceLinearPlan(world_state["robot"], q0, qg,
                               step_size=0.1,
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
        for pt in msg.points:
            print("check", pt.positions, "=", robot.validate(q=pt.positions))
        #client.execute_joint_trajectory([pt.positions for pt in msg.points])
        client.execute(msg)
        return True

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
            print("waiting for cfg to arrive")
            rate.sleep()
        else:
            break

def set_positions(base_pose, sim, min_dim=0.2, move_bowl=True):

    objs = ["bowl_1", "mug_1", "mug_2", "mug_3"]
    for i, obj in enumerate(objs):
        offset = np.eye(4)
        offset[0,3] = -0.25 - 0.21 * i
        offset[2,3] = 0.05
        pose = base_pose.dot(offset)
        if i > 0 or move_bowl:
            sim.set_pose(obj, pose, do_correction=False)

    bx, by = np.random.random(2)
    by = (by - 0.5) * 0.4
    bx = bx * 0.2 + 0.4
    print("bowl at", bx, by)
    offset = np.eye(4)
    offset[0,3] = bx
    offset[1,3] = by
    offset[2,3] = 0.05

    if move_bowl:
        sim.set_pose(objs[0], base_pose.dot(offset), do_correction=False)

    mugs = []
    for i in range(2):
        ok = False
        while not ok:
            direction = np.random.random(2) - 0.5
            direction = direction / np.linalg.norm(direction)
            distance = np.random.random() * 0.2 + min_dim + 0.001
            x = distance[0] * np.cos(direction[0])
            y = distance[1] * np.cos(direction[1])
            print("try placing mug at ", x, y)
            x = x + 0.6
            ok = True
            if np.linalg.norm([x - bx, y - by]) < min_dim:
                print("too close to bowl")
                ok = False
                continue
            for ox, oy in mugs:
                if np.linalg.norm([ox - x, oy -y]) < 0.1:
                    ok = False
                    print("too close to obj")
                    break
            if ok:
                mugs.append((x, y))
    for (x, y,), name in zip(mugs, objs[1:]):
        offset = np.eye(4)
        offset[0,3] = x
        offset[1,3] = y
        offset[2,3] = 0.05
        sim.set_pose(name, base_pose.dot(offset), do_correction=False)

    rospy.sleep(1.)
    return bx, by

def wait_for_position(observer, position, max_t=30., actor="robot"):
    rate = rospy.Rate(10)
    position = np.array(position)
    t0 = rospy.Time.now()
    while not rospy.is_shutdown():
        world_state = observer.observe(blocking=False)
        current = world_state["robot"].q
        error = np.linalg.norm(current - position)
        t = (rospy.Time.now() - t0).to_sec()
        print(t, actor, "position error =", error)
        if error < 0.02:
            break
        elif t > max_t:
            break
        rate.sleep()

if __name__ == '__main__':
    rospy.init_node('smpl_geom_test_node')

    print("Wait for get_path service")
    rospy.wait_for_service('get_path')
    print("Service is active; start moving the robot")

    get_path = rospy.ServiceProxy('get_path', GetPath)
    js_sub = rospy.Subscriber("joint_states", JointState, _cb, queue_size=100)
    marker_pub = rospy.Publisher("tabletop_objects", MarkerArray, queue_size=1)
    display_pub = make_display_trajectory_pub()
    listener = tf.TransformListener()

    sim = True

    # --------------------------
    # Trajectory execution 
    if not sim:
        # Create interfaces with real robot
        gripper = FrankaGripperControl()
        gripper.open()
    else:
        gripper = None
    # Create world model, IK solver, and control interfaces
    args = parse_args()
    env = problems.franka_cart(args.assets_path, args.visualize, "d435")
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
    home_q = np.array([0.1560555699455867,
                       -1.306638358404831,
                       -1.11498810450236,
                       -2.2451363150613366,
                       -0.3784684199359682,
                       1.5615516125625295,
                       -0.5023804819997814])

    ee_link = "right_gripper"
    if not sim:
        client = LulaFrankaControl(base_link, ee_link, 7)
    else:
        # Isaac sim manager
        from isaac_bridge.manager import SimulationManager
        sim = SimulationManager(lula=False)
        # MoveIt Bridge
        client = MoveitBridge(group_name='panda_arm',
                              robot_interface=None, #franka,
                              dilation=1.,
                              verbose=0,
                              use_lula=False)
        client.open_gripper()

    wait_for_config(observer)
    go_home(observer, client)
    at_home = True
    client.open_gripper()
    base_pose = get_base(listener)
    print("Base pose in sim coords:")
    print(base_pose)
    # raw_input('wait to execute - at home')

    # Get things to do
    args = parse_args()
    if args.path > 0:
        goal, obs1, obs2 = get_test_path_params(args.path)
        res = query_test_path(get_path, goal, obs1, obs2)
        print(res)
    else:
        rate = rospy.Rate(10)
        have_obj = False
        while not rospy.is_shutdown():
            go_home(observer, client)
            set_positions(base_pose, sim)
            world_state = observer.observe(blocking=False)
            q0 = world_state["robot"].q
            if q0 is None:
                continue
            elif not at_home:
                go_home(observer, client)
                at_home = True
                continue
            
            query = get_objects(marker_pub, listener, "bowl")
            if query is not None:
                res = get_path(goal_pose=query[0], obstacle_poses=query[1])
                if len(res.trajectory.points) < 1:
                    print("!!! CANNOT DO THIS !!!")
                    rospy.sleep(1.)
                    continue

                world_state["robot"].q = res.trajectory.points[-1].positions
                env.update(world_state)
                pose = env.get_object("robot").get_ee_pose()
                offset = np.eye(4)
                offset[1,3] = 0.03
                offset[2,3] = 0.03
                pose = pose.dot(offset)
                #cfg = ik(world_state["robot"].ref, pose, q0=world_state["robot"].q)
                #ee_ref = world_state["robot"].ref.ee_ref
                #ee_ref.set_pose_matrix(pose)
                #for pt in res.trajectory.points:
                #    print(pt.positions)
                #print("to arm config =", cfg)

                show_trajectory(display_pub, res.trajectory, q0)
                print("Trajectory of length:", len(res.trajectory.points))
                res.trajectory.points = res.trajectory.points[:-1]
                print(res)
                t0 = -1
                for i, pt in enumerate(res.trajectory.points):
                    print(pt.time_from_start.to_sec())
                #raw_input('wait to execute')
                exec_trajectory(observer, client, res.trajectory,
                                reprofile=False)
                wait_for_position(observer, res.trajectory.points[-1].positions)

                #raw_input('wait to execute - go to grasp position')
                client.go_local(T=pose, wait=True)
                rospy.sleep(1.5)

                world_state = observer.observe(blocking=False)
                q0 = world_state["robot"].q
                print("q =", q0)
                print("goal was", res.trajectory.points[-1].positions)
                g = res.trajectory.points[-1].positions
                # print(np.linalg.norm(q0-g))
                
                # 1cm forward and grasp
                client.close_gripper()
                world_state = observer.observe(blocking=False)
                q0 = world_state["robot"].q

                # Reverse the trajectory
                traj2 = reverse_trajectory(res.trajectory)
                show_trajectory(display_pub, traj2, q0)
                rospy.sleep(0.5)
                exec_trajectory(observer, client, traj2, reprofile=True)
                wait_for_position(observer, traj2.points[-1].positions, max_t=2.)

                # Move objects around for placement
                x, y = set_positions(base_pose, sim, move_bowl=False)
                query = get_objects(marker_pub, listener, "bowl")
                res = get_path(goal_pose=_to_goal_msg((x, y, 0.02)),
                               obstacle_poses=query[1])

                # Placement trajectory
                world_state = observer.observe(blocking=False)
                q0 = world_state["robot"].q
                show_trajectory(display_pub, res.trajectory, q0)
                exec_trajectory(observer, client, res.trajectory, reprofile=False)
                wait_for_position(observer, res.trajectory.points[-1].positions)

                # Retreat after placing
                world_state = observer.observe(blocking=False)
                q0 = world_state["robot"].q
                traj2 = reverse_trajectory(res.trajectory)
                #gripper.open()
                client.open_gripper()
                show_trajectory(display_pub, traj2, q0)
                rospy.sleep(0.5)
                exec_trajectory(observer, client, traj2, reprofile=True)
                wait_for_position(observer, traj2.points[-1].positions)

            rate.sleep()

