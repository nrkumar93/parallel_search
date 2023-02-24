# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import math
import numpy as np

from scipy.interpolate import CubicSpline # LinearNDInterpolator, NearestNDInterpolator, bisplev, bisplrep, splprep

# Based on the SS-Replan code from the summer
#ARM_SPEED = 0.15*np.pi # radians / sec
ARM_SPEED = 0.2 # percent
DEFAULT_SPEED_FRACTION = 0.3
INF = np.inf
PI = np.pi
CIRCULAR_LIMITS = -PI, PI
UNBOUNDED_LIMITS = -INF, INF
DEFAULT_TIME_STEP = 1./240. # seconds

import msg_tools
from brain2.robot.trajectory import *

################################################################################
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

def spline_parameterization(robot_state, path, time_step=None, **kwargs):

    # Can always interpolate each DOF independently
    # Univariate interpolation just means that the input is one dimensional (aka time)
    # The output can be arbitrary dimension
    # Bivariate interpolation has a 2D input space

    # Was initially using scipy 0.17.0
    # https://docs.scipy.org/doc/scipy-0.17.0/reference/interpolate.html
    # https://docs.scipy.org/doc/scipy-0.17.0/reference/tutorial/interpolate.html
    # Upgraded to scipy 0.18.0 to use the CubicSpline method
    # sudo pip2 install scipy==0.18.0
    # https://docs.scipy.org/doc/scipy-0.18.0/reference/interpolate.html

    # BPoly.from_derivatives
    # PPoly.from_spline # PPoly.from_bernstein_basis
    #path = list(path)
    #time_from_starts = retime_path(robot_state, joints, path, **kwargs)
    #time_from_starts = slow_trajectory(robot_state, joints, path, **kwargs)
    # TODO: interpolate through the waypoints
    path, time_from_starts = retime_trajectory(robot_state, path, **kwargs)
    #ensure_increasing(path, time_from_starts)
    #positions = interp1d(time_from_starts, path, kind='linear')
    positions = CubicSpline(time_from_starts, path, bc_type='clamped', # clamped | natural
                            extrapolate=False) # bc_type=((1, 0), (1, 0))
    #positions = CubicHermiteSpline(time_from_starts, path, extrapolate=False)
    velocities = positions.derivative(nu=1)
    accelerations = velocities.derivative(nu=1)
    # Could resample at this point
    # TODO: could try passing incorrect accelerations (bounded)

    #for i, t in enumerate(time_from_starts):
    #    print(i, t, path[i], positions(t), velocities(t), accelerations(t))
    #wait_for_user('Continue?')

    # https://github.com/scipy/scipy/blob/v1.3.0/scipy/interpolate/_cubic.py#L75-L158
    trajectory = JointTrajectory()
    trajectory.header.frame_id = "base_link"
    trajectory.header.stamp = rospy.Time(0)
    trajectory.joint_names = get_joint_names(robot_state)

    if time_step is not None:
        time_from_starts = np.append(np.arange(0, time_from_starts[-1], time_step), [time_from_starts[-1]])
    for t in time_from_starts:
        point = JointTrajectoryPoint()
        point.positions = positions(t) # positions alone is insufficient
        point.velocities = velocities(t)
        point.accelerations = accelerations(t) # accelerations aren't strictly needed
        #point.effort = list(np.ones(len(joints)))
        point.time_from_start = rospy.Duration(t)
        trajectory.points.append(point)
    #print((np.array(path[-1]) - np.array(trajectory.points[-1].positions)).round(5))
    return trajectory

def spline_trajectory(robot_state, path, timings, time_step=None):
    #ensure_increasing(path, time_from_starts)
    #positions = interp1d(time_from_starts, path, kind='linear')
    positions = CubicSpline(timings, path, bc_type='clamped', # clamped | natural
                            extrapolate=False) # bc_type=((1, 0), (1, 0))
    #positions = CubicHermiteSpline(time_from_starts, path, extrapolate=False)
    velocities = positions.derivative(nu=1)
    accelerations = velocities.derivative(nu=1)
    # Could resample at this point
    # TODO: could try passing incorrect accelerations (bounded)

    #for i, t in enumerate(time_from_starts):
    #    print(i, t, path[i], positions(t), velocities(t), accelerations(t))
    #wait_for_user('Continue?')

    # https://github.com/scipy/scipy/blob/v1.3.0/scipy/interpolate/_cubic.py#L75-L158
    trajectory = JointTrajectory()
    trajectory.header.frame_id = "base_link"
    trajectory.header.stamp = rospy.Time(0)
    trajectory.joint_names = get_joint_names(robot_state)

    if time_step is not None:
        time_from_starts = np.append(np.arange(0, time_from_starts[-1],
            time_step), [timings[-1]])
    for t in timings:
        point = JointTrajectoryPoint()
        point.positions = positions(t) # positions alone is insufficient
        point.velocities = velocities(t)
        point.accelerations = accelerations(t) # accelerations aren't strictly needed
        #point.effort = list(np.ones(len(joints)))
        point.time_from_start = rospy.Duration(t)
        trajectory.points.append(point)
    #print((np.array(path[-1]) - np.array(trajectory.points[-1].positions)).round(5))
    return trajectory



################################################################################


def follow_joint_trajectory(trajectory):
    action_topic = '/position_joint_trajectory_controller/follow_joint_trajectory'
    # /move_base_simple/goal
    # /execute_trajectory/goal
    # /position_joint_trajectory_controller/command
    client = SimpleActionClient(action_topic, FollowJointTrajectoryAction)
    print('Starting', action_topic)
    client.wait_for_server()
    client.cancel_all_goals()
    # time.sleep(0.1)
    print('Finished', action_topic)
    # TODO: create this action client once

    #error_threshold = 1e-3
    #threshold_template = '/position_joint_trajectory_controller/constraints/{}/goal'
    #for name in get_joint_names(robot_state, joints):
    #    param = threshold_template.format(name)
    #    rospy.set_param(param, error_threshold)
    #    #print(name, rospy.get_param(param))

    goal = FollowJointTrajectoryGoal(trajectory=trajectory)
    goal.goal_time_tolerance = rospy.Duration.from_sec(2.0)
    for joint in trajectory.joint_names:
        # goal.path_tolerance.append(JointTolerance(name=joint, position=1e-2)) # position | velocity | acceleration
        goal.goal_tolerance.append(JointTolerance(name=joint, position=1e-3))  # position | velocity | acceleration

    # https://github.mit.edu/Learning-and-Intelligent-Systems/ltamp_pr2/blob/master/control_tools/ros_controller.py
    while True:
        state = client.send_goal_and_wait(goal)  # send_goal_and_wait
        # state = client.get_state() # get_comm_state, get_terminal_state
        print('State:', state)
        # result = client.get_result()
        # print('Result:', result)
        # text = client.get_goal_status_text()
        text = GoalStatus.to_string(state)
        print('Goal status:', text)
        if state != GoalStatus.PREEMPTED:
            break
        # http://docs.ros.org/diamondback/api/actionlib/html/action__client_8py_source.html
        # https://docs.ros.org/diamondback/api/actionlib/html/simple__action__client_8py_source.html
    return True

def franka_control(robot_state, joints, path, interface, **kwargs):

    #joint_command_control(robot_state, joints, path, **kwargs)
    #follow_control(robot_state, joints, path, **kwargs)
    if interface.simulation: # or interface.moveit.use_lula:
        #return joint_state_control(robot_state, joints, path, interface)
        return moveit_control(robot_state, joints, path, interface)

    # /franka_control
    # https://github.com/frankaemika/franka_ros
    # https://github.mit.edu/Learning-and-Intelligent-Systems/ltamp_pr2/blob/master/control_tools/ros_controller.py
    # https://erdalpekel.de/?p=55
    # http://wiki.ros.org/joint_trajectory_controller
    # https://frankaemika.github.io/docs/franka_ros.html
    # http://docs.ros.org/indigo/api/moveit_tutorials/html/doc/pr2_tutorials/planning/src/doc/controller_configuration.html

    # rosservice call /controller_manager/list_controller_types
    # rosservice call /controller_manager/list_controllers
    # controller_manager_tests/VelEffController, effort_controllers/JointTrajectoryController,
    #   joint_state_controller/JointStateController, pos_vel_acc_controllers/JointTrajectoryController,
    #   pos_vel_controllers/JointTrajectoryController, position_controllers/JointTrajectoryController,
    #   velocity_controllers/JointTrajectoryController]

    update_robot_state_conf(interface)
    start_conf = get_joint_positions(robot_state, joints)
    print('Initial error:', (np.array(start_conf) - np.array(path[0])).round(5))
    # TODO: only add if the error is substantial
    #path = path
    path = [start_conf] + list(path)
    trajectory = spline_parameterization(robot_state, joints, path, **kwargs)
    total_duration = trajectory.points[-1].time_from_start.to_sec()
    print('Following {} {}-DOF waypoints in {:.3f} seconds'.format(
        len(trajectory.points), len(trajectory.joint_names), total_duration))
    # path_tolerance, goal_tolerance, goal_time_tolerance
    # http://docs.ros.org/diamondback/api/control_msgs/html/msg/FollowJointTrajectoryGoal.html
    publish_display_trajectory(interface.moveit, trajectory)
    #wait_for_user('Execute?')
    # TODO: adjust to the actual current configuration

    start_time = time.time()
    if interface.moveit.use_lula:
        lula_joint_trajectory(interface, trajectory)
    else:
        follow_joint_trajectory(trajectory)

    print("!!!!!!!!! NOT YET WORKING HERE")
    # TODO: different joint distance metric. The last joint seems to move slowly
    # TODO: extra effort to get to the final conf
    update_robot_state_conf(interface)
    end_conf = get_joint_positions(robot_state, joints)
    print('Target conf:', np.array(path[-1]).round(5))
    print('Final conf:', np.array(end_conf).round(5))
    print('Lower limits:', np.array(get_min_limits(robot_state, joints)).round(5))
    print('Upper limits:', np.array(get_max_limits(robot_state, joints)).round(5))
    error = np.array(end_conf) - np.array(path[-1])
    print('Final error:', error.round(5))
    magnitude = get_length(error)
    print('Error magnitude: {:.3f}'.format(magnitude))
    print('Execution took {:.3f} seconds (expected {:.3f} seconds)'.format(
        elapsed_time(start_time), total_duration))
    #print((np.array(path[-1]) - np.array(trajectory.points[-1].positions)).round(5))
    #wait_for_user('Continue?')
    # TODO: remove display messages
    # http://docs.ros.org/kinetic/api/actionlib_msgs/html/msg/GoalStatus.html

    max_error = 0.1
    return magnitude <= max_error
    #return state == GoalStatus.SUCCEEDED

