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


def get_joint_names(robot_state):
    return robot_state.ref.get_joint_names()


def get_max_velocities(robot_state):
    return np.array(robot_state.ref.get_max_velocities())


def get_duration_fn(robot, velocities=None, norm=INF):
    """ Duractions for the motion waypoints """
    if velocities is None:
        velocities = robot.ref.get_max_velocities()

    # difference_fn = get_difference_fn(body, joints)
    # No circular joints for now
    def fn(q1, q2):
        # distance = np.array(difference_fn(q2, q1))
        distance = q2 - q1
        duration = np.divide(distance, velocities)
        return np.linalg.norm(duration, ord=norm)
    return fn

def ensure_increasing(path, time_from_starts):
    assert len(path) == len(time_from_starts)
    for i in reversed(range(1, len(path))):
        if time_from_starts[i-1] == time_from_starts[i]:
            path.pop(i)
            time_from_starts.pop(i)

def instantaneous_retime_path(robot, path, speed=ARM_SPEED):
    #duration_fn = get_distance_fn(robot, joints)
    duration_fn = get_duration_fn(robot)
    mid_durations = [duration_fn(*pair) for pair in zip(path[:-1], path[1:])]
    durations = [0.] + mid_durations
    time_from_starts = np.cumsum(durations) / speed
    return time_from_starts


################################################################################


def instantaneous_retime_path(robot, path, speed=ARM_SPEED):
    #duration_fn = get_distance_fn(robot, joints)
    duration_fn = get_duration_fn(robot)
    mid_durations = [duration_fn(*pair) for pair in zip(path[:-1], path[1:])]
    durations = [0.] + mid_durations
    time_from_starts = np.cumsum(durations) / speed
    return time_from_starts


def slow_trajectory(robot, path, **kwargs):
    min_fraction = 0.1 # percentage
    ramp_duration = 1.0 # seconds
    # path = waypoints_from_path(path) # Neither moveit or lula benefit from this

    time_from_starts = instantaneous_retime_path(robot, path, **kwargs)
    mid_times = [np.average(pair) for pair in zip(time_from_starts[:-1], time_from_starts[1:])]
    mid_durations = [t2 - t1 for t1, t2 in zip(time_from_starts[:-1], time_from_starts[1:])]
    new_time_from_starts = [0.]
    for mid_time, mid_duration in zip(mid_times, mid_durations):
        time_from_start = mid_time - time_from_starts[0]
        up_fraction = clip(time_from_start / ramp_duration, min_value=min_fraction, max_value=1.)
        time_from_end = time_from_starts[-1] - mid_time
        down_fraction = clip(time_from_end / ramp_duration, min_value=min_fraction, max_value=1.)
        new_fraction = min(up_fraction, down_fraction)
        new_duration = mid_duration / new_fraction
        #print(new_time_from_starts[-1], up_fraction, down_fraction, new_duration)
        new_time_from_starts.append(new_time_from_starts[-1] + new_duration)
    # print(time_from_starts)
    # print(new_time_from_starts)
    # raw_input('Continue?)
    # time_from_starts = new_time_from_starts
    return new_time_from_starts

################################################################################

def compute_min_duration(distance, max_velocity, acceleration):
    if distance == 0:
        return 0
    max_ramp_duration = max_velocity / acceleration
    ramp_distance = 0.5 * acceleration * math.pow(max_ramp_duration, 2)
    remaining_distance = distance - 2 * ramp_distance
    if 0 <= remaining_distance:  # zero acceleration
        remaining_time = remaining_distance / max_velocity
        total_time = 2 * max_ramp_duration + remaining_time
    else:
        half_time = np.sqrt(distance / acceleration)
        total_time = 2 * half_time
    return total_time

def compute_ramp_duration(distance, max_velocity, acceleration, duration):
    discriminant = max(0, math.pow(duration * acceleration, 2) - 4 * distance * acceleration)
    velocity = 0.5 * (duration * acceleration - math.sqrt(discriminant))  # +/-
    #assert velocity <= max_velocity
    ramp_time = velocity / acceleration
    predicted_distance = velocity * (duration - 2 * ramp_time) + acceleration * math.pow(ramp_time, 2)
    assert abs(distance - predicted_distance) < 1e-6
    return ramp_time

def compute_position(ramp_time, max_duration, acceleration, t):
    velocity = acceleration * ramp_time
    max_time = max_duration - 2 * ramp_time
    t1 = clip(t, 0, ramp_time)
    t2 = clip(t - ramp_time, 0, max_time)
    t3 = clip(t - ramp_time - max_time, 0, ramp_time)
    #assert t1 + t2 + t3 == t
    return 0.5 * acceleration * math.pow(t1, 2) + velocity * t2 + \
           (velocity * t3 - 0.5 * acceleration * math.pow(t3, 2))

def ramp_retime_path(path, max_velocities, acceleration_fraction=1.5, sample_step=None):
    assert np.all(max_velocities)
    accelerations = max_velocities * acceleration_fraction
    #dim = len(max_velocities)
    dim = len(path[0])
    #difference_fn = get_difference_fn(robot, joints)
    # TODO: more fine grain when moving longer distances

    # Assuming instant changes in accelerations
    waypoints = [path[0]]
    # time_from_starts = [0.]
    timings = np.zeros(len(path))
    for i, (q1, q2) in enumerate(zip(path[:-1], path[1:])):
        # assumes not circular anymore
        differences = q2 - q1
        #differences = difference_fn(q1, q2)
        distances = np.abs(differences)
        duration = 0
        for idx in range(dim):
            total_time = compute_min_duration(distances[idx], max_velocities[idx], accelerations[idx])
            duration = max(duration, total_time)

        #time_from_start = time_from_starts[-1]
        #if sample_step is not None:
        #    ramp_durations = [compute_ramp_duration(distances[idx], max_velocities[idx], accelerations[idx], duration)
        #                      for idx in range(dim)]
        #    directions = np.sign(differences)
        #    for t in np.arange(sample_step, duration, sample_step):
        #        positions = []
        #        for idx in range(dim):
        #            distance = compute_position(ramp_durations[idx], duration, accelerations[idx], t)
        #            positions.append(q1[idx] + directions[idx] * distance)
        #        waypoints.append(positions)
        #        time_from_starts.append(time_from_start + t)
        #        timings[i+1] = time_from_start + t
        waypoints.append(q2)
        # time_from_starts.append(time_from_start + duration)
        timings[i+1] = timings[i] + duration

    return waypoints, timings


def get_length(vec, norm=2):
    # TODO remove this 
    return np.linalg.norm(vec, ord=norm)


def get_unit_vector(vec, norm=2):
    #norm = get_length(vec)
    norm = np.linalg.norm(vec, ord=norm)
    if norm == 0:
        return vec
    return np.array(vec) / norm


def adjust_path(robot_state, path):
    """ Get a path based on current configuration -- ends in same place
    relative to the start though."""
    start_positions = robot_state.q
    # difference_fn = get_difference_fn(robot, joints)
    # differences = [difference_fn(q2, q1) for q1, q2 in zip(path, path[1:])]
    differences = [q2 - q1 for q1, q2 in zip(path, path[1:])]
    adjusted_path = [np.array(start_positions)]
    for difference in differences:
        adjusted_path.append(adjusted_path[-1] + difference)
    return adjusted_path


def remove_redundant(path, tolerance=1e-3):
    """ Get rid of unnecessary trajectory points in the path """
    assert path
    new_path = [path[0]]
    for conf in path[1:]:
        difference = np.array(new_path[-1]) - np.array(conf)
        # print("!", difference)
        # print(len(new_path))
        # print(np.allclose(np.zeros(len(difference)), difference))
        # raw_input()
        if not np.allclose(np.zeros(len(difference)), difference,
                           atol=tolerance,
                           rtol=0):
            new_path.append(conf)
    return new_path


def waypoints_from_path(path, tolerance=1e-3):
    """
    Turn path into a list of positions. Removes anything that's substantially
    different, so this will be an extremely minimal path. If we need that.
    """
    path = remove_redundant(path, tolerance=tolerance)
    if len(path) < 2:
        return path
    difference_fn = lambda q2, q1: np.array(q2) - np.array(q1)
    #difference_fn = get_difference_fn(body, joints)

    waypoints = [path[0]]
    last_conf = path[1]

    # Check to see if there's a difference here between the different
    # directions -- if not, do not add. Straight lines to goals do not
    # get added here.
    last_difference = get_unit_vector(difference_fn(last_conf, waypoints[-1]))
    for conf in path[2:]:
        difference = get_unit_vector(difference_fn(conf, waypoints[-1]))
        if not np.allclose(last_difference, difference, atol=tolerance, rtol=0):
            waypoints.append(last_conf)
            difference = get_unit_vector(difference_fn(conf, waypoints[-1]))
        last_conf = conf
        last_difference = difference
    waypoints.append(last_conf)
    return waypoints


def retime_trajectory(robot_state, path, velocity_fraction=DEFAULT_SPEED_FRACTION,
                      make_waypoints=False,
                      acceleration_fraction=1.,
                      adjust=False):
    """
    :param robot_state:
    :param path:
    :param velocity_fraction: fraction of max_velocity
    :param acceleration_fraction: fraction of velocity_fraction*max_velocity per second
    :param sample_step:
    :return:
    """
    if adjust:
        path = adjust_path(robot_state, path)
    # print("adjust path =", path)
    if make_waypoints:
        path = waypoints_from_path(path)
    else:
        path = remove_redundant(path)
    # print("stripped path =", path)
    max_velocities = get_max_velocities(robot_state)
    waypoints, time_from_starts = ramp_retime_path(path, max_velocities,
            acceleration_fraction=acceleration_fraction)
    return waypoints, time_from_starts

################################################################################

def linear_parameterization(robot_state, joints, path, speed=ARM_SPEED):
    import rospy
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    distance_fn = get_distance_fn(robot_state, joints)
    distances = [0] + [distance_fn(*pair) for pair in zip(path[:-1], path[1:])]
    time_from_starts = np.cumsum(distances) / speed

    # https://en.wikipedia.org/wiki/Finite_difference
    trajectory = JointTrajectory()
    trajectory.header.frame_id = "base_link"
    trajectory.header.stamp = rospy.Time(0)
    trajectory.joint_names = get_joint_names(robot_state, joints)
    for i in range(len(path)):
       point = JointTrajectoryPoint()
       point.positions = list(path[i])
       # Don't need velocities, accelerations, or efforts
       #vector = np.array(path[i]) - np.array(path[i-1])
       #duration = (time_from_starts[i] - time_from_starts[i-1])
       #point.velocities = list(vector / duration)
       #point.accelerations = list(np.ones(len(joints)))
       #point.effort = list(np.ones(len(joints)))
       point.time_from_start = rospy.Duration(time_from_starts[i])
       trajectory.points.append(point)
    return trajectory

def spline_trajectory(robot_state, path, **kwargs):
    path, time_from_starts = retime_trajectory(robot_state, path, **kwargs)
    #ensure_increasing(path, time_from_starts)
    #positions = interp1d(time_from_starts, path, kind='linear')
    positions = CubicSpline(time_from_starts, path, bc_type='clamped', # clamped | natural
                            extrapolate=False) # bc_type=((1, 0), (1, 0))
    #positions = CubicHermiteSpline(time_from_starts, path, extrapolate=False)
    velocities = positions.derivative(nu=1)
    accelerations = velocities.derivative(nu=1)

def compute_derivatives(trajectory, time_from_starts, scale=1):
    """ Get the velocities and accelerations we will be sending to the robot """
    if scale is not 1:
        time_from_starts *= scale
    positions = CubicSpline(time_from_starts, trajectory, bc_type='clamped', # clamped | natural
                            extrapolate=False) # bc_type=((1, 0), (1, 0))
    velocities = positions.derivative(nu=1)
    accelerations = velocities.derivative(nu=1)
    return positions, velocities, accelerations




