# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


#!/usr/bin/env python2

# From Jonathan Tremblay

#import roslib
import rospy
import numpy as np
from scipy.spatial.transform import Rotation as R
import tf2_ros as tf2
import numpy as np
from geometry_msgs.msg import Transform
from geometry_msgs.msg import TransformStamped
import geometry_msgs
import std_msgs
import pyquaternion
import copy

import argparse

# Example:
# python temporal_filter_tf_pose.py --tf_child obj_sugar

latest_tfcam_list = []
latest_tfcam_sent = None

# constants
max_len_sec = 5.  # number of seconds in the past to remember
half_life = 1.  # for exponential decay (in seconds)
alpha = 0.99  # coefficient for PD control
max_speed_trans = 0.1  # max translation speed (in meters per second)
max_speed_rot = 0.1  # max rotation speed (in delta quaternion value per second)

def on_transform_received(tf_cam):
    '''When a new transform message has been received, this method stores it for later processing (lazy evaluation).'''
    global latest_tfcam_list

    tf_cam.transform.rotation = ensure_quaternion_positive_sign(tf_cam.transform.rotation)
    latest_tfcam_list.insert(0, tf_cam)  # insert latest at beginning of list
        
    # Sort based on time (since timestamps might not arrive in order)
    def GetKeyForSorting(val):
        return val.header.stamp.to_sec()
    latest_tfcam_list.sort(key=GetKeyForSorting, reverse=True)

    # Remove oldest transforms
    time_latest = latest_tfcam_list[0].header.stamp.to_sec()
    while True:
        time = latest_tfcam_list[-1].header.stamp.to_sec()  # oldest transform time
        if time_latest - time > max_len_sec:
            latest_tfcam_list = latest_tfcam_list[:-1]
        else:
            break

def ensure_quaternion_positive_sign(quat):
    if quat.x < 0:
        quat.x *= -1
        quat.y *= -1
        quat.z *= -1
        quat.w *= -1
    return quat

def do_filter():
    '''Do exponential decay filter'''
    global latest_tfcam_list
    tau = half_life / np.log(2)
    n = len(latest_tfcam_list)
    if n > 0:
        latest_time = latest_tfcam_list[0].header.stamp.to_sec()
        tran = copy.deepcopy(latest_tfcam_list[0].transform)
        total_weight = 1.  # because exp(0)=1
        for i in range(1, n):
            time = latest_tfcam_list[i].header.stamp.to_sec()
            weight = np.exp(-(latest_time - time) / tau)
            lat = latest_tfcam_list[i].transform
            # translation
            tran.translation.x += weight * lat.translation.x
            tran.translation.y += weight * lat.translation.y
            tran.translation.z += weight * lat.translation.z
            # rotation
            tran.rotation.x += weight * lat.rotation.x
            tran.rotation.y += weight * lat.rotation.y
            tran.rotation.z += weight * lat.rotation.z
            tran.rotation.w += weight * lat.rotation.w
            # accumulate weight
            total_weight += weight
        # normalize
        tran.translation.x /= total_weight
        tran.translation.y /= total_weight
        tran.translation.z /= total_weight
        tran.rotation.x /= total_weight
        tran.rotation.y /= total_weight
        tran.rotation.z /= total_weight
        tran.rotation.w /= total_weight
        return tran

    else:
        tran = Transform()
        tran.translation.x = 0
        tran.translation.y = 0
        tran.translation.z = 0
        tran.rotation.x = 0
        tran.rotation.y = 0
        tran.rotation.z = 0
        tran.rotation.w = 1
        return tran

def compute_average_time_delta():
    '''Returns the average delta time (in seconds) between consecutive timestamped transforms'''
    global latest_tfcam_list
    min_delta = 0.001  # one millisecond by default
    n = len(latest_tfcam_list)
    if n > 1:
        delta = 0.
        for i in range(1,n):
            delta += latest_tfcam_list[i-1].header.stamp.to_sec() - latest_tfcam_list[i].header.stamp.to_sec()
            #print('    ### {}'.format(latest_tfcam[i-1].header.stamp.to_sec() - latest_tfcam[i].header.stamp.to_sec()))
        delta /= (n-1)
        return max(delta, min_delta)
    else:
        return min_delta

def get_latest_transform():
    global latest_tfcam_list
    global latest_tfcam_sent
    t = Transform()
    t.rotation.w = 1.
    n = len(latest_tfcam_list)
    if n > 0:
        if latest_tfcam_sent is None:
            latest_tfcam_sent = copy.deepcopy(latest_tfcam_list[0].transform)

        # Exponential filtering
        tran_filtered = do_filter()

        # Convert max_speed from meters per second to meters per frame
        #               (same for quaternions)
        delta_time = compute_average_time_delta()
        mt = max_speed_trans * delta_time
        mr = max_speed_rot * delta_time
        #print('**$$ {} {} {}'.format(delta_time, max_speed_trans, mt))

        # PD control on translation
        t1 = tran_filtered.translation
        t2 = latest_tfcam_sent.translation
        t.translation.x = t2.x + alpha * min(max(t1.x - t2.x, -mt), mt)
        t.translation.y = t2.y + alpha * min(max(t1.y - t2.y, -mt), mt)
        t.translation.z = t2.z + alpha * min(max(t1.z - t2.z, -mt), mt)

        # PD control on rotation
        t1 = tran_filtered.rotation
        t2 = latest_tfcam_sent.rotation
        t.rotation.x = t2.x + alpha * min(max(t1.x - t2.x, -mr), mr)
        t.rotation.y = t2.y + alpha * min(max(t1.y - t2.y, -mr), mr)
        t.rotation.z = t2.z + alpha * min(max(t1.z - t2.z, -mr), mr)
        t.rotation.w = t2.w + alpha * min(max(t1.w - t2.w, -mr), mr)

        # normalize rotation
        #print(type(t.rotation))
        norm = np.sqrt(t.rotation.x ** 2. + t.rotation.y ** 2. + t.rotation.z ** 2. + t.rotation.w ** 2.) 
        t.rotation.x /= norm
        t.rotation.y /= norm
        t.rotation.z /= norm
        t.rotation.w /= norm

        # Store for next time
        latest_tfcam_sent = copy.deepcopy(t)

    return t

def get_latest_transform_stamped(base_id,child_id_out):
    t = TransformStamped()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = base_id
    t.child_frame_id  = child_id_out
    t.transform = get_latest_transform()
    return t

def main(args):
    rospy.init_node('temporal_filter_tf_pose_{}'.format(args.tf_child))

    tfBuffer = tf2.Buffer()
    tf_broadcaster = tf2.TransformBroadcaster()
    tf_listener = tf2.TransformListener(tfBuffer)

    rate = rospy.Rate(10.0)

    tf_cam = None

    while not rospy.is_shutdown():

        try:
            # listen and store
            tf_cam = tfBuffer.lookup_transform(
                args.base, 
                args.tf_child,
                rospy.Time())
            on_transform_received(tf_cam)

            # publish filtered result
            t = get_latest_transform_stamped(str(args.base), str(args.tf_child) + str(args.suffix_out))
            print('send time = {}'.format(t.header.stamp.to_sec()))
            tf_broadcaster.sendTransform(t)

        except:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Exponential temporal filtering of TF pose between base and child')
    parser.add_argument('--base', 
                        default = 'base_link',
                        help='base transform')
    parser.add_argument('--tf_child', 
                        default = 'c920',
                        help='child transform to filter')
    parser.add_argument('--suffix_out',
                        default="_filtered",
                        help='suffix appended to the input "tf_child" to create output transform name')
    args = parser.parse_args()
    main(args)
