# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

import numpy as np
import rospy
from brain2.utils.pose import make_pose
import brain2.utils.transformations as tra
import tf

def _make_box_poses(x, y, z, directions=[], step=0.02):
    """ Simple grasp generator for preferred grasp positions """
    poses = []
    for d in directions:
        # rotate frame
        # apply offset
        theta = 0 if d[0] == "+" else np.pi
        pm = 1 if d[0] == "+" else -1
        if d[1] == "z":
            # rotate around the x axis
            T = tra.euler_matrix(theta, 0, 0)
            T[2,3] = (pm * z/2.) - (pm * step)
            poses.append(T)
        elif d[1] == "y":
            # z along x
            T = tra.euler_matrix(np.pi/2, 0, 0)
            # rotate around the x axis
            T2 = tra.euler_matrix(theta, 0, 0)
            #T = T.dot(T2)
            T2[2,3] = (pm * y/2.) - (pm * step)
            T = T.dot(T2)
            poses.append(T)
        elif d[1] == "x":
            # z along x
            T = tra.euler_matrix(0, np.pi/2, 0)
            # rotate around the x axis
            T2 = tra.euler_matrix(theta, 0, 0)
            #T = T.dot(T2)
            T2[2,3] = (pm * x/2.) - (pm * step)
            T = T.dot(T2)
            poses.append(T)
        else:
            raise RuntimeError("not recongized: "  + str(d))
    return poses

def get_household_poses(Ts=None):
    if Ts is None:
        Ts = [np.eye(4)]
    household_poses = {
            "Raisins": _make_box_poses(0.12, 0.035, 0.0825, ["+z", "-z", "+x", "-x"]),
            "MacaroniAndCheese": _make_box_poses(0.165, 0.035, 0.12, ["+z", "-z", "+x", "-x"]),
            "BBQSauce": _make_box_poses(0.14, 0.035, 0.06, ["+z", "-z", "+x"]),
            "Milk": _make_box_poses(0.19, 0.07, 0.07, ["+x", "+z", "-z", "+y", "-y"]),
            }
    for k, v in household_poses.items():
        poses = []
        for Tobj in v:
            for T in Ts:
                poses.append(Tobj.dot(T))
        household_poses[k] = poses
    return household_poses

def get_cube_poses(Ts=None, size="median"):
    if Ts is None:
        Ts = [np.eye(4)]

    return {}

def get_grasp_correction_franka():
    t0 = tra.euler_matrix(np.pi*-1, 0, np.pi)
    t1 = tra.euler_matrix(np.pi*-1, 0, np.pi).dot(tra.euler_matrix(0, 0, np.pi))
    return [t0, t1]

if __name__ == "__main__":
    rospy.init_node('visualize_object_grasps')
    obj_config = {
            'bbq_sauce': 'BBQSauce',
            'mac_and_cheese': 'MacaroniAndCheese',
            'raisins': 'Raisins',
            'milk': 'Milk',
            }
    # Flip 180 degrees for a 2 finger gripper
    t0, t1 = get_grasp_correction_franka()
    #t1 = tra.euler_matrix(np.pi*-1, 0, 0)
    #t1 = tra.euler_matrix(np.pi*-1, 0, np.pi) * tra.euler_matrix
    tf_broadcaster = tf.TransformBroadcaster()
    poses = get_household_poses([t0, t1])
    #poses = get_household_poses([t0, t1])
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        for k, v in obj_config.items():
            # lookup by class
            Ts = poses[v]
            for i, T in enumerate(Ts):
                pos = T[:3, 3]
                rot = tra.quaternion_from_matrix(T)
                tf_broadcaster.sendTransform(pos, rot, rospy.Time.now(), 
                                             "grasp_" + k + "_%02d" % i,
                                             "obs_" + k)
        rate.sleep()
