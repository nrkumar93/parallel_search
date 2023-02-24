# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


#!/usr/bin/env python

from __future__ import print_function

import math
import copy
import numpy as np
import rospy
import tf2_ros
import tf.transformations as tra
import timeit
import std_msgs

import brain2.utils.axis as axis

from isaac_bridge.manager import SimulationManager
from isaac_bridge.manager import ros_camera_pose_correction

class Simulation(object):
    """ Connects to the simulation so that we can see what happens. """

    def _define_objects(self):
        self.all_objs = [
            "potted_meat_can",
            "tomato_soup_can",
            "sugar_box",
            "cracker_box",
            "mustard_bottle",
            #"bowl",
        ]
        self.var_theta = {
            "potted_meat_can": 0.1,
            "tomato_soup_can": 0.01,
            "sugar_box": 0.1,
            "cracker_box": 0.5,
            "mustard_bottle": 0.5,
            "jello": 0.,
            "bowl": 0., 
            "mug": 0.5, 
        }
        self.drawers = ["indigo_drawer_bottom", "indigo_drawer_top"]
        self.obj_offset = {
            "potted_meat_can": 0.06,
            "tomato_soup_can": 0.15,
            "sugar_box": 0.15,
            "cracker_box": 0.15,
            "mustard_bottle": 0.15,
            "jello": 0.15,
            "bowl": 0.1, 
            "mug": 0.1, 
        }
        self.region_x_bounds = {
            "indigo_front": [4.3, 4.6],
            "indigo": [4.25, 4.75],
            "dismiss_indigo": [3.7, 4.2],
            "mid_table": [1.25, 1.6],
            "mid_table_far": [1.2, 1.8],
        }
        self.region_y_bounds = {
            "indigo_front": [-8.2, -8.05],
            "indigo": [-8.1, -7.85],
            "dismiss_indigo": [-10.5, -10.],
            "mid_table": [-11.2, -10.8],
            "mid_table_far": [-11.3, -10.7],
        }

    def check(self, poses, distance):
        """
        Determine if configuration of objects in the scene is ok
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

    def set_poses(self, objs, region="indigo_front", poses=None, rpy=False, append=True):
        if poses is None:
            poses = {}
        # Important poses:
        # potted_meat_can [ 4.3888669  -8.16653061  0.96620965]
        # mustard_bottle [ 4.63171959 -7.90024185  0.99995404]
        # tomato_soup_can [ 4.52894211 -8.16991043  0.9758876 ]
        # sugar_box [ 4.69812965 -8.16895962  1.00817561]
        # cracker_box [ 4.39447641 -7.89827394  1.02575207]
        print("Setting poses for objs:", objs)

        for obj in objs:
            #default_z = 0.9 # 0.85
            default_z = 1.1
            ok = False
            # TODO: remove this later
            cur_pose = np.eye(4)
            if "jello" in obj:
                cur_pose = tra.quaternion_matrix([0, 0.707,  np.random.random_sample()/2, 0.707])
                default_z = 0.71
            elif "bowl" in  obj:
                cur_pose = tra.quaternion_matrix([0, 0, 0, 1])
                default_z = 0.9
            x_bounds = self.region_x_bounds[region]
            y_bounds = self.region_y_bounds[region]

            count = 0
            while not ok:

                # If we're struggling...
                if count > 100:
                    return None

                # if available, we will place it at the front of the counter. Otherwise
                # the object will be placed in the back.
                vtheta = self.var_theta[obj]
                theta = np.random.rand() * 2.0 * np.pi * vtheta
                pose = np.copy(cur_pose).dot(tra.euler_matrix(0, 0, theta))
                if rpy:
                    r = np.random.randint(4) * np.pi / 2
                    p = np.random.randint(4) * np.pi / 2
                    y = np.random.randint(4) * np.pi / 2
                    pose2 = tra.euler_matrix(r, p, y)
                    pose = pose.dot(pose2)
                x = np.random.uniform(x_bounds[0], x_bounds[1])
                y = np.random.uniform(y_bounds[0], y_bounds[1])
                z = default_z
                pose[:3, axis.POS] = np.array([x, y, z])
                poses[obj] = pose
                # print("trying to put", obj, "at (%f, %f, %f)" % (x, y,
                #       theta))
                ok = self.check(poses, 0.15)
                count += 1

        self.sim.pause()
        for k, pose in poses.items():
            # TODO: make this work better
            if append:
                sim_name = k + "_1"
            else:
                sim_name = k.replace("00", "1")
            print("-->", k, pose[:3, axis.POS], "sim name =", sim_name)
            self.sim.set_pose(sim_name, pose, do_correction=False)
        self.sim.pause()

        return poses

    def set_joints_for_indigo(self, qs):
        self.sim.set_joints(
            ["indigo_drawer_top_joint", "indigo_drawer_bottom_joint"],
            [qs[0], qs[1]])

    def random_scene(self, objs, in_drawer=False):
        """ Create a random test scene with objects """
        available_objs = [o for o in self.all_objs if o not in objs]
        ok = False
        while not ok:
            print("Setting poses for:", available_objs)
            poses = self.set_poses(available_objs, region="dismiss_indigo")
            if poses is None:
                continue
            print("Setting poses for:", objs)
            poses = self.set_poses(
                objs, region="indigo_front", poses=poses)
            if poses is None:
                continue
            ok = True

        drawer = np.random.choice(self.drawers)
        print("Randomizing position for", drawer)
        if "top" in drawer:
            new_qs = np.array([1, 0])
        else:
            new_qs = np.array([0, 1])
        # Determine if a drawer should be fully open or not
        starts_open = np.random.randint(4)
        if starts_open == 0:
            self.set_joints_for_indigo(
                new_qs * 0.25 * np.random.random())
        else:
            # keep lower drawer closed
            new_qs = (new_qs * 0.05) + (0.2 * np.random.random())
            new_qs[1] = 0
            self.set_joints_for_indigo(new_qs)
        r = np.random.randint(2)
        if in_drawer:
            k = np.random.choice(objs)
            new_qs = np.array([1, 0])
            new_qs = (new_qs * 0.20) + (0.05 * np.random.random())
            new_qs[1] = 0
            if new_qs[0] > 0.15:
                self.set_joints_for_indigo(new_qs)
                #pos = [4.756, -8.324, 0.796] + \
                pos = [4.592, -8.312, 0.84] + \
                    np.array([np.random.randn(1)[0] * 0.02,
                              np.random.randn(1)[0] * 0.02, 0])
                rot = [0.394, 0.569, 0.571, 0.441] + \
                    np.random.randn(4) * 0.1
                # random position for the other object
                T = ros.make_pose((pos, rot))
                k = "potted_meat_can"
                self.sim.set_pose("%s_1" % k, T, do_correction=False)
                        
    def __init__(self, lula=False):
        self._define_objects()
        self.sim = SimulationManager(lula=lula)


if __name__ == '__main__':
    rospy.init_node('test_sim_node')
    sim = Simulation()
    sim.random_scene(["potted_meat_can", "tomato_soup_can"])
    raw_input('Press enter')
    sim.random_scene(["cracker_box"])
    raw_input('Press enter')
    sim.random_scene(["potted_meat_can", "tomato_soup_can", "mustard_bottle"])
