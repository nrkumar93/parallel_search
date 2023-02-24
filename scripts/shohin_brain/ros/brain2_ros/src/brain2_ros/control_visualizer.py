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
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import tf.transformations as tft


class ControlVisualizer(object):
    """
    This class sends a ROS visualization message.
    TODO: Wei: this is where the red arrow comes from
    """

    def __init__(self, shape=Marker.ARROW, radius=0.04, base_frame="base_link",
                 topic="action_markers"):
        self.shape = shape
        self.radius = radius
        self.base_frame = base_frame
        self.pub = rospy.Publisher(topic, MarkerArray, queue_size=1)

    def make_marker(self, T, ns, id, shape, radius):
        pos = T[:3, 3]
        rot = tft.quaternion_from_matrix(T)
        msg = Marker()
        msg.header.frame_id = self.base_frame
        msg.header.stamp = rospy.Time.now()
        msg.pose.position.x = pos[0]
        msg.pose.position.y = pos[1]
        msg.pose.position.z = pos[2]
        msg.pose.orientation.x = rot[0]
        msg.pose.orientation.y = rot[1]
        msg.pose.orientation.z = rot[2]
        msg.pose.orientation.w = rot[3]
        msg.type = shape
        if shape == Marker.ARROW:
            msg.scale.x = 3 * radius
            msg.scale.y = 0.5 * radius
            msg.scale.z = 0.5 * radius
        elif shape == Marker.MESH_RESOURCE:
            msg.scale.x = 1.
            msg.scale.y = 1.
            msg.scale.z = 1.
            msg.mesh_resource = 'package://brain2_ros/assets/panda_gripper.obj'
            msg.mesh_use_embedded_materials = True
        else:
            msg.scale.x = radius
            msg.scale.y = radius
            msg.scale.z = radius

        msg.action = Marker.ADD
        msg.id = id
        msg.ns = ns

        return msg

    def send(self, T, ns="action", id=0):
        if self.shape == Marker.ARROW:
            T = np.copy(T)
            T = np.dot(T, tft.euler_matrix(0, -1. * np.pi / 2, 0))
            T0 = np.eye(4)
            T0[:3, 3] = [-2 * self.radius, 0, 0]
            T = np.dot(T, T0)
        elif self.shape == Marker.MESH_RESOURCE:
            T0 = tft.quaternion_matrix([0, 0, -0.707, 0.707])
            T0[:3, 3] = [0, 0, -0.1]
            T = np.dot(T, T0)

        m = self.make_marker(T, ns, id, self.shape, self.radius)
        m.color.r = 1
        m.color.b = 0
        m.color.g = 0
        m.color.a = 1.

        msg = MarkerArray(markers=[m])
        self.pub.publish(msg)

    def send_obj(self, T, ns="action", id=1, radius=0.05):

        m = self.make_marker(T, ns, id, Marker.CUBE, radius)
        m.color.r = 0
        m.color.b = 1.
        m.color.g = 1.
        m.color.a = 0.75

        msg = MarkerArray(markers=[m])
        self.pub.publish(msg)

    def make_delete(self, ns, id):
        msg = Marker()
        msg.action = Marker.DELETE
        msg.id = id
        msg.ns = ns
        return msg

    def stop(self, ns='action'):
        m = self.make_delete(ns, 0)
        m2 = self.make_delete(ns, 1)
        msg = MarkerArray(markers=[m, m2])
        self.pub.publish(msg)


if __name__ == '__main__':
    rospy.init_node('control_visualizer_test')
    c = ControlVisualizer()
    while not rospy.is_shutdown():
        c.send(np.eye(4))
        rospy.sleep(5.)
        c.stop()
        rospy.sleep(1.)
