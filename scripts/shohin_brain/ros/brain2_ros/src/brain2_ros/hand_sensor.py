# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

import rospy
from geometry_msgs.msg import PoseStamped

from brain2.utils.pose import make_pose
from brain2.utils.info import logwarn, logerr
import brain2.utils.status as status
from utils import make_pose_from_unstamped_pose_msg

import brain2.utils.transformations as tra
import numpy as np
import tf2_ros
import tf
import threading

#from brain_msgs.msg import HandStatus
from visualization_msgs.msg import Marker


class HandSensor(object):
    """
    Listens to hand status messages
    """

    all_labels = set(['flat (no obj)', 'other (no obj)', 'flat', 'pinch_lift',
              'pinch_top', 'pinch_bottom', 'pinch_side'])
    has_obj_labels = set(['flat', 'pinch_lift', 'pinch_top', 'pinch_bottom',
                          'pinch_side'])
    no_obj_labels = set(['flat (no obj)', 'other (no obj)'])
    use_has_obj_labels = set(['skeleton'])

    def _cb(self, msg):
        """
        Store information
        """
        self._msg = msg
        self._parse_msg(msg)

    def _parse_msg(self, msg):
        """ Parse in the message """
        with self.lock:
            self.last_msg = msg
            self.hand_pose = make_pose_from_unstamped_pose_msg(msg.pose)
            self.frame = msg.header.frame_id
            self.obs_t = msg.header.stamp.to_sec()
            self.pose = None

            if len(self.frame) > 0:
                try:
                    pos, rot = self._tf_listener.lookupTransform(
                            self._root,
                            self.frame,
                            rospy.Time(0))
                    self.pose = make_pose(pos, rot).dot(self.hand_pose)
                except Exception as e:
                    logwarn((str(e)))

            if self.obs_t > 1.:
                # Send a message for debugging purposes
                # We'll visualize in RVIZ via TF for now
                self._send_tf()

    def _send_tf(self):
        """ Extract pos + quat information so that we can send all this to
        rviz and visualize it for debugging purposes. """
        hpos = self.hand_pose[:3, 3]
        hrot = tra.quaternion_from_matrix(self.hand_pose)
        self._tf_broadcaster.sendTransform(hpos, hrot, 
                                           rospy.Time.now(),
                                           "obs_hand_" + self._hand,
                                           self.frame)

    def print_info(self):
        print("====", self._hand, "====")
        print("Position [orig]:", self.hand_pose[:3, 3])
        print("Frame:", self.frame)
        if self.pose is not None:
            print("Position [world]:", self.pose[:3, 3])
        print("Observed:", self.obs_t)
        print("=====" + ("=" * len(self._hand)) + "=====")

    def __init__(self, observer, hand="right", topic="right_hand_status",
                 max_obs_age=0.5):
        """
        observer: the observer object managing world state
        rostopic: the topic messages will be arriving on
        """
        self.observer = observer
        self.max_obs_age = max_obs_age
        self.lock = threading.Lock()

        # For debugging purposes
        self._tf_broadcaster = observer.tf_broadcaster
        self._tf_listener = observer.tf_listener

        # Track supported hands
        self._root = observer.root
        self._hand = hand
        self._hand_topic = topic

        # Initialize parameters
        self._parse_msg(Marker())
        self._hand_msg_sub = rospy.Subscriber(self._hand_topic, Marker, self._cb, queue_size=1)

    def has(self, obj):
        """ Each sensor only tracks a single hand """
        return obj == self._hand

    def update(self, entity, t):
        """ update the object with important information about what it is """
        if t - self.obs_t < -1.0:
            raise RuntimeError('You have a timing issue! cannot observe into'
                    ' the past: ' + str(t) + ", " + str(self.obs_t)
                    + "diff =" + str(t - self.obs_t))
        with self.lock:
            if self.pose is not None:
                pose = np.copy(self.pose)
            else:
                pose = None
        res = entity._update(base_pose=pose, obs_t=self.obs_t, t=t,
                             max_obs_age=self.max_obs_age)
        return status.SUCCESS if res else status.FAILED
