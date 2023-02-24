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

from brain_msgs.msg import HandStatus

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
        self.last_msg = msg
        self.category = msg.name
        self.confidence = msg.confidence
        self.hand_stationary_counter = msg.hand_stationary_counter
        self.graspable = msg.graspable
        self.obj_in_hand = msg.object_in_hand
        self.hand_pose = make_pose_from_unstamped_pose_msg(msg.pose)
        self.frame = msg.header.frame_id

        # Correctly parse the message contents based on whether or not we
        # classified this as the kind of grasp that contains an object.
        self.obj_observed = False
        if self.category in self.use_has_obj_labels:
            if msg.object_in_hand != "NA":
                self.has_obj = True
                self.obj_observed = True
                self.obj_pose = make_pose_from_unstamped_pose_msg(msg.object_pose)
            else:
                self.has_obj = False
                self.obj_pose = None
        elif self.category in self.has_obj_labels:
            # print(" >>>>>>>>> ", self.category)
            self.has_obj = True
            # Use object pose if it's available only
            if msg.object_in_hand != "NA":
                self.obj_observed = True
                self.obj_pose = make_pose_from_unstamped_pose_msg(msg.object_pose)
            else:
                self.obj_pose = None
        else: # No object labels
            # Update here.
            # print("unrecognized cat =", self.category)
            self.has_obj = False
            self.obj_pose = None

        self.obs_t = msg.header.stamp.to_sec()

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
        if self.has_obj and self.obj_pose is not None:
            pos = self.obj_pose[:3, 3]
            rot = tra.quaternion_from_matrix(self.obj_pose)
            self._tf_broadcaster.sendTransform(hpos, hrot, 
                                               rospy.Time.now(),
                                               "obs_obj_" + self._hand + "_obj",
                                               self.frame)

    def print_info(self):
        print("====", self._hand, "====")
        print("Category:", self.category)
        print("Confidence:", self.confidence)
        print("Observed:", self.obs_t)
        print("=====" + ("=" * len(self._hand)) + "=====")

    def __init__(self, observer, hand="right", topic="right_hand_status"):
        """
        observer: the observer object managing world state
        rostopic: the topic messages will be arriving on
        """
        self.observer = observer

        # For debugging purposes
        self._tf_broadcaster = observer.tf_broadcaster

        # Track supported hands
        self._hand = hand
        self._hand_topic = topic

        # Initialize parameters
        self._parse_msg(HandStatus(object_in_hand="NA"))
        self._hand_msg_sub = rospy.Subscriber(self._hand_topic,
                                              HandStatus,
                                              self._cb,
                                              queue_size=1)

    def has(self, obj):
        """ Each sensor only tracks a single hand """
        return obj == self._hand

    def update(self, entity, t):
        """ update the object with important information about what it is """
        obj_pose = np.copy(self.obj_pose) if self.has_obj else None
        if t - self.obs_t < -1.0:
            raise RuntimeError('You have a timing issue! cannot observe into'
                    ' the past: ' + str(t) + ", " + str(self.obs_t)
                    + "diff =" + str(t - self.obs_t))
        res = entity._update(base_pose=np.copy(self.hand_pose),
                             obs_t=self.obs_t,
                             obj_pose=obj_pose,
                             has_obj=self.has_obj,
                             t=t,
                             max_obs_age=0.5,
                             hand_shape=self.category)
        return status.SUCCESS if res else status.FAILED
