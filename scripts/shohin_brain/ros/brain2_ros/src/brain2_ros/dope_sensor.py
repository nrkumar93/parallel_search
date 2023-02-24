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
from utils import make_pose_from_pose_msg

import brain2.utils.transformations as tra
import numpy as np
import tf2_ros
import tf

class DopeSensor(object):
    """collects dope information"""

    def _cb(self, cls, msg):
        frame = msg.header.frame_id
        try:
            t, r = self.observer.tf_listener.lookupTransform(self.observer.root,
                                                             frame,
                                                             rospy.Time(0))
            base_pose = make_pose(t, r)
            self._poses[cls] = (base_pose.dot(
                        make_pose_from_pose_msg(msg)),
                        msg.header.stamp.to_sec())

            # For debugging and implementation aid
            t = self._poses[cls][0][:3, 3]
            r = tra.quaternion_from_matrix(self._poses[cls][0])
            self._tf_broadcaster.sendTransform(t, r,
                                               rospy.Time.now(),
                                               "obs_" + cls,
                                               self.observer.root)
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            logwarn(str(e))
            return

    def __init__(self, observer, config):
        self.observer = observer
        self._tf_broadcaster = tf.TransformBroadcaster()
        self._poses = {}
        self._objs = set()
        self._subs = []
        self.expected_update_t = 3.
        for name, cls in config.items():
            # Listen to the differnet incoming pose topics
            def _get_cb(name):
                return lambda msg: self._cb(name, msg)
            self._objs.add(name)
            self._subs.append(rospy.Subscriber("/dope/pose_" + cls,
                                               PoseStamped,
                                               _get_cb(name),
                                               queue_size=1))
    
    def has(self, name):
        """says if this sensor should be tracking the object or not"""
        return name in self._objs

    def update(self, entity, t):
        """gets our best running pose estimate from this sensor"""
        if entity.name in self._poses:
            pose, obs_t = self._poses[entity.name]
            # print("in update", entity.name, obs_t, t, rospy.Time.now().to_sec())
            res = entity._update(base_pose=np.copy(pose),
                                obs_t=obs_t, t=t,
                                max_obs_age=self.expected_update_t)
            return status.SUCCESS if res else status.FAILED
        elif not entity.name in self._objs:
            return status.IDLE
        else:
            # Set object far away so it doesn't interfere with anything
            T = np.eye(4)
            T[2,3] = 1000
            entity.set_base_pose(T)
            entity.observed = False
            return status.FAILED
