# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

import rospy
from sensor_msgs.msg import JointState

from brain2.utils.pose import make_pose
from brain2.utils.info import logwarn, logerr
import brain2.utils.status as status
from utils import make_pose_from_pose_msg

import brain2.utils.transformations as tra
import numpy as np
import tf2_ros
import tf

from collections import namedtuple
ArticulatedObjInfo = namedtuple('ArticulatedObjectInfo',
                                ['name', 'tf_frame', 'joint_name'])


class ArticulatedObjSensor(object):
    """ This basic sensor just uses TF to determine the location of an object.
    It's useful if you have perfect perception or if you're in a simulator. Any
    situation where you can blindly trust what your sensors are giving you."""

    def __init__(self, observer, tracker_topic, config, expected_update_t=0.1):
        """ Create it. Don't worry about the details. """
        self.observer = observer
        self._frames = {}         # stores the TF frames for positions
        self._joint_to_name = {}  # stores the names of each joint
        self._obs = {}            # stores the latest observations of each joint
        self._objs = set()
        self._subs = []
        self.expected_update_t = expected_update_t
        for name, info in config.items():
            self._objs.add(name)
            self._frames[name] = info.tf_frame
            self._joint_to_name[info.joint_name] = name

        self._sub = rospy.Subscriber(tracker_topic, JointState, self._cb, queue_size=100)

    def _cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            if name in self._joint_to_name:
                self._obs[self._joint_to_name[name]] = pos
    
    def has(self, name):
        """says if this sensor should be tracking the object or not"""
        return name in self._objs

    def update(self, entity, t):
        """gets our best running pose estimate from this sensor"""
        if not entity.name in self._objs:
            return status.IDLE
        if not entity.name in self._obs:
            return status.FAILED
        else:
            try:
                trans, rot = self.observer.tf_listener.lookupTransform(self.observer.root,
                                                                       self._frames[entity.name],
                                                                       rospy.Time(0))

                # For debugging and implementation aid
                res = entity._update(pose=make_pose(trans, rot),
                                    obs_t=t, t=t,
                                    q=self._obs[entity.name],
                                    max_obs_age=self.expected_update_t)
                return status.SUCCESS if res else status.FAILED

            except (tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                logwarn(str(e))

        T = np.eye(4)
        T[2,3] = 1000
        entity.pose = T
        entity.observed = False
        return status.FAILED
