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

import brain2.utils.transformations as tra
import numpy as np
import tf2_ros
import tf

class TfSensor(object):
    """ This basic sensor just uses TF to determine the location of an object.
    It's useful if you have perfect perception or if you're in a simulator. Any
    situation where you can blindly trust what your sensors are giving you."""

    def _cb(self, cls, msg):
        frame = msg.header.frame_id

    def __init__(self, observer, config, reset_orientation=False, expected_update_t=0.1):
        """ Create it. Don't worry about the details. """
        self.observer = observer
        self._frames = {}
        self._objs = set()
        self._subs = []
        self.expected_update_t = expected_update_t
        self.reset_orientation = reset_orientation
        for name, frame in config.items():
            self._objs.add(name)
            self._frames[name] = frame
    
    def has(self, name):
        """says if this sensor should be tracking the object or not"""
        return name in self._objs

    def update(self, entity, t):
        """gets our best running pose estimate from this sensor"""
        if not entity.name in self._objs:
            return status.IDLE
        else:
            try:
                trans, rot = self.observer.tf_listener.lookupTransform(self.observer.root,
                                                                       self._frames[entity.name],
                                                                       rospy.Time(0))
                if self.reset_orientation:
                    rot = [0, 0, 0, 1]
                # For debugging and implementation aid
                res = entity._update(base_pose=make_pose(trans, rot),
                                    obs_t=t, t=t,
                                    max_obs_age=self.expected_update_t)
                return status.SUCCESS if res else status.FAILED

            except (tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                logwarn(str(e))

        T = np.eye(4)
        T[2,3] = 1000
        entity.set_base_pose(T)
        entity.observed = False
        return status.FAILED
