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

from posecnn_pytorch.msg import DetectionList

class MultiPoseSensor(object):
    """Listens to messages from the azure + realsence for picking up
    objects and localizing them in space."""

    def __init__(self, observer, obj_config):
        """
        :param observer - links to the data collector
        :param obj_config - contains information on the different things that
               we need to listen to.
        """
        self.observer = observer
        self.tf_listener = observer.tf_listener
        self._tf_broadcaster = observer.tf_broadcaster
        self._config = obj_config

    def has(self, name):
        return name in self._config

    def update(self, entity, t):
        entity.observed = False
        entity.stable = False
        return status.SUCCESS
