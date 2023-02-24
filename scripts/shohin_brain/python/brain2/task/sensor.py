# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from __future__ import print_function


class Sensor(object):
    """
    Base class, updates information. Sort of a placeholder right now.
    """

    def __init__(self, topic_name, weight=0):
        self.topic_name = topic_name
        self._weight = weight
        self.enable()

    def enable(self):
        self.weight = self._weight

    def disable(self):
       self.weight = 0

    def ready(self):
        """
        Override this for messages
        """
        return False
