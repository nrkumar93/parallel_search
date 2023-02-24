# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


#!/usr/bin/env python

import rospy
from std_msgs.msg import Time
from rosgraph_msgs.msg import Clock
import time
import timeit

rospy.init_node('publish_clock')
pub = rospy.Publisher("/clock", Clock, queue_size=1)

while not rospy.is_shutdown():
    t = timeit.default_timer()
    rt = rospy.Time(t)
    pub.publish(clock=rt)
    time.sleep(0.001)


