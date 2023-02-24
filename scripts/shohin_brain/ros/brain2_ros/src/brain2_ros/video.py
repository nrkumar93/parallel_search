# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import rospy
import threading

import cv2
import numpy as np
from cv2 import VideoWriter, VideoWriter_fourcc

from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image

class VideoCapture(object):
    """
    Capture incoming images from a webcam and label them in interesting ways.
    """


    def __init__(self, topic="/usb_cam/image_raw", filename="output.mp4", fps=30):
        self.lock = threading.Lock()
        self.bridge = CvBridge()
        self.fps = fps
        self.filename = filename

        # For writing
        self.msg = None
        self.msg_pos = None
        self.msg_color = None
        self.rectangle = None
        self.rectangle_color = None
        self.font = cv2.FONT_HERSHEY_SIMPLEX 

        self.video = None 
        self._sub = rospy.Subscriber(topic, Image, self._cb, queue_size=10)
        self._pub = rospy.Publisher("annotated_image", Image, queue_size=1)

    def annotate(self, msg, msg_pos=None, msg_color=(255,255,255), rectangle=None, rectangle_color=(255,0,0)):
        with self.lock:
            self.msg = msg
            self.msg_pos = msg_pos
            self.msg_color = msg_color
            self.rectangle = rectangle
            self.rectangle_color = rectangle_color

    def apply_annotation(self, image):
        """
        add text and writing
        """
        if self.msg is not None:
            if self.rectangle is not None:
                x1, y1 = self.rectangle[0], self.rectangle[1]
                x2, y2 = self.rectangle[2], self.rectangle[3]
            else:
                x1 = 0
                y1 = int(self.height * 0.9)
                x2 = self.width
                y2 = self.height
            r, g, b = self.rectangle_color
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (b, g, r), -1)

            if len(self.msg) > 0:
                # write text
                r, g, b = self.msg_color
                if self.msg_pos is not None:
                    x, y = self.msg_pos
                    scale = 1
                else:
                    x = 20
                    y = int(self.height * 0.95) + 10
                    scale = 1
                image = cv2.putText(image, self.msg, (x, y), self.font, scale,
                        color=(b, g, r), thickness=2, lineType=cv2.LINE_AA)

            return image

    def _cb(self, msg):
        with self.lock:
            if self.msg is None:  return
            try:
                image_raw = self.bridge.imgmsg_to_cv2(msg, "passthrough")
                image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
                if self.video is None:
                    # Initialize to the correct size
                    height = image.shape[0]
                    width = image.shape[1]
                    self.height = height
                    self.width = width
                    #fourcc = VideoWriter_fourcc(*'FMP4')  # This one gives a warning
                    fourcc = VideoWriter_fourcc(*'XVID')
                    #fourcc = VideoWriter_fourcc(*'ffv1')  # This one did not work
                    #fourcc = VideoWriter_fourcc(*'mp4v')
                    #fourcc = VideoWriter_fourcc(*'x264')
                    self.video = VideoWriter(self.filename, fourcc, float(self.fps), (width, height))
                image = self.apply_annotation(image)
                self.video.write(image)
                msg2 = self.bridge.cv2_to_imgmsg(image, "passthrough")
                self._pub.publish(msg2)
            except CvBridgeError, e:
                rospy.logerr("CvBridge Error: {0}".format(e))

    def close(self):
        self.video.release()

    def __del__(self):
        self.close()
