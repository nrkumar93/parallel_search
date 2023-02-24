# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

import h5py
import os

from brain2.utils.info import logwarn
from brain2.utils.info import logerr
from brain2.learning.image import *


class Writer(object):
    """ Writes out H5 trials for different trials, or things like that. """

    def __init__(self, directory):
        self.directory = directory
        try:
            os.mkdir(directory)
        except OSError as e:
            logwarn(str(e))

        self.reset_current_file()
        self.counter = 0

    def reset_current_file(self):
        self.current_file = None
        self.current_data = {}

    def start_new_file(self, name):
        """ name should contain some int """
        self.reset_current_file()
        
