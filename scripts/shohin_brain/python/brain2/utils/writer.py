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
from brain2.utils.info import *
import brain2.utils.image as img

class H5fWriter(object):
    """ creates an h5f data blob containing some information """
    
    def __init__(self, directory='.'):
        self.directory = directory 

        try:
            os.mkdir(directory)
        except FileExistsError as e:
            pass
        self.data = {}
        self.current = None

    def start(self, i):
        """ This starts the file and actually does the creation """
        name = os.path.join(self.directory, "data%08d.h5" % i)
        self.current = h5py.File(name, 'w')
        self.data = {}

    def add_data(self, **data):
        """ this writes repeated data """
        if self.current is None:
            raise RuntimeError('no file')
        for k, v in data.items():
            if k not in self.data:
                self.data[k] = []
            self.data[k].append(v)

    def add_info(self, **data):
        """ This writes one-time "header" information """
        if self.current is None:
            raise RuntimeError('no file')
        for k, v in data.items():
            if k not in self.data:
                self.data[k] = v
            else:
                raise RuntimeError('cannot add info for ' + str(k)
                                   + ' as it already exists')

    def add_png(self, **pngs):
        """ This helper should write PNGs to save them as well """
        if self.current is None:
            raise RuntimeError('no file')
        for k, v in pngs.items():
            if k not in self.data:
                self.data[k] = []
            self.data[k].append(img.GetPNG(v))
            # print(k, 'len =', len(self.data[k]))

    def finish(self):
        """ Convert everything and dump to h5 file """
        for k in self.data.keys():
            if self.data[k] is None:
                raise RuntimeError('data was not properly populated')
            self.current.create_dataset(k, data=self.data[k])
        self.current.close()
        self.current = None

