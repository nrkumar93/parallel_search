# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

import numpy as np

def choose2(num, test=False):
    if test:
        a = num - 1
        b = np.random.randint(a)
    else:
        a = np.random.randint(num-1)
        b = (a + np.random.randint(1, num)) % num
    assert a != b
    return a, b

def choose3(num, test=False):
    if test:
        a = num - 1
    else:
        a = np.random.randint(num-1)
    opts = list(range(num))
    opts.remove(a)
    np.random.shuffle(opts)
    return a, opts[0], opts[1]

def choose4(num, test=False):
    if test:
        a = num - 1
    else:
        a = np.random.randint(num - 1)
    opts = list(range(num))
    opts.remove(a)
    np.random.shuffle(opts)
    return a, opts[0], opts[1], opts[2]

