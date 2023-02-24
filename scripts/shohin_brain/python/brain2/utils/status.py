# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""
Define common status codes used across system execution
"""

# For actions and policies
FINISHED = 2
SUCCESS = 2
IDLE = 1
RUNNING = 0
FAILED = -1
FAILURE = -1

# For attachment and detaching things
PLACE = 2
ATTACH = 1
NO_CHANGE = 0
DETACH = -1

# For states
READY = 0
ENTERED = 1
EXITED = -1
