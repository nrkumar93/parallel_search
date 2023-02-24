# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from __future__ import print_function

from termcolor import colored


logfile = None


def start_log(filename):
    global logfile
    logfile = open(filename, 'w')


def end_log():
    global logfile
    if logfile is not None:
        logfile.close()
        logfile = None


def write_to_log(*args):
    global logfile
    if logfile is not None:
        msg = ''
        if len(args) > 1:
            msg = ""
            for i, term in enumerate(args):
                msg += str(term)
                if i < len(args) - 1:
                    msg += " "
        else:
            msg = args[0]
        logfile.write(msg + "\n")
        logfile.flush()


def log(*args):
    print(*args)
    write_to_log(*args)


def output_colored(color, prelog="", *msg):
    msg2 = ""
    for tok in msg:
        msg2 += str(tok) + " "
    print(colored(msg2, color))
    write_to_log(prelog + msg2)


def say(*msg): output_colored("green", "=====> ", *msg)


def inform(*msg): output_colored("cyan", "-----> ", *msg)


def logwarn(msg):
    print(colored("[WARNING]", 'yellow'), str(msg))
    write_to_log("[WARNING] " + str(msg))


def logerr(msg):
    print(colored("[ERROR]", 'red'), str(msg))
    write_to_log("[ERROR] " + str(msg))
