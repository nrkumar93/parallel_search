# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import argparse
from brain.info import logwarn

def parse_kitchen_args(lula=0, sim=1, lula_opt=0):
    """
    Arguments for the test/isaac sim version of the demo.
    lula: should we use RMPs?
    sim: is this simulation or not?
    """

    logwarn("===================================")
    logwarn("Using simulation = " + str(sim))
    logwarn("Using lula for motions = " + str(sim))
    logwarn("===================================")

    parser = argparse.ArgumentParser("Kitchen Domain Execution")
    parser.add_argument(
        '--script_timeout',
        type=float,
        default=None,
        help='If specified, times out and exits after this amount of time.')
    parser.add_argument(
        "--gui",
        action="store_true",
        help="show backend GUI")
    parser.add_argument(
        '--no_planning',
        action='store_true',
        help='If planner is available, do not run it.')
    parser.add_argument(
        '--real',
        action='store_true',
        help='Run in the real world.')
    parser.add_argument(
        '--debug_planner',
        action='store_true',
        help='Stop after planner, check plans to make sure they are valid.')
    parser.add_argument(
        '--pause',
        action='store_true',
        help='pause at certain times, like after planning.')
    parser.add_argument(
        '--image',
        type=str,
        default='',
        help='If provided, will save images assuming template e.g.'
        'img%02d.png')
    parser.add_argument('--max_count', type=int, default=999999, help='max '
                        'number of steps to execute')
    parser.add_argument('--disrupt', action="store_true", help="interrupt the"
                        "task execution partway through to test robustness")
    parser.add_argument(
        '--linear',
        action="store_true",
        help="linear execution"
        " only")
    parser.add_argument('--replan', action="store_true", help='replan if '
                        'execution fails')
    parser.add_argument('--seed', default=None, help='random seed for trial')
    parser.add_argument(
        '--image_topic',
        default="/sim/left_color_camera/image",
        help='image topic to listen on')
    parser.add_argument('--iter', default=1, type=int, help="Number of test"
                        "iterations to run. If iter <= 0, then run forever.")
    parser.add_argument('--max_t', default=180., type=float,
                        help="amount of time in seconds a trial can take")
    parser.add_argument('--randomize_textures', type=int, default=0,
                        help="if > 0, randomize textures at each trial.")
    parser.add_argument('--randomize_camera', type=int, default=1,
                        help="if > 0, randomize camera at each trial.")
    parser.add_argument('--sigma', type=float, default=0., help='noise added'
                        ' to task execution goals')
    parser.add_argument('--p_sample', type=float, default=0, help='chance to'
                        ' sample object pose according to noise')
    parser.add_argument('--lula_collisions', action='store_true',
                        help='use lula collision avoidance')
    parser.add_argument('--babble', action='store_true',
                        help='Collect babbling data.')
    parser.add_argument('--task', default="pick spam", type=str,
                        help='set task type')
    parser.add_argument('--lula', default=lula, type=int,
                        help="use RMPs as motion backend.")
    parser.add_argument('--lula_opt', default=lula_opt, type=int,
                        help="Use Lula motion optimization where supported.")
    parser.add_argument('--sim', default=sim, type=int,
                        help="create in simulation mode")
    parser.add_argument('--verbose', default=0, type=int,
                        help="Task model verbosity level")

    args = parser.parse_args()
    if args.seed is not None:
        args.seed = int(args.seed)
    return args
