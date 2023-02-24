# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

import argparse
import os


def addPlanningArgs(parser):
    """ Add some basic planning argumnets """
    parser.add_argument("--seed", type=int, default=0, help="numpy seed")
    parser.add_argument("--pause", type=int, default=0, help="pause after planning (0/1)")
    parser.add_argument("--visualize", type=int, default=0, help="show pybullet"
        " gui. Should we also visualize the environment if supported? Note "
        " that this will also slow down planning. (0/1)")
    parser.add_argument("--max_planning_time", type=float, default=30., help="maximum planning time")
    parser.add_argument("--greedy_planning", action="store_true", help="terminate when any solution is found")
    return parser


def addRobotArgs(parser):
    """ Add options for inverse kinematics and the like """
    if "HOME" in os.environ:
        default_path = os.path.join(os.environ["HOME"], 'src/brain_gym/assets/urdf')
    else:
        default_path = '../../../assets/urdf',
    parser.add_argument('--assets_path',
        default=default_path,
        help='assets path')
    #parser.add_argument('--visualize', type=int, default=0,
    #        help='should we also visualize the environment if supported? Note'
    #              ' that this will also slow down planning.')
    parser.add_argument("--parallel_ik", action="store_true",
                        help="WARNING still work in progress")
    parser.add_argument("--num_ik_threads", type=int, default=10)
    return parser

def createScriptParser(title):
    parser = argparse.ArgumentParser(title)
    parser = addPlanningArgs(parser)
    parser.add_argument("--iter", type=int, default=1, help="number of examples")
    parser.add_argument("--dir", type=str, default=".", help="place to put examples")
    parser.add_argument("--images", type=int, default=1, help="should we save images (0/1)")
    parser.add_argument("--tamp", type=int, default=1, help="run task and motion planner (0/1)")
    parser.add_argument("--hard", type=int, default=1, help="generate slightly harder scenes (0+)")
    return parser


def createDataScriptParser(title):
    parser = createScriptParser(title)
    parser.add_argument("--outdir", type=str, default='/data/brain2',
                        help="location of output data files")
    if "HOME" in os.environ:
        default_path = os.path.join(os.environ["HOME"], 'src/brain_gym/assets/urdf')
    else:
        default_path = './assets/urdf',
    parser.add_argument('--assets_path',
        default=default_path,
        help='assets path')
    return parser

def createLanguageScriptParser(title, problems=None):
    parser= createDataScriptParser(title)
    parser.add_argument("--num_goal_expressions", type=int, default=5, help="num goal expressions "
                        "to generate")
    parser.add_argument("--num_plan_expressions", type=int, default=5, help="num plan expressions "
                        "to generate")
    parser.add_argument("--save_language", type=int, default=0, help="save language (0/1)")
    parser.add_argument("--lang_file", type=str, default='language_data.json', help="file to save language data to")
    if problems is not None:
        parser.add_argument("--problem", type=str, help="problem to try",
                        choices=problems)
    return parser

def createPredicateScriptParser(title):
    """ Parameters for creating the predicate training data """
    parser = createDataScriptParser(title)
    parser.add_argument("--num_views", type=int, default=10,
                        help="views to generate per scene")
    parser.add_argument("--num_scenes", type=int, default=10,
                        help="random scenes to generate at a time")
    parser.add_argument("--num_objs", type=int, default=3,
                        help="num objects per scene")
    parser.add_argument("--shapenet_obj_percentage", type=float, default=0.8)
    parser.add_argument("--cuboid_obj_percentage", type=float, default=0.1)
    parser.add_argument("--nv_grasp_dataset", default="/data/NVGraspDataset_clemens_complete")
    return parser

