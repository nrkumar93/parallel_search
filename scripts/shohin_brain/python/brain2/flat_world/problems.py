# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import numpy as np

from brain2.flat_world.planar_arm import PlanarArmWorld, PlanarArm

def setup_problem1(env):
    env.add_obstacle(8., 30., 8., 30.,)
    env.set_start(0., 0.)
    env.set_goal(36., 36.)
    return env

def setup_problem2(env):
    env.add_obstacle(8., 30., 8., 30.,)
    env.add_obstacle(-10., 10., -60, 30.,)
    env.set_start(-50., -50.)
    env.set_goal(50., 50.)
    return env

def setup_problem3(env):
    env.add_obstacle(8., 30., 8., 30.,)
    env.add_obstacle(-10., 10., -60, 30.,)
    env.add_obstacle(-100., -30., 20, 50.,)
    env.set_start(-50., -50.)
    env.set_goal(50., 50.)
    return env

def setup_problem4(env):
    env.add_obstacle(8., 30., 8., 30.,)
    env.add_obstacle(-10., 10., -60, 30.,)
    env.add_obstacle(-100., -30., 20, 50.,)
    env.add_obstacle(0., 20., 20, 100.,)
    env.set_start(-50., -50.)
    env.set_goal(50., 50.)
    return env

def setup_problem_narrow_passage_v1(env):
    env.set_size(100., 0.)
    env.add_obstacle(45., 55., 0., 49.)
    env.add_obstacle(45., 55., 51., 100.)
    env.set_start(100., 100.)
    env.set_goal(5., 91.)
    return env

def setup_problem_narrow_passage(env):
    env.set_size(100., 0.)
    env.add_obstacle(45., 55., 0., 49.95)
    env.add_obstacle(45., 55., 50.05, 100.)
    env.set_start(100., 100.)
    env.set_goal(5., 91.)
    return env

def get_planar_arm_env():
    arm = PlanarArm(links=[25., 25., 25.])
    env = PlanarArmWorld(arm, 100, q_init=[0.,0.,0.])
    env.set_size(100., 0.)
    return env

def setup_planar_arm_prob1(env):
    env.add_obstacle(38., 60., 8., 30.,)
    env.add_obstacle(20., 40., -60, 30.,)
    env.add_obstacle(-100., -50., 20, 50.,)
    env.add_obstacle(30., 50., 20, 100.,)
    env.set_start(np.pi/2, 0., 0.)
    env.set_goal(-np.pi/2, -np.pi/3, -np.pi/2)
    return env
