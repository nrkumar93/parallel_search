# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function, division, absolute_import

import os
import math
import numpy as np
from carbongym import gymapi
from carbongym import gymutil

def create_gym_sim(desc="Simple Experiment"):
        """Creates the simulator"""

        # initialize gym
        gym = gymapi.acquire_gym()
        sim_params = gymapi.SimParams()
        args = gymutil.parse_arguments(description=desc)
        print("Args =", args)

        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 6
        sim_params.gravity = gymapi.Vec3(0, -9.8, 0)
        if not args.flex:
            sim_params.physx.rest_offset= 0.001
            sim_params.physx.contact_offset = 0.01
            sim_params.physx.bounce_threshold_velocity= 7.0
            # sim_params.physx.solver_type = 1 
            sim = gym.create_sim(
                args.compute_device_id, 
                args.graphics_device_id, 
                gymapi.SIM_PHYSX, 
                sim_params)  
        else:
            #sim_params.flex.solver_type = 5
            #sim_params.flex.num_outer_iterations = 4
            #sim_params.flex.num_inner_iterations = 20
            #sim_params.flex.relaxation = 0.9
            sim_params.flex.solver_type = 5
            sim_params.flex.num_outer_iterations = 5
            sim_params.flex.num_inner_iterations = 25
            sim_params.flex.relaxation = 0.75
            sim_params.flex.warm_start = 0.75
            sim_params.flex.shape_collision_margin = 0.003
            sim_params.flex.contact_regularization = 1e-7
            # sim_params.flex.geometric_stiffness = 1e3
            sim_params.flex.deterministic_mode = True
            sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
            sim = gym.create_sim(
                args.compute_device_id, 
                args.graphics_device_id, 
                gymapi.SIM_FLEX, 
                sim_params)
        return gym, sim

class Experiment:
    """
    Basic version of the experiment class. Copied from gym:
    https://carbon-gym.gitlab-master-pages.nvidia.com/carbgym/learning.html

    And then modified to add missing features.
    """

    def __init__(self, EnvClass, TaskClass, num_envs=64, env_param=None, task_param=None, asset_path=None):

        gym, sim = create_gym_sim()

        self._num_envs = num_envs
        self._gym = gym
        self._sim = sim

        # add a ground plane
        plane_params = gymapi.PlaneParams()
        gym.add_ground(sim, plane_params)

        # Set verbosity level; should really come from args.
        self.verbose = 0

        # acquire data to be shared by all environment instances
        if asset_path is not None:
            shared_data = EnvClass.create_shared_data(self._gym, self._sim, data_dir=asset_path)
        else:
            shared_data = EnvClass.create_shared_data(self._gym, self._sim)

        # env bounds
        spacing = 3.
        lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # create environment and task instances
        self.envs = []

        if env_param is None:
            env_param = dict()

        if task_param is None:
            task_param = dict()

        num_per_row = int(np.sqrt(num_envs))
        for i in range(num_envs):
            # create env instance
            env_ptr = self._gym.create_env(self._sim, lower, upper, num_per_row)
            env_base_args = {
                "gym": self._gym,
                "env": env_ptr,
                "env_index": i,
            }
            env = EnvClass(shared_data, **env_param, **env_base_args)
            self.envs.append(env)

        task_base_args={"gym":self._gym}
        self._task = TaskClass(self.envs, **task_param, **task_base_args)
        self._num_actions = self.envs[0].num_actions() #this is a safe assumption as it is required for vectorized training of any env
        self._num_obs = self._task.num_observations()

        self.observation_space=np.array([self._num_obs,])
        self.action_space = np.array([self._num_actions,])

    def get_num_obs(self):
        return self._num_obs

    def get_num_actions(self):
        return self._num_actions

    def reset(self):
        return self._task.reset(None)

    def step(self, actions):
        """
        Step() function takes in a set of actions and applies them to the environments.
        """

        # apply the actions
        for env in self.envs:
            env.step(actions)

        # simulate
        self._gym.simulate(self._sim)
        self._gym.fetch_results(self._sim, True)
        obs = self._task.fill_observations(actions)
        dones = self._task.fill_dones(actions)
        rews = self._task.fill_rewards(actions)

        indexer = np.where(dones)
        if indexer[0].size > 0:
            self._task.reset(dones.squeeze())

        return obs, rews, dones

    def spin(self):
        """
        Simple helper function to show UI and make sure things are loading properly.
        """
        viewer = self._gym.create_viewer(self._sim, gymapi.CameraProperties())
        if viewer is None:
            print("*** Failed to create viewer")
            quit()

        frame = 0
        while not self._gym.query_viewer_has_closed(viewer):
            # Block: run simulation and render in viewer
            self._gym.simulate(self._sim)
            self._gym.fetch_results(self._sim, True)
            self._gym.step_graphics(self._sim)
            self._gym.draw_viewer(viewer, self._sim, False)
            self._gym.sync_frame_time(self._sim)
            frame = frame + 1

        return frame

    def shutdown(self):
        if self.verbose:
            print("Shutting down.")
        self._gym.shutdown()
