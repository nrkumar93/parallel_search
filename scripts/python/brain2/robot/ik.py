# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from abc import ABCMeta
import multiprocessing as mp
import queue as Queue


class AbstractIKSolver(object):
    """Wrapper for inverse kinematics"""

    def __init__(self, domain=None):
        """Takes in an object reference? Not for now. Empty."""
        self.domain = domain
        
    def is_parallel(self):
        return False

    def __call__(self, robot_ref, goal_pose, q0=None):
        """if pose or config are provided, make sure to set them first.
        otherwise get something else"""
        return NotImplemented

    def reset(self):
        """In case you have some state internal to this that needs to be
        used -- clear it here"""
        pass

    def solve_batch(self, poses, q0s, robot_ref):
        """ Used to take a whole bunch of poses to query at once """
        raise NotImplementedError()

# TODO take a look at this and figure out why its slow
# https://gitlab-master.nvidia.com/amousavian/closed_loop/-/blob/master/self_supervised/mppi.py#L35-82
class IKProcess(mp.Process):
    def __init__(self, output_queue, ik_solver):
        super(IKProcess, self).__init__()
        self.output_queue = output_queue
        self.input_queue = mp.Queue()
        self.ik_solver = ik_solver
        self.robot_ref = None
        self.running = True

    def _ik(self, robot_ref, ee_pose, q0, pose_as_matrix=True):
        """ do the actual inverse kinematics work """
        return self.ik_solver(robot_ref, ee_pose, q0=q0,
                              pose_as_matrix=pose_as_matrix)

    def ik(self, robot_ref, ee_pose, q0, idx, pose_as_matrix=True):
        """ Add one to the queue """
        self.robot_ref = robot_ref
        self.input_queue.put((ee_pose, q0, pose_as_matrix, idx))

    def run(self):
        while self.running:
            try:
                request = self.input_queue.get(timeout=1)
            except Queue.Empty:
                continue
            pose, q0, as_matrix = request[:3]
            idx = request[3]
            self.output_queue.put(
                    (
                        self._ik(self.robot_ref, pose, q0=q0,
                                 pose_as_matrix=as_matrix),
                        idx
                    )
            )

    def stop_running(self):
        self.running = False


class ParallelIKSolver(AbstractIKSolver):
    """ This solver can be used to create an inverse kinematics thing that works
    in parallel. """

    def __init__(self):
        self.processes = []
        self.output_queue = mp.Queue()
        self.num_processes = 0
        self.idx = 0
        self.running = False

    def add_ik_solver(self, ik_solver):
        self.processes.append(IKProcess(self.output_queue, ik_solver))
        self.num_processes += 1

    def __delete__(self):
        for proc in self.processes:
            proc.stop_running()

    def is_parallel(self):
        return self.running

    def solve_batch(self, poses, q0s, robot_ref, pose_as_matrix=True):
        """
        World state and actor should be the same for now
        """
        num_grasps = len(poses)
        for i, (pose, q0) in enumerate(zip(poses, q0s)):
            # Add all the information we need for each ik query
            # Add the index so that we can find our results later
            self.processes[self.idx].ik(robot_ref, pose, q0, i, pose_as_matrix)
            self.idx = (self.idx + 1) % self.num_processes
        
        results = [None] * num_grasps
        for _ in range(num_grasps):
            output = self.output_queue.get(True)
            assert isinstance(output, tuple)
            assert len(output) == 2
            idx = output[1]
            results[idx] = output[0]
        return results

    def __call__(self, robot_ref, goal_pose, q0=None, pose_as_matrix=True):
        #self.processes[self.idx].ik(ws, actor, pose, q0, 0)
        #self.idx = (self.idx + 1) % self.num_processes
        # TODO: actually return this - wait until we get a response
        return self.processes[0].ik_solver(robot_ref, goal_pose, q0=q0,
                pose_as_matrix=pose_as_matrix)

        
    def start(self):
        for proc in self.processes:
            proc.daemon = True
            proc.start()
        self.running = True
