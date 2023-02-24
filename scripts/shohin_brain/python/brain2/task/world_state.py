# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from __future__ import print_function

import numpy as np
import time
import threading


class WorldState(object):

    """
    tracks binary-valued predicates constituting the logical world state as well as objects
    """

    def __init__(self, domain=None):
        # Track states of all objects and the actor
        self.entities = {}
        self.actor = None
        self.domain = domain

        # Logical state is a binary numpy vector
        self.logical_state = None
        # Relaxed state is a binary numpy vector twice the size of logical state
        self.relaxed_state = None

        # Track parent state
        self.parent = None
        self.forked = False

        # Time estimate
        self.time = None

        # Check to see if this is currently tracking unreliable stuff.
        # This is a blackboard variable, but we sort of need to use it so we
        # can track the state of things.
        self.update_unreliable = False
        self.lock = threading.Lock()

    def update_relaxed(self):
        """ Create relaxed state for planning:
        0 ---> length: tracks true fluents
        length ---> 2 * length: tracks false fluents
        """
        n = self.logical_state.shape[0]
        self.relaxed_state = np.zeros((2 * n,))
        self.relaxed_state[:n] = self.logical_state
        self.relaxed_state[n:] = 1 - self.logical_state

    def set_update_unreliable(self):
        self.lock.acquire()
        self.update_unreliable = True
        self.lock.release()

    def freeze_unreliable(self):
        self.lock.acquire()
        self.update_unreliable = False
        self.lock.release()

    def items(self):
        """ Get items from the whole world state """
        return self.entities.items()

    def __getitem__(self, name):
        if name in self.entities:
            return self.entities[name]
        else:
            raise RuntimeError('entity not observed: ' + str(name))

    def __setitem__(self, name, value):
        if not self.forked:
            raise RuntimeError('cannot set continuous values for non-'
                               'hallucinated world states.')
        elif name not in self.entities:
            raise RuntimeError('entity not tracked: ' + str(name))
        else:
            self.entities[name] = value

    def fork(self):
        """
        Create a new world state from this one.
        """

        child = WorldState()

        # These should all be aliased and should not ever change!
        child.parent = self
        child.actor = self.actor
        child.domain = self.domain

        # This is a copy and is allowed to change
        child.logical_state = np.copy(self.logical_state)
        child.relaxed_state = np.copy(self.relaxed_state)

        # New dictionary to store entities -- should not be an alias of the
        # parent's entities. Once copied, these are also allowed to change.
        # TODO: verify that entities are copied properly when fork() is called
        for k, v in self.entities.items():
            child.entities[k] = v.copy()
        child.forked = True

        return child

    def __hash__(self):
        # hash the current logical state
        val = 0
        for i, predicate in self.logical_state:
            val += (predicate << i)
        return val


class EntityState(object):
    """
    Base class for the entity states associated with a particular world
    observation/hallucination.
    """

    def __init__(self):
        """
        Should initialize whatever you need for this entity.
        """
        self.name = None
        self.unreliable = False
        self.clear()

        # Attachment belief
        self.attached_to = None
        self.attached_pose = None

    def __str__(self):
        return "ENT:" + str(self.name)

    def clear(self):
        self.updated = False

         # Attachment belief
        self.attached_to = None
        self.attached_pose = None

        # Goals and other information
        self.goal_pose = None
        self.goal_obj = None

    def attach(self, to_obj, to_pose):
        self.attached_to = to_obj
        self.attached_pose = to_pose

    def detach(self):
        self.attached_to = None
        self.attached_pose = None

    def copy(self):
        """
        Must return a "safe" copy of the entity -- global values get aliased,
        but pose information has to be EXPLICTLY copied over.
        """
        raise NotImplementedError('copy() not implemente -- no planning'
                                  ' allowed')

    def is_unreliable(self, world_state):
        return (self.unreliable and not world_state.update_unreliable)


class WorldStateObserver(object):
    """
    This is the abstract class that generates new world states.
    """

    def __init__(self, domain):
        if not domain.compiled:
            raise RuntimeError("Cannot create observer until domain compiled!")

        self.domain = domain
        self.current_state = domain.get_state().fork()

        # Contain a list of objects that act as "sensors" and update our 
        # world knowledge.
        self.sensors = []

    def update(self, entities):
        raise NotImplementedError("update() must return an appropriate "
                                  "world state.")

    def set_update_unreliable(self):
        self.current_state.set_update_unreliable()

    def freeze_unreliable(self):
        self.current_state.freeze_unreliable()

    def observe(self, blocking=True, entities=None, *args, **kwargs):
        """
        Return the current world state after updating it.
        """
        # self.current_state.lock.acquire()
        if entities is None:
            entities = self.current_state.entities.keys()
        if blocking:
            while not self.update(entities, *args, **kwargs):
                print("Waiting for update() to succeed...")
                time.sleep(0.01)
        else:
            self.update(entities, *args, **kwargs)
        
        # Logical state updated here
        self.domain.update_logical(self.current_state)

        # self.current_state.lock.release()
        # self.update_unreliable = False
        return self.current_state
