# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

import numpy as np
import os
import yaml
import h5py

from pyquaternion import Quaternion

import brain2.utils.transformations as tra
import brain2.utils.axis as axis

# Pose info
import brain2.robot.kitchen_poses as kitchen_poses

# Basics
from brain2.bullet.interface import BulletInterface
from brain2.utils.extend import simple_extend
from brain2.utils.info import logwarn

# Robot and planning stuff
from brain2.motion_planners.problem import MotionPlanningProblem
from brain2.robot.surface import Surface
from brain2.robot.samplers import *
from brain2.robot.affordance import *
from brain2.robot.cube import *
from brain2.robot.kitchen import GetHandleEndPoses


default_assets_path = "./assets/urdf/"
# default_franka_home_q = [0.01200158428400755, -0.5697816014289856,
#                          5.6801487517077476e-05,
#                          -2.8105969429016113, -0.00025768374325707555, 3.0363450050354004,
#                          0.7410701513290405, -0.04, 0.04]
#default_franka_home_q = [0.00020981034659145166, -0.9068717458473784, -0.12026004237459417,
#                         -2.5031618405681004, -0.011846827605535024, 1.9764829670853086,
#                         0.6822841902705312, 0.03957270458340645, 0.03957270458340645]
default_franka_home_q = np.array([0., -0.9, -0.12,
                                -2.5, 0., 2.0, 0.68,])
#default_franka_home_q = np.array([0., -1.2, 0.,
#                                -2.7, 0., 2.1, 0.75,])
                                # 0.04, 0.04])

ycb_extent = {
        "spam": [0, 0, 0.075 * 2],
        "mustard": [0, 0, 0.11 * 2],
        "tomato_soup": [0, 0, 0.088 * 2],
    }


def load_franka(iface, assets_path, name="robot", camera=None, mobile=False,
                padding=0.):
    """Simple helper function for loading the robot."""

    if assets_path is None:
        assets_path = default_assets_path

    if camera is not None:
        if mobile:
            model_file = "franka_description/robots/franka_carter_%s.urdf" % (str(camera))
        else:
            model_file = "franka_description/robots/franka_panda_%s.urdf" % (str(camera))
    else:
        # Default model with no camera
        if mobile:
            model_file = "franka_description/robots/franka_panda_hand_on_carter.urdf"
        else:
            model_file = "franka_description/robots/franka_panda.urdf"

    # Load URDF
    logwarn("Loading robot with path = " + str(assets_path) + model_file)
    robot = iface.load_urdf(name=name,
                            pkg=assets_path,
                            fixed_base=True,
                            model=model_file,
                            padding=padding)

    robot = iface.get_object(name)
    robot.set_active_joints(["panda_joint%d" % (i+1) for i in range(7)])
    # ["panda_finger_joint%d" % (i+1) for i in range(2)])
    robot.set_allowed_self_collisions("panda_hand_joint", "panda_joint7", True)
    robot.set_allowed_self_collisions("panda_joint1", "panda_joint3", True)
    robot.set_allowed_self_collisions("panda_joint1", "panda_joint4", True)
    robot.set_allowed_self_collisions("panda_joint1", "panda_joint5", True)
    robot.set_allowed_self_collisions("panda_joint2", "panda_joint4", True)
    robot.set_allowed_self_collisions("panda_joint2", "panda_joint5", True)
    robot.set_allowed_self_collisions("panda_joint3", "panda_joint5", True)
    robot.set_allowed_self_collisions("panda_joint3", "panda_joint7", True)
    robot.set_allowed_self_collisions("panda_joint4", "panda_joint6", True)
    robot.set_allowed_self_collisions("panda_joint4", "panda_joint7", True)
    robot.set_allowed_self_collisions("panda_joint4", "panda_joint8", True)
    robot.set_allowed_self_collisions("panda_joint5", "panda_joint7", True)
    robot.set_allowed_self_collisions("panda_joint5", "panda_joint8", True)
    robot.set_allowed_self_collisions("panda_finger_joint2", "panda_hand_joint", True)
    robot.set_gripper_joints(["panda_finger_joint1", "panda_finger_joint2"], 0.04, 0)

    # Restrict motion in final joints
    #robot.active_min[-1] = -1
    #robot.active_max[-1] = 1

    robot.set_joint_positions(default_franka_home_q)
    robot.set_ee("right_gripper")
    robot.verbose = 1

    # Metadata
    robot.set_home_config(default_franka_home_q)
    robot.open_gripper()

    # --------------------
    # Load the franka hand into the planning scene too
    model_file = "franka_description/robots/franka_hand.urdf"
    hand  = iface.load_urdf(name=name+"_hand",
                            pkg=assets_path,
                            physical=False, # Not a real object
                            fixed_base=True,
                            model=model_file)
    hand.set_gripper_joints(["panda_finger_joint1", "panda_finger_joint2"], 0.04, 0)
    hand.set_allowed_self_collisions("panda_hand_joint", "panda_hand_base", True)
    hand.set_allowed_self_collisions("panda_hand_joint", "panda_finger_joint2", True)
    hand.open_gripper()
    robot.set_ee_ref(hand)
    robot.set_mobile(mobile)

    # Return reference
    return iface.get_object(name)


def load_sektion_cabinet(iface, assets_path, name="cabinet"):
    """Simple helper function for loading the robot."""
    if assets_path is None:
        assets_path = default_assets_path
    cabinet = iface.load_urdf(name=name,
                            pkg=assets_path,
                            fixed_base=True,
                            model="sektion_cabinet_model/urdf/sektion_cabinet.urdf")

    cabinet = iface.get_object(name)
    surface = Surface(name="top",
                      offset=np.array([0, 0, 0.42]),
                      extent=[0.6, 0.7, 0.1],
                      rgba=[1., 0., 0., 0.25])
    cabinet.add_surface(surface)


def load_hand(iface, assets_path, hand):
    """ load a marker so we don't hit hands by accident.
    Hand will be created with name provided.
    :param iface - planning interface to physics
    :param assets_path - where files are stored
    :param hand - left or right, which one this is"""
    if assets_path is None:
        assets_path = default_assets_path
    if hand not in ["left", "right"]:
        raise RuntimeError('hand not recognized: ' + str(hand))

    name = "%s_hand_marker" % (hand)
    filename = "blocks/" + name + ".urdf"

    block = iface.load_urdf(name=hand,
                            pkg=assets_path,
                            model=filename,
                            physical=True)

    # Return obj reference
    return iface.get_object(hand)
    

def load_block(iface, assets_path, size, color, id=None):
    """ Load a block """
    if assets_path is None:
        assets_path = default_assets_path
    if size not in ["big", "small", "median"]:
        raise RuntimeError('size not supported: ' + str(size))
    if color not in ["red", "blue", "yellow", "green"]:
        raise RuntimeError('color not supported: ' + str(color))
    size_name = "%s_block_%s" % (size, color)
    name = "%s_block" % (color)
    filename = "blocks/" + size_name + ".urdf"
    if id is not None:
        name += "_%02d" % (int(id))
    block = iface.load_urdf(name=name,
                            pkg=assets_path,
                            model=filename)

    get_grasps = CubeGraspLookupTable(approach_distance=-0.1,
                                           cage_distance=-0.03,
                                           offset_axis=axis.Z)
    get_stack = CubeStackLookupTable(approach_distance=-0.2,  # 0.12
                                               cage_distance=-0.05,  # 0.07
                                               offset_axis=axis.Z,
                                               use_angled=False)
    block.add_affordance("grasp", DiscreteAffordance(get_grasps.grasps))
    block.add_affordance("stack", DiscreteAffordance(get_stack.grasps))

    # Return obj reference
    return iface.get_object(name)



def load_kitchen(iface, assets_path):
    """Simple helper function for loading the robot, kitchen, and another table
    like we have set up in Seattle."""
    if assets_path is None:
        assets_path = default_assets_path


    # ------------------------------------- -
    # Load and set up the kitchen
    cabinet = iface.load_urdf(name="kitchen",
                            pkg=assets_path,
                            #model="kitchen_description/urdf/kitchen_part_right_gen_obj.urdf",
                            model="kitchen_description/urdf/kitchen_part_right_gen_convex.urdf",
                            fixed_base=True)
    cabinet.set_active_joints(["indigo_drawer_top_joint", "indigo_drawer_bottom_joint",
    "hitman_drawer_top_joint", "hitman_drawer_bottom_joint"])

    pose1, pose2 = GetHandleEndPoses()
    surface = Surface(name="indigo",
                      offset=np.array([0, 0, 0.05]),
                      #extent=[0.6, 0.7, 0.1],
                      extent=[0.55, 0.65, 0.1],
                      rgba=[1., 0., 0., 0.05],
                      parent_frame="indigo_countertop")
    surface.add_affordance("grasp", InterpolationAffordance(pose1, pose2))
    cabinet.add_surface(surface)
    surface = Surface(name="indigo_top_drawer",
                      offset=np.array([0, 0, 0.05]),
                      #extent=[0.6, 0.7, 0.1],
                      extent=[0.55, 0.65, 0.1],
                      rgba=[1., 0., 0., 0.05],
                      parent_frame="indigo_drawer_top")
    surface.add_affordance("grasp", InterpolationAffordance(pose1, pose2))
    cabinet.add_surface(surface)
    surface = Surface(name="indigo_bottom_drawer",
                      offset=np.array([0, 0, 0.05]),
                      #extent=[0.6, 0.7, 0.1],
                      extent=[0.55, 0.65, 0.1],
                      rgba=[1., 0., 0., 0.05],
                      parent_frame="indigo_drawer_bottom")
    cabinet.add_surface(surface)
    surface = Surface(name="hitman",
                      offset=np.array([0, 0, 0.05]),
                      #extent=[0.6, 0.7, 0.1],
                      extent=[0.55, 0.65, 0.1],
                      rgba=[1., 0., 0., 0.05],
                      parent_frame="hitman_countertop")
    cabinet.add_surface(surface)
    surface = Surface(name="hitman_top_drawer",
                      offset=np.array([0, 0, 0.05]),
                      #extent=[0.6, 0.7, 0.1],
                      extent=[0.55, 0.65, 0.1],
                      rgba=[1., 0., 0., 0.05],
                      parent_frame="hitman_drawer_top")
    cabinet.add_surface(surface)

    cabinet.set_pose(np.array([-1.0, 0, 1.3]), 
                     Quaternion(1, 0, 0, 0),
                     wxyz=True)

    # --------------------------------------
    # Load and set up the table
    table = iface.load_urdf(name="table",
                            pkg=assets_path,
                            model="simple_table.urdf",
                            fixed_base=True)

    # Add surface for placing stuff on the table
    surface = Surface(name="top",
                      offset=np.array([0, 0, 0.375]),
                      extent=[0.8, 2.1, 0.1],
                      rgba=[1., 0., 0., 0.05])
    table.add_surface(surface)

    # Add surface for floating stuff like hands
    surface = Surface(name="over",
                      offset=np.array([0.05, 0, 0.875]),
                      #extent=[1.3, 1.0, 1.0],
                      extent=[0.8, 2.1, 0.9],
                      rgba=[1., 1., 0., 0.05])
    table.add_surface(surface)
    
    # Show where table will be
    table.set_pose(np.array([2.0, 0, 0.375]), 
                   Quaternion(0, 0, 0, 1),
                   wxyz=True)

    # Set up the camera position
    # pos, quat =  [0.492, 0.160, 1.128], [-0.002, 0.823, -0.567, 0.010]
    # TODO: this should really be set by the system itself.
    #  Computed via:
    #   rosrun tf tf_echo measured/base_link depth_camera_link
    # pos, quat = [0.509, 0.993, 0.542], [-0.002, 0.823, -0.567, -0.010]
    # iface.set_camera((pos, quat), matrix=False)

    return cabinet


def load_simple_cart(iface, assets_path, name="table"):
    """Simple helper function for loading the robot."""
    if assets_path is None:
        assets_path = default_assets_path
    cabinet = iface.load_urdf(name=name,
                            pkg=assets_path,
                            fixed_base=True,
                            model="simple_cart.urdf")

    # Add surface for placing stuff on the table
    cabinet = iface.get_object(name)
    surface = Surface(name="top",
                      offset=np.array([0, 0, 0.375]),
                      extent=[1.2, 0.8, 0.02],
                      rgba=[1., 0., 0., 0.05])
    cabinet.add_surface(surface)

    surface = Surface(name="workspace",
                      offset=np.array([0.15, 0, 0.375]),
                      extent=[0.6, 0.6, 0.2],
                      rgba=[1., 0., 0., 0.05])
    cabinet.add_surface(surface)

    surface = Surface(name="left",
                      offset=np.array([0.15, 0.2, 0.375]),
                      extent=[0.6, 0.15, 0.2],
                      rgba=[1., 0., 0., 0.1])
    cabinet.add_surface(surface)

    surface = Surface(name="right",
                      offset=np.array([0.15, -0.2, 0.375]),
                      extent=[0.6, 0.15, 0.2],
                      rgba=[1., 0., 0., 0.1])
    cabinet.add_surface(surface)

    surface = Surface(name="far",
                      offset=np.array([0.4, 0, 0.375]),
                      extent=[0.2, 0.6, 0.2],
                      rgba=[1., 0., 0., 0.1])
    cabinet.add_surface(surface)

    surface = Surface(name="center",
                      offset=np.array([0.15, 0, 0.375]),
                      extent=[0.2, 0.2, 0.2],
                      rgba=[0., 1., 0., 0.1])
    cabinet.add_surface(surface)

    # Add surface for floating stuff like hands
    surface = Surface(name="over",
                      offset=np.array([0.05, 0, 0.875]),
                      #extent=[1.3, 1.0, 1.0],
                      extent=[1.1, 0.8, 0.9],
                      rgba=[1., 1., 0., 0.05])
    cabinet.add_surface(surface)

    return cabinet


def load_mocap_table(iface, assets_path, name="table"):
    """Simple helper function for loading the robot."""
    if assets_path is None:
        assets_path = default_assets_path
    cabinet = iface.load_urdf(name=name,
                            pkg=assets_path,
                            fixed_base=True,
                            model="mocap_table.urdf")

    # Add surface for placing stuff on the table
    cabinet = iface.get_object(name)
    surface = Surface(name="top",
                      offset=np.array([0, 0, 0]),
                      extent=[1.2, 0.8, 0.02],
                      rgba=[1., 0., 0., 0.05])
    cabinet.add_surface(surface)

    surface = Surface(name="workspace",
                      offset=np.array([0.15, 0, 0]),
                      extent=[0.6, 0.6, 0.2],
                      rgba=[1., 0., 0., 0.05])
    cabinet.add_surface(surface)

    # Add surface for floating stuff like hands
    surface = Surface(name="over",
                      offset=np.array([0.05, 0, 0.5]),
                      #extent=[1.3, 1.0, 1.0],
                      extent=[1.1, 0.8, 0.9],
                      rgba=[1., 1., 0., 0.05])
    cabinet.add_surface(surface)

    return cabinet


def load_ycb_object(iface, assets_path, name, obj_class):
    if obj_class is None:
        obj_class = name
    if assets_path is None:
        assets_path = default_assets_path
    #obj = iface.load_obj(name, pkg=assets_path,
    #                     model="../ycb/%s/textured.obj" % obj_class)
    obj = iface.load_urdf(name, pkg=assets_path,
                          model="../objects2/%s/model_normalized.urdf" % obj_class)

    grasps_path = os.path.join(assets_path, '../grasps/omg_ycb/simulated/' + obj_class + '.npy')
    print("Loading YCB object grasps from", grasps_path)
    #- Translation: [0.000, 0.000, 0.100]
    #- Rotation: in Quaternion [0.000, 0.000, 0.924, 0.383]
    #correction_pose = tra.quaternion_matrix([0, 0, 0.924, 0.383])
    correction_pose = tra.quaternion_matrix([0, 0, 0.707, 0.707])
    correction_pose[2,3] = 0.07 #0.207
    #correction_pose = tra.inverse_matrix(correction_pose)
    #grasps = [g.dot(correction_pose) for g in np.load(grasps_path)["poses"]]
    obj.add_affordance("grasp", GraspsAffordance(grasps_path, correction_pose))

    return obj

def load_simple_object(iface, assets_path, name, obj_class=None, urdf=True):
    """ get simple household objects """
    if obj_class is None:
        obj_class = name
    if assets_path is None:
        assets_path = default_assets_path
    if urdf:
        load_object(
            iface, 
            assets_path,
            name,
            "../objects/%s/google_16k/object.urdf" % obj_class
        )
    else:
        iface.load_obj(name, pkg=assets_path,
                #model="../ycb/%s/textured.obj" % obj_class)
                model="../objects/%s/google_16k/textured.obj" % obj_class,
                scale=0.001)
    return iface.get_object(name)


def load_object(iface, assets_path, name, urdf_path):
    """ 
    load any object given the urd_path and asset_path pointing to the 
    assets folder root.
    """
    iface.load_urdf(name=name,
                    pkg=assets_path,
                    model=urdf_path)
    
    return iface.get_object(name)


def simple_franka_with_objects(
    robot_assets_path, 
    robot_pose,
    object_assets_path,
    object_names,
    object_urdfs, 
    object_poses,
    gui=False, 
):
    """Create a franka and a list of objects with given poses."""
    iface = BulletInterface(gui=gui, add_ground_plane=False)
    load_franka(iface, robot_assets_path, "robot")
    iface.get_object('robot').set_pose(
        robot_pose[:3, 3],
        Quaternion(matrix=robot_pose),
        wxyz=True,
    )
    for name, urdf, pose in zip(object_names, object_urdfs, object_poses):
        if name == 'robot':
            raise ValueError(
                'object name cannot be robot! recieved {}'.format(name)
            )
        obj = load_object(
            iface,
            object_assets_path,
            name,
            urdf,
        )
        obj.set_pose(
            pose[:3, 3],
            Quaternion(matrix=pose),
            wxyz=True,
        )
    
    return iface



def simple_franka_cabinet(assets_path, gui=False, blocks=True):
    """Create a franka and a cabinet for some basic tests."""
    iface = BulletInterface(gui=gui, add_ground_plane=False)
    load_franka(iface, assets_path, "robot")
    load_sektion_cabinet(iface, assets_path, "cabinet")
    robot = iface.get_object("robot")
    cabinet = iface.get_object("cabinet")
    # Base link can touch cabinet
    robot.set_allowed_collisions(cabinet, idx=-1)
    robot.set_pose(np.array([0.0, 0, 0.6]),
                   Quaternion([1, 0, 0, 0,]),
                   wxyz=True)
    cabinet.set_pose(np.array([0.5, 0, 0.35]), 
                     Quaternion(0, 0, 0, 1),
                     wxyz=True)
    return iface


def simple_franka_cabinet_with_objects(assets_path, object_names, *args, **kwargs):
    """Put a list of objects on top of a franka cabinet."""
    env = simple_franka_cabinet(assets_path, *args, **kwargs)
    cabinet = env.get_object("cabinet")
    top = cabinet.get_surface("top")
    for obj_name in object_names:
        load_simple_object(env, assets_path, obj_name)
        obj = env.get_object(obj_name)
        pose = top.sample_pose(obj.default_pose)
        print("setting pose for", obj_name, "=\n", pose)
        pos = pose[:3, 3]
        quat = tra.quaternion_from_matrix(pose)
        obj.set_pose(pos, quat, wxyz=False)

    return env


def franka_cart(assets_path, gui, camera, padding=0.):
    if assets_path is None:
        assets_path = default_assets_path
    iface = BulletInterface(gui=gui, add_ground_plane=False)
    robot = load_franka(iface, assets_path, "robot", camera=camera,
                        padding=padding)
    table = load_simple_cart(iface, assets_path, "table")
    # Base link can touch cabinet
    robot.set_allowed_collisions(table, idx=-1)
    robot.set_allowed_collisions(table, idx=0)
    robot = iface.get_object("robot")
    table = iface.get_object("table")
    #robot.set_pose(np.array([0.15, 0.0, 0.7]),
    robot.set_pose(np.array([0, 0, 0]),
                   Quaternion([1, 0, 0, 0,]),
                   wxyz=True)
    table.set_pose(np.array([0.35, 0, -0.35]), 
                   Quaternion(0, 0, 0, 1),
                   wxyz=True)
    pos, quat = [0.509, 0.993, 0.542], [-0.002, 0.823, -0.567, -0.010]
    iface.set_camera((pos, quat), matrix=False)
    return iface


def franka_kitchen(assets_path, gui, camera, add_ground_plane=False):
    if assets_path is None:
        assets_path = default_assets_path
    iface = BulletInterface(gui=gui, add_ground_plane=add_ground_plane)
    robot = load_franka(iface, assets_path, "robot", camera=camera,
                        mobile=True)
    table = load_kitchen(iface, assets_path)
    # Base link can touch cabinet
    robot.set_allowed_collisions(table, idx=-1)
    robot = iface.get_object("robot")
    table = iface.get_object("table")
    #robot.set_pose(np.array([0, 0.0, 0.25]),
    #               Quaternion([0, 0, 0, 1,]),
    #               wxyz=True)
    robot.set_pose(np.array([1.2, 0.0, 0.25]),
                   Quaternion([1, 0, 0, 0,]),
                   wxyz=True)
    return iface


def franka_cart_objects(assets_path=None, gui=False, camera="d435",
                        objects={}, hands=False):
    """Create table plus blocks for block-stacking experiments."""
    iface = franka_cart(assets_path, gui, camera)
    table = iface.get_object("table")

    for obj_name, obj_type in objects.items():
        load_simple_object(iface, assets_path, obj_name, obj_type)
        obj = iface.get_object(obj_name)
        p, r = table.sample_surface_pose("top")
        obj.set_pose(p, r, wxyz=True)
        # obj.set_pose_matrix(pose)

    if hands:
        for i, hand in enumerate(["left", "right"]):
            hand_obj = load_hand(iface, assets_path, hand)
            x, y, z = 0.95 + (i * 0.1), 0., 1.
            hand_obj.set_pose((x, y, z), (0, 0, 0, 1))

    return iface


def franka_cart_blocks(assets_path=None, gui=False, camera="d435",
                       hands=False, padding=0.):
    """Create table plus blocks for block-stacking experiments."""

    if assets_path is None:
        assets_path = default_assets_path

    iface = franka_cart(assets_path, gui, camera, padding=padding)

    colors = ["red", "green", "blue", "yellow"]
    sizes = ["median"] * len(colors)
    xs = [0.45, 0.55, 0.45, 0.55]
    ys = [-0.05, -0.05, 0.05, 0.05]
    z = 0.72
    for s, c, x, y in zip(sizes, colors, xs, ys):
        block = load_block(iface, assets_path, size=s, color=c)
        block.set_pose((x, y, z), (1, 0, 0, 0,))

    if hands:
        for i, hand in enumerate(["left", "right"]):
            hand_obj = load_hand(iface, assets_path, hand)
            x, y, z = 0.95 + (i * 0.1), 0., 1.
            hand_obj.set_pose((x, y, z), (0, 0, 0, 1))
    return iface


def franka_kitchen_right(assets_path=None, gui=False, camera="d435",
                       hands=False, add_ground_plane=False, load_ycb=False,):
    """Create table plus blocks for block-stacking experiments."""

    if assets_path is None:
        assets_path = default_assets_path

    iface = franka_kitchen(assets_path, gui, camera, add_ground_plane)

    colors = ["red", "green", "blue", "yellow"]
    sizes = ["median"] * len(colors)
    xs = [1.8, 1.9, 1.8, 1.9]
    ys = [-0.05, -0.05, 0.05, 0.05]
    z = 0.75 + 0.02
    for s, c, x, y in zip(sizes, colors, xs, ys):
        block = load_block(iface, assets_path, size=s, color=c)
        block.set_pose((x, y, z), (1, 0, 0, 0,))

    if load_ycb:
        objs = [("cracker_box", "003_cracker_box"),
                ("sugar", "004_sugar_box"),
                ("tomato_soup", "005_tomato_soup_can"),
                ("mustard", "006_mustard_bottle"),
                #("pudding", "008_pudding_box"),
                #("jello", "009_gelatin_box"),
                ("spam", "010_potted_meat_can"),
                #("bowl", "024_bowl")]
                ]

        kitchen = iface.get_object("kitchen")
        surfaces = [kitchen.get_surface("indigo"), kitchen.get_surface("hitman")]
        for name, folder in objs:
            obj = load_ycb_object(iface, assets_path, name, folder)
            surface = np.random.choice(surfaces)
            pose = surface.sample_pose(np.eye(4))
            p = pose[:3, 3]
            r = tra.quaternion_from_matrix(pose)
            print(name, "was assigned pose =", (p, r))
            obj.set_pose(p, r, wxyz=True)

    if hands:
        for i, hand in enumerate(["left", "right"]):
            hand_obj = load_hand(iface, assets_path, hand)
            x, y, z = 0.95 + (i * 0.1), 0., 1.
            hand_obj.set_pose((x, y, z), (0, 0, 0, 1))
    return iface


def get_planning_problem(iface, q_goal, name="robot"):
    """Create environment and motion planning problem for flat world"""
    robot = iface.get_object(name)
    pb_config = {
            'dof': robot.dof,
            'p_sample_goal': 0.2,
            'iterations': 1000,
            'goal_iterations': 100,
            'verbose': 1,
            }
    get_goal = lambda: q_goal
    is_valid = lambda q: not iface.check_collisions(robot, q)
    is_done = lambda q: np.linalg.norm(q - q_goal) < 1e-4
    extend = lambda q1, q2: simple_extend(q1, q2, 0.2)
    return MotionPlanningProblem(sample_fn=robot.sample_uniform,
                                 goal_fn=get_goal,
                                 extend_fn=extend,
                                 is_valid_fn=is_valid,
                                 is_done_fn=is_done,
                                 config=pb_config,
                                 distance_fn=None)

def load_object_list(assets_path=None):
    if assets_path is None:
        assets_path = default_assets_path
    path = os.path.join(assets_path, "../object_list.yaml")
    with open(path, 'r') as f:
        data = yaml.load(f)
    return data
