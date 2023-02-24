# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

import torch


def quat_loss(q1, q2):
    '''
    Compute quaternion error metric
    q1, q2 = (x y z w)
    '''
    a = q1*q2
    return 1 - ((q1*q2).sum(dim=-1).pow(2))


class WeightedPoseDistance(object):
    """ Create a weighted distance metric, for pose losses or whatever """

    def __init__(self, position_wt=1., orientation_wt=0.001):
        self.position_wt = position_wt
        self.orientation_wt = orientation_wt

    def __call__(self, pq1, pq2):
        pos = self.position_wt * torch.sum((pq1[:, :3] - pq2[:, :3])**2, dim=-1)
        rot = self.orientation_wt * quat_loss(pq1[:, 3:], pq2[:, 3:])
        return pos + rot


def compose_qp(q, pt):
    '''
    Assume we're dealing with unit quaternions. $q$ is a quaternion, $p$ is a
    position.
    '''
    pt2 = torch.zeros(q.shape[0], 3)
    px, py, pz = pt[:,0], pt[:,1], pt[:,2]
    x = q[:,0]
    y = q[:,1]
    z = q[:,2]
    qx = q[:,3]
    qy = q[:,4]
    qz = q[:,5]
    qw = q[:,6]

    qxx = qx**2
    qyy = qy**2
    qzz = qz**2
    qwx = qw*qx
    qwy = qw*qy
    qwz = qw*qz
    qxy = qx*qy
    qxz = qx*qz
    qyz = qy*qz

    pt2[:, 0] = x + px + 2*(
            (-1*(qyy + qzz)*px) +
            ((qxy-qwz)*py) +
            ((qwy+qxz)*pz))
    pt2[:,1] = y + py + 2*(
            ((qwz+qxy)*px) +
            (-1*(qxx+qzz)*py) +
            ((qyz-qwx)*pz))
    #print("---")
    #print(qxz, qwx, qxz-qwx)
    #print(qxx, qyy)
    #print(qwx, qyz, qxz+qyz)
    #print(z+pz-(2*(qxx+qyy)*pz))
    #print(px, py)
    #print("----")
    pt2[:,2] = z + pz + 2*(
            ((qxz-qwy)*px) +
            ((qwx+qyz)*py) +
            (-1*(qxx+qyy)*pz)
            )
    return pt2

def compose_qq(q1, q2):
    '''
    Compose two poses represented as:
        [x,y,z,qx,qy,qz,qw]
    '''
    QX, QY, QZ, QW = 3, 4, 5, 6

    qww = q1[:,QW]*q2[:,QW]
    qxx = q1[:,QX]*q2[:,QX]
    qyy = q1[:,QY]*q2[:,QY]
    qzz = q1[:,QZ]*q2[:,QZ]
    # For new qx
    q1w2x = q1[:,QW]*q2[:,QX]
    q2w1x = q2[:,QW]*q1[:,QX]
    q1y2z = q1[:,QY]*q2[:,QZ]
    q2y1z = q2[:,QY]*q1[:,QZ]
    # For new qy
    q1w2y = q1[:,QW]*q2[:,QY]
    q2w1y = q2[:,QW]*q1[:,QY]
    q1z2x = q1[:,QZ]*q2[:,QX]
    q2z1x = q2[:,QZ]*q1[:,QX]
    # For new qz
    q1w2z = q1[:,QW]*q2[:,QZ]
    q2w1z = q2[:,QW]*q1[:,QZ]
    q1x2y = q1[:,QX]*q2[:,QY]
    q2x1y = q2[:,QX]*q1[:,QY]
    
    q3 = torch.zeros(q1.shape)
    q3[:,:3] = compose_qp(q1,q2[:,:3])
    q3[:,QX] = (q1w2x+q2w1x+q1y2z-q2y1z)
    q3[:,QY] = (q1w2y+q2w1y+q1z2x-q2z1x)
    q3[:,QZ] = (q1w2z+q2w1z+q1x2y-q2x1y)
    q3[:,QW] = (qww - qxx - qyy - qzz)
    return q3

if __name__ == '__main__':
    import numpy as np
    import PyKDL as kdl

    q0 = torch.zeros(1,7); q0[0,-1] = 1.
    pt0 = torch.zeros(1,3); pt0[0,1] = 0.2; pt0[0,2] = 0.4
    print("ex1:", q0, pt0)
    pt0 = compose_qp(q0, pt0)
    print("result:", pt0)

    #r1 = kdl.Rotation.RPY(-1*np.pi/4,np.pi,np.pi/2)
    #r1 = kdl.Rotation.RPY(0,0,0.923)
    quat = np.array([1,2,3,4])
    quat = quat / np.linalg.norm(quat)
    r1 = kdl.Rotation.Quaternion(*quat)
    p1 = kdl.Vector(-0.123,0.2,0.6)
    q1 = torch.zeros(1,7)
    q1[0,3:] = torch.Tensor(np.array(list(r1.GetQuaternion())))
    q1[0,0] = 0.1
    pt1 = torch.zeros(1,3)
    pt1[0] = torch.Tensor(np.array(list(p1)))

    T1a = kdl.Frame(r1, kdl.Vector(0.1,0,0))
    T1b = kdl.Frame(kdl.Rotation.RPY(0,0,0), p1)
    T1 = T1a*T1b
    print("ex2:", q1, pt1)
    print("kdl result:", T1.p, T1.M.GetQuaternion())
    print("result:", compose_qp(q1, pt1))


    r2 = kdl.Rotation.RPY(0.2*np.pi/4,0.5*np.pi,-1*np.pi/2)
    p2 = kdl.Vector(1,0,0)
    T1 = kdl.Frame(r1,p1)
    T2 = kdl.Frame(r2,p2)
    q1[0,3:] = torch.Tensor(np.array(list(r1.GetQuaternion())))
    q1[0,:3] = torch.Tensor(np.array(list(p1)))
    q2 = torch.zeros(1,7)
    q2[0,3:] = torch.Tensor(np.array(list(r2.GetQuaternion())))
    q2[0,:3] = torch.Tensor(np.array(list(p2)))
    res = compose_qq(q1,q2)
    Tres = T1 * T2
    print(res)
    print(Tres.p, Tres.M.GetQuaternion())
    print("....", r1.GetQuaternion())
