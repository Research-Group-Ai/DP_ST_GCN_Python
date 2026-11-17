"""Kinect-v2 25-joint tree and fixed normalized graph construction (zero based)."""
import numpy as np
# Index convention must be checked against the downloaded dataset release.
JOINT_NAMES = ['spine_base','spine_mid','neck','head','shoulder_left','elbow_left',
 'wrist_left','hand_left','shoulder_right','elbow_right','wrist_right','hand_right',
 'hip_left','knee_left','ankle_left','foot_left','hip_right','knee_right',
 'ankle_right','foot_right','spine_shoulder','handtip_left','thumb_left',
 'handtip_right','thumb_right']
PARENTS = np.array([0,0,20,2,20,4,5,6,20,8,9,10,0,12,13,14,0,16,17,18,1,7,7,11,11])


def adjacency(mode='dual', cross_joints=None):
    if mode not in ('single','dual','dual_no_cross'):
        raise ValueError('Unknown graph mode')
    v = 25 if mode == 'single' else 50
    a = np.zeros((v,v), np.float32)
    for offset in range(0,v,25):
        for child,parent in enumerate(PARENTS):
            if child != parent:
                a[offset+child,offset+parent]=a[offset+parent,offset+child]=1
    if mode == 'dual':
        if cross_joints is None or len(cross_joints)!=9 or len(set(cross_joints))!=9:
            raise ValueError('Supply nine distinct cross-person joint indices.')
        for j in cross_joints:
            if not isinstance(j,int) or not 0<=j<25:
                raise ValueError('Cross joint indices must be integers in [0,24].')
            a[j,j+25]=a[j+25,j]=1
    a += np.eye(v,dtype=np.float32)  # exactly one self loop per node
    inv = 1/np.sqrt(a.sum(axis=1))
    return inv[:,None]*a*inv[None,:]
