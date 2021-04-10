import numpy as np
from numba import njit, float64, int64, intp, void
from numba.types import UniTuple, Tuple
from numba.pycc import CC

from math import acos, sin, cos, atan, pi, radians


cc = CC('_transforms')
cc.verbose = True


EPSILON = np.finfo(np.float32).eps
EULER_SAFE  = np.array([0,1,2,0],dtype='int64')
EULER_NEXT  = np.array([1,2,0,1],dtype='int64')
EULER_ORDER = np.array([0,8,16,4,12,20],dtype='int64')
MAYA_EA     = np.array([[0,1,2],[1,2,0],[2,0,1],[0,2,1],[1,0,2],[2,1,0]],dtype='int64')  # maya 2 euler angle
EA_MAYA     = np.array([[0,1,2],[2,0,1],[1,2,0],[0,2,1],[1,0,2],[2,1,0]],dtype='int64')  # euler angle 2 maya


@cc.export('_atan2','float64(float64,float64)')
@njit
def _atan2(y,x):
    """ Because of a linking error i can't fix at the moment, i replaced all calls for atan2
        by this formula: https://en.wikipedia.org/wiki/Atan2#Definition

        If we switch back to @jit functions instead of aot, math.atan2 can be used just fine
    """
    if x > 0.:
        return atan(y/x)

    elif x < 0. and y >= 0.:
        return atan(y/x) + pi

    elif x < 0. and y < 0.:
        return atan(y/x) - pi

    elif x== 0. and y > 0.:
        return pi/2.

    elif x == 0. and y < 0.:
        return -pi/2.

    else:
        return 0.


@cc.export('_getEulerOrder','UniTuple(int64,7)(int64)')
@njit
def _getEulerOrder(axes):
        o_ = EULER_ORDER[axes]
        f=o_&1; o_>>=1; s=o_&1; o_>>=1; n=o_&1; o_>>=1
        i=EULER_SAFE[o_&3]
        j=EULER_NEXT[i+n]
        k=EULER_NEXT[i+1-n]
        h=i
        if s:
            h=k

        return i,j,k,h,n,s,f



@cc.export('_eulerToMatrix','float64[:,:,:](float64[:,:], int64[:])')
@njit
def _eulerToMatrix(euler, axes):
    """ Converts Maya euler angles to 4x4 matrices
    """

    # Find the maximum indices to use for clipping purposes
    max0, max1 = euler.shape[0]-1, axes.shape[0]-1
    maxsize = max(max0, max1) + 1
    matrix   = np.empty((maxsize,4,4))
    ea_ = np.empty(3)

    for ii in range(maxsize):
        index0 = min(ii, max0)
        index1 = min(ii, max1)
        ea_[0], ea_[1], ea_[2] = euler[index0,0], euler[index0,1], euler[index0,2]
        ea_[0], ea_[1], ea_[2] = ea_[MAYA_EA[axes[index1],0]], ea_[MAYA_EA[axes[index1],1]], ea_[MAYA_EA[axes[index1],2]]

        i,j,k,h,n,s,f = _getEulerOrder(axes[index1])

        if s:
            h=k
        if f == 1:
            ea_[0], ea_[2] = ea_[2], ea_[0]
        if n == 1:
            ea_[0], ea_[1], ea_[2] = -ea_[0], -ea_[1], -ea_[2]

        ci = cos(ea_[0]); cj = cos(ea_[1]); ch = cos(ea_[2])
        si = sin(ea_[0]); sj = sin(ea_[1]); sh = sin(ea_[2])
        cc = ci*ch
        cs = ci*sh
        sc = si*ch
        ss = si*sh

        if s:
            matrix[ii, i, i] = cj
            matrix[ii, j, i] = sj*si
            matrix[ii, k, i] = sj*ci
            matrix[ii, i, j] = sj*sh
            matrix[ii, j, j] = -cj*ss+cc
            matrix[ii, k, j] = -cj*cs-sc
            matrix[ii, i, k] = -sj*ch
            matrix[ii, j, k] = cj*sc+cs
            matrix[ii, k, k] = cj*cc-ss

        else:
            matrix[ii, i, i] = cj*ch
            matrix[ii, j, i] = sj*sc-cs
            matrix[ii, k, i] = sj*cc+ss
            matrix[ii, i, j] = cj*sh
            matrix[ii, j, j] = sj*ss+cc
            matrix[ii, k, j] = sj*cs-sc
            matrix[ii, i, k] = -sj
            matrix[ii, j, k] = cj*si
            matrix[ii, k, k] = cj*ci

        matrix[ii, 0, 3] = 0.0
        matrix[ii, 1, 3] = 0.0
        matrix[ii, 2, 3] = 0.0
        matrix[ii, 3, 0] = 0.0
        matrix[ii, 3, 1] = 0.0
        matrix[ii, 3, 2] = 0.0
        matrix[ii, 3, 3] = 1.0


    return matrix



@cc.export('_eulerToQuaternion','float64[:,:](float64[:,:], int64[:])')
@njit
def _eulerToQuaternion(euler, axes):
    """ Converts list of Maya euler angles to quaternions qi,qj,qk,qw
    """

    # Find the maximum indices to use for clipping purposes
    max0, max1 = euler.shape[0]-1, axes.shape[0]-1
    maxsize = max(max0, max1) + 1
    quat = np.empty((maxsize,4))
    ea_ = np.empty(3)

    for ii in range(maxsize):
        index0 = min(ii, max0)
        index1 = min(ii, max1)
        ea_[0], ea_[1], ea_[2] = euler[index0,0], euler[index0,1], euler[index0,2]
        ea_[0], ea_[1], ea_[2] = ea_[MAYA_EA[axes[index1],0]], ea_[MAYA_EA[axes[index1],1]], ea_[MAYA_EA[axes[index1],2]]

        i,j,k,h,n,s,f = _getEulerOrder(axes[index1])

        if s:
            h=k
        if f == 1:
            ea_[0], ea_[2] = ea_[2], ea_[0]
        if n == 1:
            ea_[1] = -ea_[1]


        ti = ea_[0]*0.5; tj = ea_[1]*0.5; th = ea_[2]*0.5
        ci = cos(ti); cj = cos(tj); ch = cos(th)
        si = sin(ti); sj = sin(tj); sh = sin(th)
        cc = ci*ch
        cs = ci*sh
        sc = si*ch
        ss = si*sh

        if s == 1:
            quat[ii,0] = cj*(cs + sc)
            quat[ii,1] = sj*(cc + ss)
            quat[ii,2] = sj*(cs - sc)
            quat[ii,3] = cj*(cc - ss)

        else:
            quat[ii,0] = cj*sc - sj*cs
            quat[ii,1] = cj*ss + sj*cc
            quat[ii,2] = cj*cs - sj*sc
            quat[ii,3] = cj*cc + sj*ss

        if n == 1:
            quat[ii,j] = -quat[ii,j]

    return quat




@cc.export('_matrixIdentity','float64[:,:,:](int64)')
@njit
def _matrixIdentity(count):
    matrix = np.empty((count,4,4))
    for ii in range(count):
        matrix[ii,0,0] = 1.; matrix[ii,0,1] = 0.; matrix[ii,0,2] = 0.; matrix[ii,0,3] = 0.;
        matrix[ii,1,0] = 0.; matrix[ii,1,1] = 1.; matrix[ii,1,2] = 0.; matrix[ii,1,3] = 0.;
        matrix[ii,2,0] = 0.; matrix[ii,2,1] = 0.; matrix[ii,2,2] = 1.; matrix[ii,2,3] = 0.;
        matrix[ii,3,0] = 0.; matrix[ii,3,1] = 0.; matrix[ii,3,2] = 0.; matrix[ii,3,3] = 1.;

    return matrix



@cc.export('_matrixToEuler','float64[:,:](float64[:,:,:], int64[:])')
@njit
def _matrixToEuler(matrix, axes):
    """ Converts list of 4x4 matrices to Maya euler angles.
    """
    # Find the maximum indices to use for clipping purposes
    max0, max1 = matrix.shape[0]-1, axes.shape[0]-1
    maxsize = max(max0, max1) + 1
    euler = np.empty((maxsize,3))
    m_ = np.empty((3,3))


    for ii in range(maxsize):
        index0 = min(ii, max0)
        index1 = min(ii, max1)
        i,j,k,h,n,s,f = _getEulerOrder(axes[index1])

        # Normalize xyz axes in case of scale
        x,y,z = 0.,0.,0.
        for jj in range(3):
            x = x + matrix[index0,0,jj] ** 2
            y = y + matrix[index0,1,jj] ** 2
            z = z + matrix[index0,2,jj] ** 2

        x = x ** 0.5
        y = y ** 0.5
        z = z ** 0.5

        for jj in range(3):
            m_[jj,0] = matrix[index0,0,jj] / x
            m_[jj,1] = matrix[index0,1,jj] / y
            m_[jj,2] = matrix[index0,2,jj] / z

        if s:
            yy = (m_[i,j]**2 + m_[i,k]**2) ** 0.5
            euler[ii,1] = _atan2( yy,  m_[i,i])

            if yy > EPSILON:
                euler[ii,0] = _atan2( m_[i,j],  m_[i,k])
                euler[ii,2] = _atan2( m_[j,i], -m_[k,i])
            else:
                euler[ii,0] = _atan2(-m_[j,k],  m_[j,j])
                euler[ii,2] = 0.

        else:
            yy = (m_[i,i]**2 + m_[j,i]**2) ** 0.5

            euler[ii,1] = _atan2(-m_[k,i],  yy)

            if yy > EPSILON:
                euler[ii,0] = _atan2( m_[k,j],  m_[k,k])
                euler[ii,2] = _atan2( m_[j,i],  m_[i,i])
            else:
                euler[ii,0] = _atan2(-m_[j,k],  m_[j,j])
                euler[ii,2] = 0.

        if n:
            euler[ii,0], euler[ii,1], euler[ii,2] = -euler[ii,0], -euler[ii,1], -euler[ii,2]

        if f:
            euler[ii,0], euler[ii,2] = euler[ii,2], euler[ii,0]


        # From euler angle to maya
        euler[ii,0], euler[ii,1], euler[ii,2]  = euler[ii,EA_MAYA[axes[index1],0]], euler[ii,EA_MAYA[axes[index1],1]], euler[ii,EA_MAYA[axes[index1],2]]


    return euler



@cc.export('_matrixToQuaternion','float64[:,:](float64[:,:,:])')
@njit
def _matrixToQuaternion(matrix):
    """ Converts 4x4 matrix to quaternions qi,qj,qk,qw
    """
    quat = np.empty((matrix.shape[0],4))

    for ii in range(matrix.shape[0]):
        trace = matrix[ii,0,0] + matrix[ii,1,1] + matrix[ii,2,2]

        if trace > 0.:
            s = 0.5 / (trace + 1.0) ** 0.5
            quat[ii,0] = (matrix[ii,1,2] - matrix[ii,2,1] ) * s
            quat[ii,1] = (matrix[ii,2,0] - matrix[ii,0,2] ) * s
            quat[ii,2] = (matrix[ii,0,1] - matrix[ii,1,0] ) * s
            quat[ii,3] = 0.25 / s

        elif matrix[ii,0,0] > matrix[ii,1,1] and matrix[ii,0,0] > matrix[ii,2,2]:
            s = 2.0 * ( 1.0 + matrix[ii,0,0] - matrix[ii,1,1] - matrix[ii,2,2]) ** 0.5
            quat[ii,0] = 0.25 * s
            quat[ii,1] = (matrix[ii,1,0] + matrix[ii,0,1] ) / s
            quat[ii,2] = (matrix[ii,2,0] + matrix[ii,0,2] ) / s
            quat[ii,3] = (matrix[ii,1,2] - matrix[ii,2,1] ) / s

        elif matrix[ii,1,1] > matrix[ii,2,2]:
            s = 2.0 * ( 1.0 + matrix[ii,1,1] - matrix[ii,0,0] - matrix[ii,2,2]) ** 0.5
            quat[ii,0] = (matrix[ii,1,0] + matrix[ii,0,1] ) / s
            quat[ii,1] = 0.25 * s
            quat[ii,2] = (matrix[ii,2,1] + matrix[ii,1,2] ) / s
            quat[ii,3] = (matrix[ii,2,0] - matrix[ii,0,2] ) / s

        else:
            s = 2.0 * ( 1.0 + matrix[ii,2,2] - matrix[ii,0,0] - matrix[ii,1,1]) ** 0.5
            quat[ii,0] = (matrix[ii,2,0] + matrix[ii,0,2] ) / s
            quat[ii,1] = (matrix[ii,2,1] + matrix[ii,1,2] ) / s
            quat[ii,2] = 0.25 * s
            quat[ii,3] = (matrix[ii,0,1] - matrix[ii,1,0] ) / s

    return quat



@cc.export('_matrixInverse','float64[:,:,:](float64[:,:,:])')
@njit
def _matrixInverse(matrix):
    """ Assumes matrix is 4x4 orthogonal
    """

    # Init inverse Matrix
    m_ = np.empty(matrix.shape)

    # For every matrix
    for i in range(matrix.shape[0]):

        # Calculate the scale components
        sx = (matrix[i,0,0]**2 + matrix[i,0,1]**2 + matrix[i,0,2]**2)
        sy = (matrix[i,1,0]**2 + matrix[i,1,1]**2 + matrix[i,1,2]**2)
        sz = (matrix[i,2,0]**2 + matrix[i,2,1]**2 + matrix[i,2,2]**2)

        # Normalize scale component
        m_[i,0,0] = matrix[i,0,0] / sx
        m_[i,0,1] = matrix[i,1,0] / sx
        m_[i,0,2] = matrix[i,2,0] / sx
        m_[i,0,3] = 0.0
        m_[i,1,0] = matrix[i,0,1] / sy
        m_[i,1,1] = matrix[i,1,1] / sy
        m_[i,1,2] = matrix[i,2,1] / sy
        m_[i,1,3] = 0.0
        m_[i,2,0] = matrix[i,0,2] / sz
        m_[i,2,1] = matrix[i,1,2] / sz
        m_[i,2,2] = matrix[i,2,2] / sz
        m_[i,2,3] = 0.0
        m_[i,3,0] = -1 * (m_[i,0,0]*matrix[i,3,0] + m_[i,1,0]*matrix[i,3,1] + m_[i,2,0]*matrix[i,3,2])
        m_[i,3,1] = -1 * (m_[i,0,1]*matrix[i,3,0] + m_[i,1,1]*matrix[i,3,1] + m_[i,2,1]*matrix[i,3,2])
        m_[i,3,2] = -1 * (m_[i,0,2]*matrix[i,3,0] + m_[i,1,2]*matrix[i,3,1] + m_[i,2,2]*matrix[i,3,2])
        m_[i,3,3] = 1.0

    return m_



@cc.export('_matrixNormalize','float64[:,:,:](float64[:,:,:])')
@njit
def _matrixNormalize(matrix):
    """ Normalizes the rotation component of a transform matrix
    """
    m_ = np.empty(matrix.shape)

    # For every matrix
    for i in range(matrix.shape[0]):
        x = (matrix[i,0,0]**2 + matrix[i,0,1]**2 + matrix[i,0,2]**2)**0.5
        y = (matrix[i,1,0]**2 + matrix[i,1,1]**2 + matrix[i,1,2]**2)**0.5
        z = (matrix[i,2,0]**2 + matrix[i,2,1]**2 + matrix[i,2,2]**2)**0.5

        m_[i,0,0] = matrix[i,0,0]/x
        m_[i,0,1] = matrix[i,0,1]/x
        m_[i,0,2] = matrix[i,0,2]/x
        m_[i,0,3] = matrix[i,0,3]

        m_[i,1,0] = matrix[i,1,0]/y
        m_[i,1,1] = matrix[i,1,1]/y
        m_[i,1,2] = matrix[i,1,2]/y
        m_[i,1,3] = matrix[i,1,3]

        m_[i,2,0] = matrix[i,2,0]/z
        m_[i,2,1] = matrix[i,2,1]/z
        m_[i,2,2] = matrix[i,2,2]/z
        m_[i,2,3] = matrix[i,2,3]

        m_[i,3,0] = matrix[i,3,0]
        m_[i,3,1] = matrix[i,3,1]
        m_[i,3,2] = matrix[i,3,2]
        m_[i,3,3] = matrix[i,3,3]

    return m_










@cc.export('_matrixMultiply','float64[:,:,:](float64[:,:,:],float64[:,:,:])')
@njit
def _matrixMultiply(matrix0,matrix1):

    # Find the maximum indices to use for clipping purposes
    max0, max1 = matrix0.shape[0]-1, matrix1.shape[0]-1
    maxsize = max(max0, max1) + 1

    m = np.empty((maxsize,4,4))

    for i in range(maxsize):
        index0 = min(i, max0)
        index1 = min(i, max1)

        m[i,0,0] = matrix0[index0,0,0] * matrix1[index1,0,0] + matrix0[index0,0,1] * matrix1[index1,1,0] + matrix0[index0,0,2] * matrix1[index1,2,0] + matrix0[index0,0,3] * matrix1[index1,3,0]
        m[i,0,1] = matrix0[index0,0,0] * matrix1[index1,0,1] + matrix0[index0,0,1] * matrix1[index1,1,1] + matrix0[index0,0,2] * matrix1[index1,2,1] + matrix0[index0,0,3] * matrix1[index1,3,1]
        m[i,0,2] = matrix0[index0,0,0] * matrix1[index1,0,2] + matrix0[index0,0,1] * matrix1[index1,1,2] + matrix0[index0,0,2] * matrix1[index1,2,2] + matrix0[index0,0,3] * matrix1[index1,3,2]
        m[i,0,3] = matrix0[index0,0,0] * matrix1[index1,0,3] + matrix0[index0,0,1] * matrix1[index1,1,3] + matrix0[index0,0,2] * matrix1[index1,2,3] + matrix0[index0,0,3] * matrix1[index1,3,3]

        m[i,1,0] = matrix0[index0,1,0] * matrix1[index1,0,0] + matrix0[index0,1,1] * matrix1[index1,1,0] + matrix0[index0,1,2] * matrix1[index1,2,0] + matrix0[index0,1,3] * matrix1[index1,3,0]
        m[i,1,1] = matrix0[index0,1,0] * matrix1[index1,0,1] + matrix0[index0,1,1] * matrix1[index1,1,1] + matrix0[index0,1,2] * matrix1[index1,2,1] + matrix0[index0,1,3] * matrix1[index1,3,1]
        m[i,1,2] = matrix0[index0,1,0] * matrix1[index1,0,2] + matrix0[index0,1,1] * matrix1[index1,1,2] + matrix0[index0,1,2] * matrix1[index1,2,2] + matrix0[index0,1,3] * matrix1[index1,3,2]
        m[i,1,3] = matrix0[index0,1,0] * matrix1[index1,0,3] + matrix0[index0,1,1] * matrix1[index1,1,3] + matrix0[index0,1,2] * matrix1[index1,2,3] + matrix0[index0,1,3] * matrix1[index1,3,3]

        m[i,2,0] = matrix0[index0,2,0] * matrix1[index1,0,0] + matrix0[index0,2,1] * matrix1[index1,1,0] + matrix0[index0,2,2] * matrix1[index1,2,0] + matrix0[index0,2,3] * matrix1[index1,3,0]
        m[i,2,1] = matrix0[index0,2,0] * matrix1[index1,0,1] + matrix0[index0,2,1] * matrix1[index1,1,1] + matrix0[index0,2,2] * matrix1[index1,2,1] + matrix0[index0,2,3] * matrix1[index1,3,1]
        m[i,2,2] = matrix0[index0,2,0] * matrix1[index1,0,2] + matrix0[index0,2,1] * matrix1[index1,1,2] + matrix0[index0,2,2] * matrix1[index1,2,2] + matrix0[index0,2,3] * matrix1[index1,3,2]
        m[i,2,3] = matrix0[index0,2,0] * matrix1[index1,0,3] + matrix0[index0,2,1] * matrix1[index1,1,3] + matrix0[index0,2,2] * matrix1[index1,2,3] + matrix0[index0,2,3] * matrix1[index1,3,3]

        m[i,3,0] = matrix0[index0,3,0] * matrix1[index1,0,0] + matrix0[index0,3,1] * matrix1[index1,1,0] + matrix0[index0,3,2] * matrix1[index1,2,0] + matrix0[index0,3,3] * matrix1[index1,3,0]
        m[i,3,1] = matrix0[index0,3,0] * matrix1[index1,0,1] + matrix0[index0,3,1] * matrix1[index1,1,1] + matrix0[index0,3,2] * matrix1[index1,2,1] + matrix0[index0,3,3] * matrix1[index1,3,1]
        m[i,3,2] = matrix0[index0,3,0] * matrix1[index1,0,2] + matrix0[index0,3,1] * matrix1[index1,1,2] + matrix0[index0,3,2] * matrix1[index1,2,2] + matrix0[index0,3,3] * matrix1[index1,3,2]
        m[i,3,3] = matrix0[index0,3,0] * matrix1[index1,0,3] + matrix0[index0,3,1] * matrix1[index1,1,3] + matrix0[index0,3,2] * matrix1[index1,2,3] + matrix0[index0,3,3] * matrix1[index1,3,3]


    return m



@cc.export('_matrixPointMultiply','float64[:,:](float64[:,:],float64[:,:,:])')
@njit
def _matrixPointMultiply(point,matrix):

    # Find the maximum indices to use for clipping purposes
    max0, max1 = point.shape[0]-1, matrix.shape[0]-1
    maxsize = max(max0, max1) + 1
    p = np.empty((maxsize,3))

    for i in range(maxsize):
        index0 = min(i, max0)
        index1 = min(i, max1)

        p[i,0] = (matrix[index1,0,0] * point[index0,0]) + (matrix[index1,1,0] * point[index0,1]) + (matrix[index1,2,0] * point[index0,2]) + matrix[index1,3,0]
        p[i,1] = (matrix[index1,0,1] * point[index0,0]) + (matrix[index1,1,1] * point[index0,1]) + (matrix[index1,2,1] * point[index0,2]) + matrix[index1,3,1]
        p[i,2] = (matrix[index1,0,2] * point[index0,0]) + (matrix[index1,1,2] * point[index0,1]) + (matrix[index1,2,2] * point[index0,2]) + matrix[index1,3,2]

    return p




@cc.export('_vectorSlerp','float64[:,:](float64[:,:],float64[:,:],float64[:])')
@njit
def _vectorSlerp(vector0,vector1,weight):

    # Find the maximum indices to use for clipping purposes
    max0, max1, max2 = vector0.shape[0]-1, vector1.shape[0]-1, weight.shape[0]-1
    maxsize = max(max0, max1, max2) + 1

    v = np.empty((maxsize,3))
    vector0_ = np.empty(3)
    vector1_ = np.empty(3)

    for i in range(maxsize):
        index0 = min(i, max0)
        index1 = min(i, max1)
        index2 = min(i, max2)

        m = (vector0[index0,0]**2 + vector0[index0,1]**2 + vector0[index0,2]**2) ** 0.5
        if m > 0.:
            vector0_[0] = vector0[index0,0] / m
            vector0_[1] = vector0[index0,1] / m
            vector0_[2] = vector0[index0,2] / m

        m = (vector1[index1,0]**2 + vector1[index1,1]**2 + vector1[index1,2]**2) ** 0.5
        if m > 0.:
            vector1_[0] = vector1[index1,0] / m
            vector1_[1] = vector1[index1,1] / m
            vector1_[2] = vector1[index1,2] / m

        angle  = acos((vector0_[0]*vector1_[0]) + (vector0_[1]*vector1_[1]) + (vector0_[2]*vector1_[2]))
        sangle = sin(angle)
        if sangle > 0.:
            w0 = sin((1-weight[index2]) * angle)
            w1 = sin(weight[index2] * angle)

            v[i,0] = (vector0[index0,0] * w0 + vector1[index1,0] * w1) / sangle
            v[i,1] = (vector0[index0,1] * w0 + vector1[index1,1] * w1) / sangle
            v[i,2] = (vector0[index0,2] * w0 + vector1[index1,2] * w1) / sangle
        else:
            v[i,0] = vector0[index0,0]
            v[i,1] = vector0[index0,1]
            v[i,2] = vector0[index0,2]


    return v



@cc.export('_vectorLerp','float64[:,:](float64[:,:],float64[:,:],float64[:],intp)')
@njit
def _vectorLerp(vector0,vector1,weight,method):

    # method 0 = linear
    # method 1 = pow multiply (used for scale)

    # Find the maximum indices to use for clipping purposes
    max0, max1, max2 = vector0.shape[0]-1, vector1.shape[0]-1, weight.shape[0]-1
    maxsize = max(max0, max1, max2) + 1

    v = np.empty((maxsize,vector0.shape[1]))
    if  method == 0:
        for i in range(maxsize):
            index0 = min(i, max0)
            index1 = min(i, max1)
            index2 = min(i, max2)

            for j in range(vector0.shape[1]):
                #weight_ = 1-weight[index2]
                #v[i,j] = vector1[index1,j]*weight[index2] + vector0[index0,j]*weight_
                v[i,j] = vector0[index0,j] + weight[index2]*(vector1[index1,j] - vector0[index0,j])
                
                
    else:
        for i in range(maxsize):
            index0 = min(i, max0)
            index1 = min(i, max1)
            index2 = min(i, max2)

            for j in range(vector0.shape[1]):
                v[i,j] = vector1[index1,j]**weight[index2] * vector0[index0,j]**(1-weight[index2])

    return v




@cc.export('_vectorToMatrix','float64[:,:,:](float64[:,:], float64[:,:], int64[:], int64[:], intp)')
@njit
def _vectorToMatrix(vector0, vector1, aim_axis, up_axis, extrapolate):

    # Find the maximum indices to use for clipping purposes
    max0, max1, max2, max3 = vector0.shape[0]-1, vector1.shape[0]-1, aim_axis.shape[0]-1, up_axis.shape[0]-1
    maxsize = max(max0, max1, max2, max3) + 1

    vector0_ = np.empty(3)
    vector1_ = np.empty(3)
    matrix   = np.empty((maxsize,4,4))


    for i in range(maxsize):
        index0 = min(i, max0)
        index1 = min(i, max1)
        index2 = min(i, max2)
        index3 = min(i, max3)

        ii = aim_axis[index2]
        jj = up_axis[index3]
        kk = (min(ii,jj)-max(ii,jj)+min(ii,jj)) % 3

        flip = 0
        if ii == 0 and jj == 2:
            flip = 1
        elif ii == 1 and jj == 0:
            flip = 1
        elif ii == 2 and jj == 1:
            flip = 1


        for j in range(3):
            vector0_[j] = vector0[index0,j]
            vector1_[j] = vector1[index1,j]

        if extrapolate and i > 0:
            if index0 > index1:
                for j in range(3):
                    vector1_[j] = matrix[i-1,jj,j] # up vector = last up vector in matrix

            elif index0 < index1:
                for j in range(3):
                    vector0_[j] = matrix[i-1,ii,j]  # aim vector = last up vector in matrix


        # init matrix output
        matrix[i,0,0] = 1.
        matrix[i,0,1] = 0.
        matrix[i,0,2] = 0.
        matrix[i,0,3] = 0.

        matrix[i,1,0] = 0.
        matrix[i,1,1] = 1.
        matrix[i,1,2] = 0.
        matrix[i,1,3] = 0.

        matrix[i,2,0] = 0.
        matrix[i,2,1] = 0.
        matrix[i,2,2] = 1.
        matrix[i,2,3] = 0.

        matrix[i,3,0] = 0.
        matrix[i,3,1] = 0.
        matrix[i,3,2] = 0.
        matrix[i,3,3] = 1.

        x = vector0_[1] * vector1_[2] - vector0_[2] * vector1_[1]
        y = vector0_[2] * vector1_[0] - vector0_[0] * vector1_[2]
        z = vector0_[0] * vector1_[1] - vector0_[1] * vector1_[0]

        na = (vector0_[0]**2 + vector0_[1]**2 + vector0_[2]**2) ** 0.5
        nc = (x**2 + y**2 + z**2) ** 0.5        

        # Fill in the matrix values if we're in a valid state
        if na > 0. and nc > 0.:        
            matrix[i,kk,0] = x/nc
            matrix[i,ii,0] = vector0_[0] / na
            matrix[i,kk,1] = y/nc
            matrix[i,ii,1] = vector0_[1] / na
            matrix[i,kk,2] = z/nc
            matrix[i,ii,2] = vector0_[2] / na

            matrix[i,jj,0] = matrix[i,kk,1] * matrix[i,ii,2] - matrix[i,kk,2] * matrix[i,ii,1]
            matrix[i,jj,1] = matrix[i,kk,2] * matrix[i,ii,0] - matrix[i,kk,0] * matrix[i,ii,2]
            matrix[i,jj,2] = matrix[i,kk,0] * matrix[i,ii,1] - matrix[i,kk,1] * matrix[i,ii,0]

            if flip:
                matrix[i,kk,0] = 0 - matrix[i,kk,0]
                matrix[i,kk,1] = 0 - matrix[i,kk,1]
                matrix[i,kk,2] = 0 - matrix[i,kk,2]

    return matrix



@cc.export('_vectorCross','float64[:,:](float64[:,:], float64[:,:])')
@njit
def _vectorCross(vector0, vector1):
    # Find the maximum indices to use for clipping purposes
    max0, max1 = vector0.shape[0]-1, vector1.shape[0]-1
    maxsize = max(max0, max1) + 1
    vector = np.empty((maxsize,3))

    for i in range(maxsize):
        index0 = min(i, max0)
        index1 = min(i, max1)

        vector[i,0] = vector0[index0, 1] * vector1[index1, 2] - vector0[index0, 2] * vector1[index1, 1]
        vector[i,1] = vector0[index0, 2] * vector1[index1, 0] - vector0[index0, 0] * vector1[index1, 2]
        vector[i,2] = vector0[index0, 0] * vector1[index1, 1] - vector0[index0, 1] * vector1[index1, 0]

    return vector


@cc.export('_vectorDot','float64[:](float64[:,:], float64[:,:])')
@njit
def _vectorDot(vector0, vector1):
    # Find the maximum indices to use for clipping purposes
    max0, max1 = vector0.shape[0]-1, vector1.shape[0]-1
    maxsize = max(max0, max1) + 1
    dot = np.empty(maxsize)

    for i in range(maxsize):
        index0 = min(i, max0)
        index1 = min(i, max1)

        dot[i] = 0

        for j in range(vector0.shape[1]):
            dot[i] += vector0[index0, j] * vector1[index1, j]

    return dot



@cc.export('_vectorMagnitude','float64[:](float64[:,:])')
@njit
def _vectorMagnitude(vector):
    mag = np.empty(vector.shape[0])
    for i in range(vector.shape[0]):
        mag[i] = 0.
        for j in range(vector.shape[1]):
            mag[i] = mag[i] + vector[i,j]**2

        mag[i] = mag[i] ** 0.5

    return mag



@cc.export('_vectorNormalize','float64[:,:](float64[:,:])')
@njit
def _vectorNormalize(vector):
    vector_ = np.empty(vector.shape)

    # For every vector
    for i in range(vector.shape[0]):

        # calc the magnitude
        mag = 0
        for j in range(vector.shape[1]):
            mag += vector[i,j]**2

        mag = mag ** 0.5

        # normalize
        for j in range(vector.shape[1]):
            vector_[i,j] = vector[i,j]/mag

    return vector_





@cc.export('_quaternionSlerp','float64[:,:](float64[:,:],float64[:,:],float64[:])')
@njit
def _quaternionSlerp(quat0, quat1, weight):
    """ Calculates spherical interpolation between two lists quaternions
    """

    # Find the maximum indices to use for clipping purposes
    max0, max1, max2 = quat0.shape[0]-1, quat1.shape[0]-1, weight.shape[0]-1
    maxsize = max(max0, max1, max2) + 1
    quat = np.empty((maxsize,4))

    for ii in range(maxsize):
        index0 = min(ii, max0)
        index1 = min(ii, max1)
        index2 = min(ii, max2)


        # If quat0=quat1 or quat0=-quat1 then theta = 0 and we can return quat0
        cosHalfTheta = quat0[index0,3] * quat1[index1,3] + quat0[index0,0] * quat1[index1,0] + quat0[index0,1] * quat1[index1,1] + quat0[index0,2] * quat1[index1,2]
        if abs(cosHalfTheta) >= 1.0:
            quat[ii,0], quat[ii,1], quat[ii,2], quat[ii,3] = quat0[index0,0], quat0[index0,1], quat0[index0,2], quat0[index0,3]

        else:
            halfTheta = acos(cosHalfTheta)
            sinHalfTheta  = (1.0 - cosHalfTheta*cosHalfTheta) ** 0.5

            # If theta = 180 degrees then result is not fully defined
            # we could rotate around any axis normal to quat0 or quat1
            if abs(sinHalfTheta) < EPSILON:
                quat[ii,0] = (quat0[index0,0] * 0.5 + quat1[index1,0] * 0.5)
                quat[ii,1] = (quat0[index0,1] * 0.5 + quat1[index1,1] * 0.5)
                quat[ii,2] = (quat0[index0,2] * 0.5 + quat1[index1,2] * 0.5)
                quat[ii,3] = (quat0[index0,3] * 0.5 + quat1[index1,3] * 0.5)

            else:
                ratioA = sin((1 - weight[index2]) * halfTheta) / sinHalfTheta
                ratioB = sin(weight[index2] * halfTheta) / sinHalfTheta

                quat[ii,0] = (quat0[index0,0] * ratioA + quat1[index1,0] * ratioB)
                quat[ii,1] = (quat0[index0,1] * ratioA + quat1[index1,1] * ratioB)
                quat[ii,2] = (quat0[index0,2] * ratioA + quat1[index1,2] * ratioB)
                quat[ii,3] = (quat0[index0,3] * ratioA + quat1[index1,3] * ratioB)

    return quat




@cc.export('_quaternionMultiply','float64[:,:](float64[:,:], float64[:,:])')
@njit
def _quaternionMultiply(quat0, quat1):
    """ Multiplies 2 lists of quaternions
    """

    # Find the maximum indices to use for clipping purposes
    max0, max1 = quat0.shape[0]-1, quat1.shape[0]-1
    maxsize = max(max0, max1) + 1
    quat = np.empty((maxsize,4))

    for ii in range(maxsize):
        index0 = min(ii, max0)
        index1 = min(ii, max1)

        quat[ii,0] =  quat0[index0,0] * quat1[index1,3] + quat0[index0,1] * quat1[index1,2] - quat0[index0,2] * quat1[index1,1] + quat0[index0,3] * quat1[index1,0]
        quat[ii,1] = -quat0[index0,0] * quat1[index1,2] + quat0[index0,1] * quat1[index1,3] + quat0[index0,2] * quat1[index1,0] + quat0[index0,3] * quat1[index1,1]
        quat[ii,2] =  quat0[index0,0] * quat1[index1,1] - quat0[index0,1] * quat1[index1,0] + quat0[index0,2] * quat1[index1,3] + quat0[index0,3] * quat1[index1,2]
        quat[ii,3] = -quat0[index0,0] * quat1[index1,0] - quat0[index0,1] * quat1[index1,1] - quat0[index0,2] * quat1[index1,2] + quat0[index0,3] * quat1[index1,3]


    return quat



@cc.export('_quaternionAdd','float64[:,:](float64[:,:], float64[:,:])')
@njit
def _quaternionAdd(quat0, quat1):
    """ Multiplies 2 lists of quaternions
    """

    # Find the maximum indices to use for clipping purposes
    max0, max1 = quat0.shape[0]-1, quat1.shape[0]-1
    maxsize = max(max0, max1) + 1
    quat = np.empty((maxsize,4))

    for ii in range(maxsize):
        index0 = min(ii, max0)
        index1 = min(ii, max1)

        quat[ii,0] = quat0[index0,0] + quat1[index1,0]
        quat[ii,1] = quat0[index0,1] + quat1[index1,1]
        quat[ii,2] = quat0[index0,2] + quat1[index1,2]
        quat[ii,3] = quat0[index0,3] + quat1[index1,3]

    return quat



@cc.export('_quaternionSub','float64[:,:](float64[:,:], float64[:,:])')
@njit
def _quaternionSub(quat0, quat1):
    """ Multiplies 2 lists of quaternions
    """

    # Find the maximum indices to use for clipping purposes
    max0, max1 = quat0.shape[0]-1, quat1.shape[0]-1
    maxsize = max(max0, max1) + 1
    quat = np.empty((maxsize,4))

    for ii in range(maxsize):
        index0 = min(ii, max0)
        index1 = min(ii, max1)

        quat[ii,0] = quat0[index0,0] - quat1[index1,0]
        quat[ii,1] = quat0[index0,1] - quat1[index1,1]
        quat[ii,2] = quat0[index0,2] - quat1[index1,2]
        quat[ii,3] = quat0[index0,3] - quat1[index1,3]

    return quat




@cc.export('_quaternionToMatrix','float64[:,:,:](float64[:,:])')
@njit
def _quaternionToMatrix(quat):

    # Init Matrix
    matrix = np.empty((quat.shape[0],4,4))

    # For every quaternion
    for i in range(quat.shape[0]):
        xx = quat[i,0]*quat[i,0]
        xy = quat[i,0]*quat[i,1]
        xz = quat[i,0]*quat[i,2]
        xw = quat[i,0]*quat[i,3]

        yy = quat[i,1]*quat[i,1]
        yz = quat[i,1]*quat[i,2]
        yw = quat[i,1]*quat[i,3]

        zz = quat[i,2]*quat[i,2]
        zw = quat[i,2]*quat[i,3]

        matrix[i,0,0] = 1 - 2 * ( yy + zz )
        matrix[i,1,0] =     2 * ( xy - zw )
        matrix[i,2,0] =     2 * ( xz + yw )

        matrix[i,0,1] =     2 * ( xy + zw )
        matrix[i,1,1] = 1 - 2 * ( xx + zz )
        matrix[i,2,1] =     2 * ( yz - xw )

        matrix[i,0,2] =     2 * ( xz - yw )
        matrix[i,1,2] =     2 * ( yz + xw )
        matrix[i,2,2] = 1 - 2 * ( xx + yy )

        matrix[i,0,3] = 0.
        matrix[i,1,3] = 0.
        matrix[i,2,3] = 0.
        matrix[i,3,0] = 0.
        matrix[i,3,1] = 0.
        matrix[i,3,2] = 0.
        matrix[i,3,3] = 1.

    return matrix



@cc.export('_quaternionConjugate','float64[:,:](float64[:,:])')
@njit
def _quaternionConjugate(quat):
    """ Conjugates a list of quaternions
    """
    quat_ = np.empty((quat.shape[0],4))

    # For every quaternion
    for i in range(quat.shape[0]):
        quat_[i,0] = -quat[i,0]
        quat_[i,1] = -quat[i,1]
        quat_[i,2] = -quat[i,2]
        quat_[i,3] =  quat[i,3]

    return quat_


@cc.export('_quaternionInverse','float64[:,:](float64[:,:])')
@njit
def _quaternionInverse(quat):
    """ Inverses a list of quaternions
    """
    quat_ = np.empty((quat.shape[0],4))

    # For every quaternion
    for i in range(quat.shape[0]):
        mag = quat[i,0]**2 + quat[i,1]**2 + quat[i,2]**2 + quat[i,3]**2

        quat_[i,0] = -quat[i,0]/mag
        quat_[i,1] = -quat[i,1]/mag
        quat_[i,2] = -quat[i,2]/mag
        quat_[i,3] =  quat[i,3]/mag

    return quat_



@cc.export('_quaternionNegate','float64[:,:](float64[:,:])')
@njit
def _quaternionNegate(quat):
    """ Negates a list of quaternions
    """
    quat_ = np.empty((quat.shape[0],4))

    # For every quaternion
    for i in range(quat.shape[0]):
        quat_[i,0] = -quat[i,0]
        quat_[i,1] = -quat[i,1]
        quat_[i,2] = -quat[i,2]
        quat_[i,3] = -quat[i,3]

    return quat_




@cc.export('_axisAngleToQuaternion','float64[:,:](float64[:,:], float64[:])')
@njit
def _axisAngleToQuaternion(axis,angle):
    """ Computes a list of quaternions qi,qj,qk,qw from lists of axes and angles
    """
    # Find the maximum indices to use for clipping purposes
    max0, max1 = axis.shape[0]-1, angle.shape[0]-1
    maxsize = max(max0, max1) + 1
    quat = np.empty((maxsize,4))
    axis_ = np.empty(3)

    for ii in range(maxsize):
        index0 = min(ii, max0)
        index1 = min(ii, max1)

        mag = (axis[index0,0] ** 2 + axis[index0,1] ** 2 + axis[index0,2] ** 2) ** 0.5
        axis_[0] = axis[index0,0]/mag
        axis_[1] = axis[index0,1]/mag
        axis_[2] = axis[index0,2]/mag


        s = sin(angle[index1]/2)
        quat[ii,0] = axis_[0] * s
        quat[ii,1] = axis_[1] * s
        quat[ii,2] = axis_[2] * s
        quat[ii,3] = cos(angle[index1]/2)

    return quat



@cc.export('_axisAngleToMatrix','float64[:,:,:](float64[:,:], float64[:])')
@njit
def _axisAngleToMatrix(axis,angle):
    """ Computes a list of orthogonal 4x4 matrices from lists of axes and angles
    """

    # Find the maximum indices to use for clipping purposes
    max0, max1 = axis.shape[0]-1, angle.shape[0]-1
    maxsize = max(max0, max1) + 1
    matrix = np.empty((maxsize,4,4))

    for ii in range(maxsize):
        index0 = min(ii, max0)
        index1 = min(ii, max1)

        sin_ = sin(angle[index1])
        cos_ = cos(angle[index1])
        inv_cos = 1-cos_

        mag = (axis[index0,0] ** 2 + axis[index0,1] ** 2 + axis[index0,2] ** 2) ** 0.5
        u = axis[index0,0]/mag
        v = axis[index0,1]/mag
        w = axis[index0,2]/mag
        uv = u*v
        uw = u*w
        vw = v*w
        usin = u*sin_
        vsin = v*sin_
        wsin = w*sin_
        u2 = u**2
        v2 = v**2
        w2 = w**2

        matrix[ii,0,0] = u2 + ((v2 + w2) * cos_)
        matrix[ii,0,1] = uv*inv_cos + (wsin)
        matrix[ii,0,2] = uw*inv_cos - (vsin)
        matrix[ii,1,0] = uv*inv_cos - (wsin)
        matrix[ii,1,1] = v2 + ((u2 + w2) * cos_)
        matrix[ii,1,2] = vw*inv_cos + (usin)
        matrix[ii,2,0] = uw*inv_cos + (vsin)
        matrix[ii,2,1] = vw*inv_cos - (usin)
        matrix[ii,2,2] = w2 + ((u2 + v2) * cos_)
        matrix[ii,0,3] = 0.
        matrix[ii,1,3] = 0.
        matrix[ii,2,3] = 0.
        matrix[ii,3,0] = 0.
        matrix[ii,3,1] = 0.
        matrix[ii,3,2] = 0.
        matrix[ii,3,3] = 1.

    return matrix




@cc.export('_vectorArcToQuaternion','float64[:,:](float64[:,:], float64[:,:])')
@njit
def _vectorArcToQuaternion(vector0,vector1):
    """ Computes a list of quaternions qi,qj,qk,qw representing the arc between lists of vectors
    """

    max0, max1 = vector0.shape[0]-1, vector1.shape[0]-1
    maxsize = max(max0, max1) + 1
    quat = np.empty((maxsize,4))

    v0 = np.empty(3) # normalized vector0
    v1 = np.empty(3) # normalized vector1
    h  = np.empty(3) # normalized vector0 + vector1

    for ii in range(maxsize):
        index0 = min(ii, max0)
        index1 = min(ii, max1)

        # normalized vector0
        mag = (vector0[index0,0]**2 + vector0[index0,1]**2 + vector0[index0,2]**2) ** 0.5
        v0[0] = vector0[index0,0]/mag
        v0[1] = vector0[index0,1]/mag
        v0[2] = vector0[index0,2]/mag

        # normalized vector1
        mag = (vector1[index1,0]**2 + vector1[index1,1]**2 + vector1[index1,2]**2) ** 0.5
        v1[0] = vector1[index1,0]/mag
        v1[1] = vector1[index1,1]/mag
        v1[2] = vector1[index1,2]/mag

        # normalized vector0 + vector1
        h[0] = v0[0] + v1[0]
        h[1] = v0[1] + v1[1]
        h[2] = v0[2] + v1[2]

        mag = (h[0]**2 + h[1]**2 + h[2]**2) ** 0.5

        h[0] = h[0] / mag
        h[1] = h[1] / mag
        h[2] = h[2] / mag

        # Generate arc-vector quaternion
        quat[ii,0] = v0[1]*h[2] - v0[2]*h[1]
        quat[ii,1] = v0[2]*h[0] - v0[0]*h[2]
        quat[ii,2] = v0[0]*h[1] - v0[1]*h[0]
        quat[ii,3] = v0[0]*h[0] + v0[1]*h[1] + v0[2]*h[2]


    return quat




#---Mix and Match---#


@cc.export('_quaternionToEuler','float64[:,:](float64[:,:], int64[:])')
@njit
def _quaternionToEuler(quat,axes):
    """ Converts quaternions qi,qj,qk,qw to euler angles
    """
    return _matrixToEuler(_quaternionToMatrix(quat), axes)


@cc.export('_eulerSlerp','float64[:,:](float64[:,:], float64[:,:], float64[:], int64[:], int64[:], int64[:])')
@njit
def _eulerSlerp(euler0,euler1,weight,axes0,axes1,axes):
    """ Spherical interpaxeslatiaxesn between twaxes euler angles
    """
    return _quaternionToEuler(_quaternionSlerp(_eulerToQuaternion(euler0,axes0),_eulerToQuaternion(euler1,axes1),weight),axes)


@cc.export('_eulerToEuler','float64[:,:](float64[:,:],int64[:],int64[:])')
@njit
def _eulerToEuler(euler, axes0, axes1):
    """ Converts euler angles from one rotate order to another.
    """
    return _quaternionToEuler(_eulerToQuaternion(euler,axes0),axes1)


@cc.export('_axisAngleToEuler','float64[:,:](float64[:,:], float64[:], int64[:])')
@njit
def _axisAngleToEuler(axis, angle, axes):
    """ Computes a list of euler angles from given axis and angles
    """
    return _matrixToEuler(_axisAngleToMatrix(axis,angle),axes)


@cc.export('_matrixLocal','float64[:,:,:](float64[:,:,:],float64[:,:,:])')
@njit
def _matrixLocal(matrix,parent_matrix):
    """ Calculates the local matrix from a child to its parent
    """
    return _matrixMultiply(matrix,_matrixInverse(parent_matrix))


@cc.export('_vectorToQuaternion','float64[:,:](float64[:,:], float64[:,:], int64[:], int64[:], intp)')
@njit
def _vectorToQuaternion(vector0, vector1, aim_axis, up_axis, extrapolate):
    return _matrixToQuaternion(_vectorToMatrix(vector0, vector1, aim_axis, up_axis, extrapolate))


@cc.export('_vectorToEuler','float64[:,:](float64[:,:], float64[:,:], int64[:], int64[:], int64[:], intp)')
@njit
def _vectorToEuler(vector0, vector1, aim_axis, up_axis, axes, extrapolate):
    return _matrixToEuler(_vectorToMatrix(vector0, vector1, aim_axis, up_axis, extrapolate), axes)


@cc.export('_matrixSlerp','float64[:,:,:](float64[:,:,:],float64[:,:,:],float64[:])')
@njit
def _matrixSlerp(matrix0, matrix1, weight):
    return _quaternionToMatrix(_quaternionSlerp(_matrixToQuaternion(matrix0),_matrixToQuaternion(matrix1),weight))


@cc.export('_quaternionDot','float64[:](float64[:,:], float64[:,:])')
@njit
def _quaternionDot(quat0, quat1):
    return _vectorDot(quat0, quat1)


@cc.export('_quaternionNormalize','float64[:,:](float64[:,:])')
@njit
def _quaternionNormalize(quat):
    return _vectorNormalize(quat)


@cc.export('_vectorArcToMatrix','float64[:,:,:](float64[:,:], float64[:,:])')
@njit
def _vectorArcToMatrix(vector0,vector1):
    return _quaternionToMatrix(_vectorArcToQuaternion(vector0,vector1))


@cc.export('_vectorArcToEuler','float64[:,:](float64[:,:], float64[:,:], int64[:])')
@njit
def _vectorArcToEuler(vector0, vector1, axes):
    return _quaternionToEuler(_vectorArcToQuaternion(vector0,vector1), axes)



if __name__ == "__main__":
    cc.compile()
