import numpy as np
from numba import njit, prange #, float64, int32, intp
#from numba.types import UniTuple, Tuple
#from numba.pycc import CC

from math import acos, sin, cos, atan2


#cc = CC('_transforms')
#cc.verbose = True
#cc.output_dir = '..'

EPSILON = np.finfo(np.float32).eps
EULER_SAFE  = np.array([0,1,2,0],dtype='int32')
EULER_NEXT  = np.array([1,2,0,1],dtype='int32')
EULER_ORDER = np.array([0,8,16,4,12,20],dtype='int32')
MAYA_EA     = np.array([[0,1,2],[1,2,0],[2,0,1],[0,2,1],[1,0,2],[2,1,0]],dtype='int32')  # maya 2 euler angle
EA_MAYA     = np.array([[0,1,2],[2,0,1],[1,2,0],[0,2,1],[1,0,2],[2,1,0]],dtype='int32')  # euler angle 2 maya


#@cc.export('_getEulerOrder','UniTuple(int32,7)(int32)')
@njit(fastmath=True)
def _getEulerOrder(axis):
    o_ = EULER_ORDER[axis]
    f=o_&1; o_>>=1; s=o_&1; o_>>=1; n=o_&1; o_>>=1
    i=EULER_SAFE[o_&3]
    j=EULER_NEXT[i+n]
    k=EULER_NEXT[i+1-n]
    h=i
    if s:
        h=k

    return i,j,k,h,n,s,f



#@cc.export('_eulerToMatrix','float64[:,:,:](float64[:,:], int32[:])')
@njit(fastmath=True, parallel=True)
def _eulerToMatrix(euler, axes):
    """ Converts Maya euler angles to 4x4 matrices
    """
    matrix = np.empty((euler.shape[0],4,4))
    ea_ = np.empty(3)

    for ii in prange(euler.shape[0]):
        ea_[0], ea_[1], ea_[2] = euler[ii,0], euler[ii,1], euler[ii,2]
        ea_[0], ea_[1], ea_[2] = ea_[MAYA_EA[axes[ii],0]], ea_[MAYA_EA[axes[ii],1]], ea_[MAYA_EA[axes[ii],2]]

        i,j,k,h,n,s,f = _getEulerOrder(axes[ii])

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



#@cc.export('_eulerToQuaternion','float64[:,:](float64[:,:], int32[:])')
@njit(fastmath=True, parallel=True)
def _eulerToQuaternion(euler, axes):
    """ Converts list of Maya euler angles to quaternions qi,qj,qk,qw
    """
    quat = np.empty((euler.shape[0],4))
    ea_ = np.empty(3)

    for ii in prange(euler.shape[0]):
        ea_[0], ea_[1], ea_[2] = euler[ii,0], euler[ii,1], euler[ii,2]
        ea_[0], ea_[1], ea_[2] = ea_[MAYA_EA[axes[ii],0]], ea_[MAYA_EA[axes[ii],1]], ea_[MAYA_EA[axes[ii],2]]

        i,j,k,h,n,s,f = _getEulerOrder(axes[ii])

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



#@cc.export('_matrixIdentity','float64[:,:,:](int32)')
@njit(fastmath=True, parallel=True)
def _matrixIdentity(count):
    matrix = np.empty((count,4,4))
    for i in prange(count):
        matrix[i,0,0] = 1.; matrix[i,0,1] = 0.; matrix[i,0,2] = 0.; matrix[i,0,3] = 0.;
        matrix[i,1,0] = 0.; matrix[i,1,1] = 1.; matrix[i,1,2] = 0.; matrix[i,1,3] = 0.;
        matrix[i,2,0] = 0.; matrix[i,2,1] = 0.; matrix[i,2,2] = 1.; matrix[i,2,3] = 0.;
        matrix[i,3,0] = 0.; matrix[i,3,1] = 0.; matrix[i,3,2] = 0.; matrix[i,3,3] = 1.;

    return matrix



#@cc.export('_matrixToEuler','float64[:,:](float64[:,:,:], int32[:])')
@njit(fastmath=True, parallel=True)
def _matrixToEuler(matrix, axes):
    """ Converts list of 4x4 matrices to Maya euler angles.
    """
    euler = np.empty((matrix.shape[0],3))
    m_ = np.empty((3,3))

    for ii in prange(matrix.shape[0]):
        i,j,k,h,n,s,f = _getEulerOrder(axes[ii])

        # Normalize xyz axes in case of scale
        x,y,z = 0.,0.,0.
        for jj in range(3):
            x = x + matrix[ii,0,jj] ** 2
            y = y + matrix[ii,1,jj] ** 2
            z = z + matrix[ii,2,jj] ** 2

        x = x ** 0.5
        y = y ** 0.5
        z = z ** 0.5

        for jj in range(3):
            m_[jj,0] = matrix[ii,0,jj] / x
            m_[jj,1] = matrix[ii,1,jj] / y
            m_[jj,2] = matrix[ii,2,jj] / z

        if s:
            yy = (m_[i,j]**2 + m_[i,k]**2) ** 0.5
            euler[ii,1] = atan2( yy,  m_[i,i])

            if yy > EPSILON:
                euler[ii,0] = atan2( m_[i,j],  m_[i,k])
                euler[ii,2] = atan2( m_[j,i], -m_[k,i])
            else:
                euler[ii,0] = atan2(-m_[j,k],  m_[j,j])
                euler[ii,2] = 0.

        else:
            yy = (m_[i,i]**2 + m_[j,i]**2) ** 0.5

            euler[ii,1] = atan2(-m_[k,i],  yy)

            if yy > EPSILON:
                euler[ii,0] = atan2( m_[k,j],  m_[k,k])
                euler[ii,2] = atan2( m_[j,i],  m_[i,i])
            else:
                euler[ii,0] = atan2(-m_[j,k],  m_[j,j])
                euler[ii,2] = 0.

        if n:
            euler[ii,0], euler[ii,1], euler[ii,2] = -euler[ii,0], -euler[ii,1], -euler[ii,2]

        if f:
            euler[ii,0], euler[ii,2] = euler[ii,2], euler[ii,0]


        # From euler angle to maya
        euler[ii,0], euler[ii,1], euler[ii,2]  = euler[ii,EA_MAYA[axes[ii],0]], euler[ii,EA_MAYA[axes[ii],1]], euler[ii,EA_MAYA[axes[ii],2]]


    return euler



#@cc.export('_matrixToQuaternion','float64[:,:](float64[:,:,:])')
@njit(fastmath=True, parallel=True)
def _matrixToQuaternion(matrix):
    """ Converts 4x4 matrix to quaternions qi,qj,qk,qw
    """
    quat = np.empty((matrix.shape[0],4))

    for i in prange(matrix.shape[0]):
        trace = matrix[i,0,0] + matrix[i,1,1] + matrix[i,2,2]

        if trace > 0.:
            s = 0.5 / (trace + 1.0) ** 0.5
            quat[i,0] = (matrix[i,1,2] - matrix[i,2,1] ) * s
            quat[i,1] = (matrix[i,2,0] - matrix[i,0,2] ) * s
            quat[i,2] = (matrix[i,0,1] - matrix[i,1,0] ) * s
            quat[i,3] = 0.25 / s

        elif matrix[i,0,0] > matrix[i,1,1] and matrix[i,0,0] > matrix[i,2,2]:
            s = 2.0 * ( 1.0 + matrix[i,0,0] - matrix[i,1,1] - matrix[i,2,2]) ** 0.5
            quat[i,0] = 0.25 * s
            quat[i,1] = (matrix[i,1,0] + matrix[i,0,1] ) / s
            quat[i,2] = (matrix[i,2,0] + matrix[i,0,2] ) / s
            quat[i,3] = (matrix[i,1,2] - matrix[i,2,1] ) / s

        elif matrix[i,1,1] > matrix[i,2,2]:
            s = 2.0 * ( 1.0 + matrix[i,1,1] - matrix[i,0,0] - matrix[i,2,2]) ** 0.5
            quat[i,0] = (matrix[i,1,0] + matrix[i,0,1] ) / s
            quat[i,1] = 0.25 * s
            quat[i,2] = (matrix[i,2,1] + matrix[i,1,2] ) / s
            quat[i,3] = (matrix[i,2,0] - matrix[i,0,2] ) / s

        else:
            s = 2.0 * ( 1.0 + matrix[i,2,2] - matrix[i,0,0] - matrix[i,1,1]) ** 0.5
            quat[i,0] = (matrix[i,2,0] + matrix[i,0,2] ) / s
            quat[i,1] = (matrix[i,2,1] + matrix[i,1,2] ) / s
            quat[i,2] = 0.25 * s
            quat[i,3] = (matrix[i,0,1] - matrix[i,1,0] ) / s

    return quat



#@cc.export('_matrixInverse','float64[:,:,:](float64[:,:,:])')
@njit(fastmath=True, parallel=True)
def _matrixInverse(matrix):
    """ Assumes matrix is 4x4 orthogonal
    """

    # Init inverse Matrix
    m_ = np.empty(matrix.shape)

    # For every matrix
    for i in prange(matrix.shape[0]):

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



#@cc.export('_matrixNormalize','float64[:,:,:](float64[:,:,:])')
@njit(fastmath=True, parallel=True)
def _matrixNormalize(matrix):
    """ Normalizes the rotation component of a transform matrix
    """
    m_ = np.empty(matrix.shape)

    # For every matrix
    for i in prange(matrix.shape[0]):
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





#@cc.export('_matrixMultiply','float64[:,:,:](float64[:,:,:],float64[:,:,:])')
@njit(fastmath=True, parallel=True)
def _matrixMultiply(matrix0,matrix1):

    m = np.empty((matrix0.shape[0],4,4))

    for i in prange(matrix0.shape[0]):
        m[i,0,0] = matrix0[i,0,0] * matrix1[i,0,0] + matrix0[i,0,1] * matrix1[i,1,0] + matrix0[i,0,2] * matrix1[i,2,0] + matrix0[i,0,3] * matrix1[i,3,0]
        m[i,0,1] = matrix0[i,0,0] * matrix1[i,0,1] + matrix0[i,0,1] * matrix1[i,1,1] + matrix0[i,0,2] * matrix1[i,2,1] + matrix0[i,0,3] * matrix1[i,3,1]
        m[i,0,2] = matrix0[i,0,0] * matrix1[i,0,2] + matrix0[i,0,1] * matrix1[i,1,2] + matrix0[i,0,2] * matrix1[i,2,2] + matrix0[i,0,3] * matrix1[i,3,2]
        m[i,0,3] = matrix0[i,0,0] * matrix1[i,0,3] + matrix0[i,0,1] * matrix1[i,1,3] + matrix0[i,0,2] * matrix1[i,2,3] + matrix0[i,0,3] * matrix1[i,3,3]

        m[i,1,0] = matrix0[i,1,0] * matrix1[i,0,0] + matrix0[i,1,1] * matrix1[i,1,0] + matrix0[i,1,2] * matrix1[i,2,0] + matrix0[i,1,3] * matrix1[i,3,0]
        m[i,1,1] = matrix0[i,1,0] * matrix1[i,0,1] + matrix0[i,1,1] * matrix1[i,1,1] + matrix0[i,1,2] * matrix1[i,2,1] + matrix0[i,1,3] * matrix1[i,3,1]
        m[i,1,2] = matrix0[i,1,0] * matrix1[i,0,2] + matrix0[i,1,1] * matrix1[i,1,2] + matrix0[i,1,2] * matrix1[i,2,2] + matrix0[i,1,3] * matrix1[i,3,2]
        m[i,1,3] = matrix0[i,1,0] * matrix1[i,0,3] + matrix0[i,1,1] * matrix1[i,1,3] + matrix0[i,1,2] * matrix1[i,2,3] + matrix0[i,1,3] * matrix1[i,3,3]

        m[i,2,0] = matrix0[i,2,0] * matrix1[i,0,0] + matrix0[i,2,1] * matrix1[i,1,0] + matrix0[i,2,2] * matrix1[i,2,0] + matrix0[i,2,3] * matrix1[i,3,0]
        m[i,2,1] = matrix0[i,2,0] * matrix1[i,0,1] + matrix0[i,2,1] * matrix1[i,1,1] + matrix0[i,2,2] * matrix1[i,2,1] + matrix0[i,2,3] * matrix1[i,3,1]
        m[i,2,2] = matrix0[i,2,0] * matrix1[i,0,2] + matrix0[i,2,1] * matrix1[i,1,2] + matrix0[i,2,2] * matrix1[i,2,2] + matrix0[i,2,3] * matrix1[i,3,2]
        m[i,2,3] = matrix0[i,2,0] * matrix1[i,0,3] + matrix0[i,2,1] * matrix1[i,1,3] + matrix0[i,2,2] * matrix1[i,2,3] + matrix0[i,2,3] * matrix1[i,3,3]

        m[i,3,0] = matrix0[i,3,0] * matrix1[i,0,0] + matrix0[i,3,1] * matrix1[i,1,0] + matrix0[i,3,2] * matrix1[i,2,0] + matrix0[i,3,3] * matrix1[i,3,0]
        m[i,3,1] = matrix0[i,3,0] * matrix1[i,0,1] + matrix0[i,3,1] * matrix1[i,1,1] + matrix0[i,3,2] * matrix1[i,2,1] + matrix0[i,3,3] * matrix1[i,3,1]
        m[i,3,2] = matrix0[i,3,0] * matrix1[i,0,2] + matrix0[i,3,1] * matrix1[i,1,2] + matrix0[i,3,2] * matrix1[i,2,2] + matrix0[i,3,3] * matrix1[i,3,2]
        m[i,3,3] = matrix0[i,3,0] * matrix1[i,0,3] + matrix0[i,3,1] * matrix1[i,1,3] + matrix0[i,3,2] * matrix1[i,2,3] + matrix0[i,3,3] * matrix1[i,3,3]


    return m



#@cc.export('_matrixPointMultiply','float64[:,:](float64[:,:],float64[:,:,:])')
@njit(fastmath=True, parallel=True)
def _matrixPointMultiply(point,matrix):

    p = np.empty((point.shape[0],3))

    for i in prange(point.shape[0]):
        p[i,0] = (matrix[i,0,0] * point[i,0]) + (matrix[i,1,0] * point[i,1]) + (matrix[i,2,0] * point[i,2]) + matrix[i,3,0]
        p[i,1] = (matrix[i,0,1] * point[i,0]) + (matrix[i,1,1] * point[i,1]) + (matrix[i,2,1] * point[i,2]) + matrix[i,3,1]
        p[i,2] = (matrix[i,0,2] * point[i,0]) + (matrix[i,1,2] * point[i,1]) + (matrix[i,2,2] * point[i,2]) + matrix[i,3,2]

    return p




#@cc.export('_vectorSlerp','float64[:,:](float64[:,:],float64[:,:],float64[:])')
@njit(fastmath=True, parallel=True)
def _vectorArc(vector0, vector1):
    
    angle = np.empty(vector0.shape[0])
    vector0_ = np.empty(3)
    vector1_ = np.empty(3)

    for i in prange(vector0.shape[0]):
        m = (vector0[i,0]**2 + vector0[i,1]**2 + vector0[i,2]**2) ** 0.5
        if m > 0.:
            vector0_[0] = vector0[i,0] / m
            vector0_[1] = vector0[i,1] / m
            vector0_[2] = vector0[i,2] / m

        m = (vector1[i,0]**2 + vector1[i,1]**2 + vector1[i,2]**2) ** 0.5
        if m > 0.:
            vector1_[0] = vector1[i,0] / m
            vector1_[1] = vector1[i,1] / m
            vector1_[2] = vector1[i,2] / m
            
        dot = (vector0_[0]*vector1_[0]) + (vector0_[1]*vector1_[1]) + (vector0_[2]*vector1_[2])
        if dot > 1.0:
            dot = 1.0
        elif dot < -1.0:
            dot = -1.0
            
        angle[i] = acos(dot)

    return angle


#@cc.export('_vectorSlerp','float64[:,:](float64[:,:],float64[:,:],float64[:])')
@njit(fastmath=True, parallel=True)
def _vectorSlerp(vector0, vector1, weight):
    np.empty(vector0.shape[0]) # TODO: WTF? parallel njit fails nopython pipeline without this line???
    v = np.empty((vector0.shape[0],3))

    for i in prange(vector0.shape[0]):
        X0, Y0, Z0 = 0., 0., 0.
        X1, Y1, Z1 = 0., 0., 0.
        
        m = (vector0[i,0]**2.0 + vector0[i,1]**2.0 + vector0[i,2]**2) ** 0.5
        if m > 0.:
            X0 = vector0[i,0] / m
            Y0 = vector0[i,1] / m
            Z0 = vector0[i,2] / m

        m = (vector1[i,0]**2.0 + vector1[i,1]**2.0 + vector1[i,2]**2.0) ** 0.5
        if m > 0.:
            X1 = vector1[i,0] / m
            Y1 = vector1[i,1] / m
            Z1 = vector1[i,2] / m

        dot = (X0*X1) + (Y0*Y1) + (Z0*Z1)
        angle  = acos(dot)
        sangle = sin(angle)

        if sangle > 0.:
            w0 = sin((1.0-weight[i]) * angle)
            w1 = sin(weight[i] * angle)

            v[i,0] = (vector0[i,0] * w0 + vector1[i,0] * w1) / sangle
            v[i,1] = (vector0[i,1] * w0 + vector1[i,1] * w1) / sangle
            v[i,2] = (vector0[i,2] * w0 + vector1[i,2] * w1) / sangle
        else:
            v[i,0] = vector0[i,0]
            v[i,1] = vector0[i,1]
            v[i,2] = vector0[i,2]
    
    return v


#@cc.export('_vectorLerp','float64[:,:](float64[:,:],float64[:,:],float64[:])')
@njit(fastmath=True, parallel=True)
def _vectorLerp(vector0, vector1, weight):

    v = np.empty((vector0.shape[0],vector0.shape[1]))
    
    for i in prange(vector0.shape[0]):
        for j in range(vector0.shape[1]):
            v[i,j] = vector0[i,j] + weight[i] * (vector1[i,j] - vector0[i,j])

    return v




#@cc.export('_vectorToMatrix','float64[:,:,:](float64[:,:], float64[:,:], int32[:], int32[:])')
@njit(fastmath=True, parallel=True)
def _vectorToMatrix(vector0, vector1, aim_axis, up_axis):

    vector0_ = np.empty(3)
    vector1_ = np.empty(3)
    matrix   = np.empty((vector0.shape[0], 4, 4))

    for i in prange(vector0.shape[0]):
        ii = aim_axis[i]
        jj = up_axis[i]
        kk = (min(ii,jj)-max(ii,jj)+min(ii,jj)) % 3

        flip = 0
        if ii == 0 and jj == 2:
            flip = 1
        elif ii == 1 and jj == 0:
            flip = 1
        elif ii == 2 and jj == 1:
            flip = 1


        for j in range(3):
            vector0_[j] = vector0[i,j]
            vector1_[j] = vector1[i,j]

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



#@cc.export('_vectorCross','float64[:,:](float64[:,:], float64[:,:])')
@njit('float64[:,:](float64[:,:], float64[:,:])', fastmath=True, parallel=True)
def _vectorCross(vector0, vector1):
    vector = np.zeros((vector0.shape[0],3), dtype=np.float64)

    for i in prange(vector0.shape[0]):
        vector[i,0] = vector0[i, 1] * vector1[i, 2] - vector0[i, 2] * vector1[i, 1]
        vector[i,1] = vector0[i, 2] * vector1[i, 0] - vector0[i, 0] * vector1[i, 2]
        vector[i,2] = vector0[i, 0] * vector1[i, 1] - vector0[i, 1] * vector1[i, 0]

    return vector


#@cc.export('_vectorDot','float64[:](float64[:,:], float64[:,:])')
@njit(fastmath=True, parallel=True)
def _vectorDot(vector0, vector1):
    dot = np.empty(vector0.shape[0])

    for i in prange(vector0.shape[0]):
        dot[i] = 0
        for j in range(vector0.shape[1]):
            dot[i] += vector0[i, j] * vector1[i, j]

    return dot



#@cc.export('_vectorMagnitude','float64[:](float64[:,:])')
@njit(fastmath=True, parallel=True)
def _vectorMagnitude(vector):
    mag = np.empty(vector.shape[0])
    for i in prange(vector.shape[0]):
        mag[i] = 0.
        for j in range(vector.shape[1]):
            mag[i] = mag[i] + vector[i,j]**2

        mag[i] = mag[i] ** 0.5

    return mag



#@cc.export('_vectorNormalize','float64[:,:](float64[:,:])')
@njit(fastmath=True, parallel=True)
def _vectorNormalize(vector):
    vector_ = np.empty(vector.shape)

    # For every vector
    for i in prange(vector.shape[0]):

        # calc the magnitude
        mag = 0
        for j in range(vector.shape[1]):
            mag += vector[i,j]**2

        mag = mag ** 0.5

        # normalize
        for j in range(vector.shape[1]):
            vector_[i,j] = vector[i,j]/mag

    return vector_





#@cc.export('_quaternionSlerp','float64[:,:](float64[:,:],float64[:,:],float64[:])')
@njit(fastmath=True, parallel=True)
def _quaternionSlerp(quat0, quat1, weight):
    """ Calculates spherical interpolation between two lists quaternions
    """
    quat = np.empty((quat0.shape[0],4))

    for i in prange(quat0.shape[0]):
        
        # If quat0=quat1 or quat0=-quat1 then theta = 0 and we can return quat0
        cosHalfTheta = quat0[i,3] * quat1[i,3] + quat0[i,0] * quat1[i,0] + quat0[i,1] * quat1[i,1] + quat0[i,2] * quat1[i,2]
        
        if abs(cosHalfTheta) >= 1.0:
            quat[i,0], quat[i,1], quat[i,2], quat[i,3] = quat0[i,0], quat0[i,1], quat0[i,2], quat0[i,3]

        else:
            halfTheta = acos(cosHalfTheta)
            sinHalfTheta  = (1.0 - cosHalfTheta*cosHalfTheta) ** 0.5

            # If theta = 180 degrees then result is not fully defined
            # we could rotate around any axis normal to quat0 or quat1
            if abs(sinHalfTheta) < EPSILON:
                quat[i,0] = (quat0[i,0] * 0.5 + quat1[i,0] * 0.5)
                quat[i,1] = (quat0[i,1] * 0.5 + quat1[i,1] * 0.5)
                quat[i,2] = (quat0[i,2] * 0.5 + quat1[i,2] * 0.5)
                quat[i,3] = (quat0[i,3] * 0.5 + quat1[i,3] * 0.5)

            else:
                ratioA = sin((1 - weight[i]) * halfTheta) / sinHalfTheta
                ratioB = sin(weight[i] * halfTheta) / sinHalfTheta

                quat[i,0] = (quat0[i,0] * ratioA + quat1[i,0] * ratioB)
                quat[i,1] = (quat0[i,1] * ratioA + quat1[i,1] * ratioB)
                quat[i,2] = (quat0[i,2] * ratioA + quat1[i,2] * ratioB)
                quat[i,3] = (quat0[i,3] * ratioA + quat1[i,3] * ratioB)

    return quat




#@cc.export('_quaternionMultiply','float64[:,:](float64[:,:], float64[:,:])')
@njit(fastmath=True, parallel=True)
def _quaternionMultiply(quat0, quat1):
    """ Multiplies 2 lists of quaternions
    """
    quat = np.empty((quat0.shape[0],4))

    for i in prange(quat0.shape[0]):
        quat[i,0] =  quat0[i,0] * quat1[i,3] + quat0[i,1] * quat1[i,2] - quat0[i,2] * quat1[i,1] + quat0[i,3] * quat1[i,0]
        quat[i,1] = -quat0[i,0] * quat1[i,2] + quat0[i,1] * quat1[i,3] + quat0[i,2] * quat1[i,0] + quat0[i,3] * quat1[i,1]
        quat[i,2] =  quat0[i,0] * quat1[i,1] - quat0[i,1] * quat1[i,0] + quat0[i,2] * quat1[i,3] + quat0[i,3] * quat1[i,2]
        quat[i,3] = -quat0[i,0] * quat1[i,0] - quat0[i,1] * quat1[i,1] - quat0[i,2] * quat1[i,2] + quat0[i,3] * quat1[i,3]

    return quat



#@cc.export('_quaternionAdd','float64[:,:](float64[:,:], float64[:,:])')
@njit(fastmath=True, parallel=True)
def _quaternionAdd(quat0, quat1):
    """ Multiplies 2 lists of quaternions
    """
    quat = np.empty((quat0.shape[0],4))

    for i in prange(quat0.shape[0]):
        quat[i,0] = quat0[i,0] + quat1[i,0]
        quat[i,1] = quat0[i,1] + quat1[i,1]
        quat[i,2] = quat0[i,2] + quat1[i,2]
        quat[i,3] = quat0[i,3] + quat1[i,3]

    return quat



#@cc.export('_quaternionSub','float64[:,:](float64[:,:], float64[:,:])')
@njit(fastmath=True, parallel=True)
def _quaternionSub(quat0, quat1):
    """ Multiplies 2 lists of quaternions
    """
    quat = np.empty((quat0.shape[0],4))

    for i in prange(quat0.shape[0]):
        quat[i,0] = quat0[i,0] - quat1[i,0]
        quat[i,1] = quat0[i,1] - quat1[i,1]
        quat[i,2] = quat0[i,2] - quat1[i,2]
        quat[i,3] = quat0[i,3] - quat1[i,3]

    return quat




#@cc.export('_quaternionToMatrix','float64[:,:,:](float64[:,:])')
@njit(fastmath=True, parallel=True)
def _quaternionToMatrix(quat):

    # Init Matrix
    matrix = np.empty((quat.shape[0],4,4))

    # For every quaternion
    for i in prange(quat.shape[0]):
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



#@cc.export('_quaternionConjugate','float64[:,:](float64[:,:])')
@njit(fastmath=True, parallel=True)
def _quaternionConjugate(quat):
    """ Conjugates a list of quaternions
    """
    quat_ = np.empty((quat.shape[0],4))

    # For every quaternion
    for i in prange(quat.shape[0]):
        quat_[i,0] = -quat[i,0]
        quat_[i,1] = -quat[i,1]
        quat_[i,2] = -quat[i,2]
        quat_[i,3] =  quat[i,3]

    return quat_


#@cc.export('_quaternionInverse','float64[:,:](float64[:,:])')
@njit(fastmath=True, parallel=True)
def _quaternionInverse(quat):
    """ Inverses a list of quaternions
    """
    quat_ = np.empty((quat.shape[0],4))

    # For every quaternion
    for i in prange(quat.shape[0]):
        mag = quat[i,0]**2 + quat[i,1]**2 + quat[i,2]**2 + quat[i,3]**2

        quat_[i,0] = -quat[i,0]/mag
        quat_[i,1] = -quat[i,1]/mag
        quat_[i,2] = -quat[i,2]/mag
        quat_[i,3] =  quat[i,3]/mag

    return quat_



#@cc.export('_quaternionNegate','float64[:,:](float64[:,:])')
@njit(fastmath=True, parallel=True)
def _quaternionNegate(quat):
    """ Negates a list of quaternions
    """
    quat_ = np.empty((quat.shape[0],4))

    # For every quaternion
    for i in prange(quat.shape[0]):
        quat_[i,0] = -quat[i,0]
        quat_[i,1] = -quat[i,1]
        quat_[i,2] = -quat[i,2]
        quat_[i,3] = -quat[i,3]

    return quat_




#@cc.export('_axisAngleToQuaternion','float64[:,:](float64[:,:], float64[:])')
@njit(fastmath=True, parallel=True)
def _axisAngleToQuaternion(axis,angle):
    """ Computes a list of quaternions qi,qj,qk,qw from lists of axes and angles
    """
    quat  = np.empty((axis.shape[0],4))
    axis_ = np.empty(3)

    for i in prange(axis.shape[0]):
        mag = (axis[i,0] ** 2 + axis[i,1] ** 2 + axis[i,2] ** 2) ** 0.5
        axis_[0] = axis[i,0]/mag
        axis_[1] = axis[i,1]/mag
        axis_[2] = axis[i,2]/mag


        s = sin(angle[i]/2)
        quat[i,0] = axis_[0] * s
        quat[i,1] = axis_[1] * s
        quat[i,2] = axis_[2] * s
        quat[i,3] = cos(angle[i]/2)

    return quat



#@cc.export('_axisAngleToMatrix','float64[:,:,:](float64[:,:], float64[:])')
@njit(fastmath=True, parallel=True)
def _axisAngleToMatrix(axis,angle):
    """ Computes a list of orthogonal 4x4 matrices from lists of axes and angles
    """
    
    matrix = np.empty((axis.shape[0],4,4))

    for i in prange(axis.shape[0]):
        sin_ = sin(angle[i])
        cos_ = cos(angle[i])
        inv_cos = 1-cos_

        mag = (axis[i,0] ** 2 + axis[i,1] ** 2 + axis[i,2] ** 2) ** 0.5
        u = axis[i,0]/mag
        v = axis[i,1]/mag
        w = axis[i,2]/mag
        uv = u*v
        uw = u*w
        vw = v*w
        usin = u*sin_
        vsin = v*sin_
        wsin = w*sin_
        u2 = u**2
        v2 = v**2
        w2 = w**2

        matrix[i,0,0] = u2 + ((v2 + w2) * cos_)
        matrix[i,0,1] = uv*inv_cos + (wsin)
        matrix[i,0,2] = uw*inv_cos - (vsin)
        matrix[i,1,0] = uv*inv_cos - (wsin)
        matrix[i,1,1] = v2 + ((u2 + w2) * cos_)
        matrix[i,1,2] = vw*inv_cos + (usin)
        matrix[i,2,0] = uw*inv_cos + (vsin)
        matrix[i,2,1] = vw*inv_cos - (usin)
        matrix[i,2,2] = w2 + ((u2 + v2) * cos_)
        matrix[i,0,3] = 0.
        matrix[i,1,3] = 0.
        matrix[i,2,3] = 0.
        matrix[i,3,0] = 0.
        matrix[i,3,1] = 0.
        matrix[i,3,2] = 0.
        matrix[i,3,3] = 1.

    return matrix




#@cc.export('_vectorArcToQuaternion','float64[:,:](float64[:,:], float64[:,:])')
@njit(fastmath=True, parallel=True)
def _vectorArcToQuaternion(vector0,vector1):
    """ Computes a list of quaternions qi,qj,qk,qw representing the arc between lists of vectors
    """

    quat = np.empty((vector0.shape[0],4))

    v0 = np.empty(3) # normalized vector0
    v1 = np.empty(3) # normalized vector1
    h  = np.empty(3) # normalized vector0 + vector1

    for i in prange(vector0.shape[0]):
        # normalized vector0
        mag = (vector0[i,0]**2 + vector0[i,1]**2 + vector0[i,2]**2) ** 0.5
        v0[0] = vector0[i,0]/mag
        v0[1] = vector0[i,1]/mag
        v0[2] = vector0[i,2]/mag

        # normalized vector1
        mag = (vector1[i,0]**2 + vector1[i,1]**2 + vector1[i,2]**2) ** 0.5
        v1[0] = vector1[i,0]/mag
        v1[1] = vector1[i,1]/mag
        v1[2] = vector1[i,2]/mag

        # normalized vector0 + vector1
        h[0] = v0[0] + v1[0]
        h[1] = v0[1] + v1[1]
        h[2] = v0[2] + v1[2]

        mag = (h[0]**2 + h[1]**2 + h[2]**2) ** 0.5

        h[0] = h[0] / mag
        h[1] = h[1] / mag
        h[2] = h[2] / mag

        # Generate arc-vector quaternion
        quat[i,0] = v0[1]*h[2] - v0[2]*h[1]
        quat[i,1] = v0[2]*h[0] - v0[0]*h[2]
        quat[i,2] = v0[0]*h[1] - v0[1]*h[0]
        quat[i,3] = v0[0]*h[0] + v0[1]*h[1] + v0[2]*h[2]


    return quat



#if __name__ == "__main__":
    #cc.compile()
