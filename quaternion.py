import numpy as np
from numpy.core.umath_tests import inner1d

from ._transforms import _axisAngleToMatrix, _axisAngleToQuaternion, _eulerToMatrix, _eulerToQuaternion
from ._transforms import _matrixIdentity, _matrixInverse, _matrixMultiply, _matrixNormalize
from ._transforms import _matrixPointMultiply, _matrixToEuler, _matrixToQuaternion, _quaternionAdd
from ._transforms import _quaternionConjugate, _quaternionInverse, _quaternionMultiply, _quaternionNegate
from ._transforms import _quaternionSlerp, _quaternionSub, _quaternionToMatrix, _vectorArcToQuaternion
from ._transforms import _vectorCross, _vectorDot, _vectorLerp, _vectorMagnitude, _vectorNormalize
from ._transforms import _vectorSlerp, _vectorToMatrix

from .euler import random as randomEuler


# axes as mapped by Maya's rotate order indices
XYZ = 0
YZX = 1
ZXY = 2
XZY = 3
YXZ = 4
ZYX = 5

# XYZ axes indices
X = 0
Y = 1
Z = 2


#----------------------------------------------- UTILS -----------------------------------------------#

def _setDimension(data, ndim=1, dtype=np.float64, reshape_matrix=False):
    """ Sets input data to expected dimension
    """
    data = np.asarray(data, dtype=dtype)
    
    while data.ndim < ndim:
        data = data[np.newaxis]
        
    # For when matrices are given as lists of 16 floats
    if reshape_matrix:
        if data.shape[-1] == 16:
            data = data.reshape(-1,4,4)

    return data


def _matchDepth(*data):
    """ Sets given data to the highest dimention.
        It is assumed all entries are already arrays
    """
    count   = [len(d) for d in data]
    highest = max(count)
    matched = list(data)
    
    for i in range(len(count)):
        if count[i] > 0:
            matched[i] = np.concatenate((data[i],) + (np.repeat([data[i][-1]],highest-count[i],axis=0),))
        
    return matched




#----------------------------------------------- QUATERNION MATH -----------------------------------------------#


def slerp(quat0, quat1, weight=0.5):
    """
    slerp(quat0, quat1, weight=0.5)
    
        Performs a spherical interpolation between two lists of quaternions qi,qj,qk,qw.

        Parameters
        ----------
        quat0 : *[float, float, float, float]* or *[[float, float, float, float],...]*
                a single, or list of quaternions which correspond to weight=0
            
        quat1 : *[float, float, float, float]* or *[[float, float, float, float],...]*
                a single, or list of quaternions which correspond to weight=1
            
        weight : *float* or *[float,...]*
                weight values to interpolate between quat0 and quat1. default = 0.5
            
            
        Returns
        -------
        quaternions : np.array(n,4)
                      a list of interpolated quaternions qi,qj,qk,qw
            

        See Also
        --------
        eulerSlerp  : Performs a spherical interpolation between two lists of euler angles.
        matrixSlerp : Performs a spherical interpolation between two lists of 4x4 matrices.
        vectorSlerp : Performs a spherical interpolation between two lists of vectors.
        

        Examples
        --------
        >>> quat0 = random(100)          # init quat0
        >>> quat1 = random(100)          # init quat1
        >>> print (slerp(quat0,quat1)    # get the halfway point between the two lists
        >>> print (slerp(quat0,quat1[0]) # get the halfway point between the all items of quat0 and the first item of quat1

    """        
    quat0  = _setDimension(quat0,2)
    quat1  = _setDimension(quat1,2)
    weight = _setDimension(weight,1)
    
    quat0, quat1, weight = _matchDepth(quat0, quat1, weight)

    return _quaternionSlerp(quat0,quat1,weight)



def dot(quat0, quat1):
    """ Calculates dot product between two quaternions
    """
    quat0  = _setDimension(quat0,2)
    quat1  = _setDimension(quat1,2)
    
    quat0, quat1 = _matchDepth(quat0, quat1)
    
    return _vectorDot(quat0, quat1)




def conjugate(quat):
    """ Calculates dot product between two quaternions
    """
    quat  = _setDimension(quat,2)
    
    return _quaternionConjugate(quat)



def inverse(quat):
    """ Calculates dot product between two quaternions
    """
    quat  = _setDimension(quat,2)

    return _quaternionInverse(quat)



def negate(quat):
    """ Calculates dot product between two quaternions
    """
    quat  = _setDimension(quat,2)

    return _quaternionNegate(quat)



def multiply(quat0, quat1):
    """ Multiplies two quaternions
    """
    quat0  = _setDimension(quat0,2)
    quat1  = _setDimension(quat1,2)
    quat0, quat1 = _matchDepth(quat0, quat1)
    
    return _quaternionMultiply(quat0,quat1)



def add(quat0, quat1):
    """ Adds two quaternions
    """
    quat0  = _setDimension(quat0,2)
    quat1  = _setDimension(quat1,2)
    quat0, quat1 = _matchDepth(quat0, quat1)
    
    return _quaternionAdd(quat0,quat1)



def sub(quat0, quat1):
    """ Subtracts two quaternions
    """
    quat0  = _setDimension(quat0,2)
    quat1  = _setDimension(quat1,2)
    quat0, quat1 = _matchDepth(quat0, quat1)
    
    return _quaternionSub(quat0,quat1)



def to_matrix(quat):
    """ Converts list of quaternions qi,qj,qk,qw to 4x4 matrices
    
        >>> q = random(2)           # make 2 random quaternions
        >>> print (to_matrix(q[0])) # from 1 quaternion make matrix
        >>> print (to_matrix(q))    # from 2 quaternions make matrices
    """
    quat = _setDimension(quat,2)
    return _quaternionToMatrix(quat)
    
    
    
def normalize(quat):
    """ Normalizes a quaternion
    """
    quat = _setDimension(quat,2)
    return _vectorNormalize(quat)
 
    
def to_euler(quat, axes=XYZ):
    """ Converts quaternions qi,qj,qk,qw to euler angles
    
        >>> q = randomEuler(2)               # make 2 random quaternions
        >>> print (quaternionToMatrix(q[0])) # from 1 quaternion make 1 matrix
        >>> print (quaternionToMatrix(q))    # from 2 quaternions make 2 matrices
    """
    quat = _setDimension(quat,2)
    axes = _setDimension(axes,1,dtype=np.int32)
    quat, axes = _matchDepth(quat, axes)
    
    return _matrixToEuler(_quaternionToMatrix(quat), axes)


def random(n, seed=None):
    """ Computes a list of random quaternions qi,qj,qk,qw
    """
    eu = randomEuler(n=n, seed=seed)
    return _eulerToQuaternion(eu, np.array([0]))


