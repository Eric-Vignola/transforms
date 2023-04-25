import numpy as np
from numpy.core.umath_tests import inner1d

from ._transforms import _axisAngleToMatrix, _axisAngleToQuaternion, _eulerToMatrix, _eulerToQuaternion
from ._transforms import _matrixIdentity, _matrixInverse, _matrixMultiply, _matrixNormalize
from ._transforms import _matrixPointMultiply, _matrixToEuler, _matrixToQuaternion, _quaternionAdd
from ._transforms import _quaternionConjugate, _quaternionInverse, _quaternionMultiply, _quaternionNegate
from ._transforms import _quaternionSlerp, _quaternionSub, _quaternionToMatrix, _vectorArcToQuaternion
from ._transforms import _vectorCross, _vectorDot, _vectorLerp, _vectorMagnitude, _vectorNormalize
from ._transforms import _vectorSlerp, _vectorToMatrix


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




#------------------------------------------- AXIS ANGLE MATH -----------------------------------------------#


def angle_to_quaternion(axis, angle=0.):
    """
    angle_to_quaternion(axis, angle=0.)
    
        Computes a list of quaternions qi,qj,qk,qw from lists of axes and angles

        Parameters
        ----------
        axis  : *[float, float, float]* or *[[float, float, float],...]*
                a single, or list vector axis.
            
        angle : *float* or *[float, ...]*
                a single, or list of angles to rotate about the given axis.
            
            
        Returns
        -------
        euler: np.array(n,3)
               a list of euler angles
            

        See Also
        --------
        axisAngleToMatrix     : Converts axis angles to matrices.
        axisAngleToQuaternion : Converts axis angles to quaternions qi,qj,qk,qw.
        

        Examples
        --------
        >>> axis = vector.random(2)                       # make 2 random axis vectors
        >>> angles = vector.random(2)[:,0]                # make 2 random angles
        >>> print (repr(axisAngleToEuler(axis,angles))    # from axis angles from 2 lists
        >>> print (repr(axisAngleToEuler(axis,angles[0])) # from axis angles from axis and first angle

    """            
    
    axis  = _setDimension(axis,2)
    angle = _setDimension(angle,1)
    axis, angle = _matchDepth(axis, angle)
    
    return _axisAngleToQuaternion(axis,angle)



def angle_to_matrix(axis, angle=0.):
    """ Computes a list of quaternions qi,qj,qk,qw from lists of axes and angles
    """
    axis  = _setDimension(axis,2)
    angle = _setDimension(angle,1)
    axis, angle = _matchDepth(axis, angle)
    
    return _axisAngleToMatrix(axis,angle)



def angle_to_euler(axis, angle=0., axes=XYZ):
    """
    angle_to_euler(euler,axes=XYZ)
    
        Computes a list of quaternions qi,qj,qk,qw from lists of axes and angles.

        Parameters
        ----------
        axis  : *[float, float, float]* or *[[float, float, float],...]*
                a single, or list vector axis.
            
        angle : *float* or *[float, ...]*
                a single, or list of angles to rotate about the given axis.
            
        axes  : *XYZ, YZX, ZXY, XZY, YXZ, ZYX*, *int* or *[int,...]*
                corresponding rotate orders. default = XYZ
            
            
        Returns
        -------
        euler: np.array(n,3)
               a list of euler angles
            

        See Also
        --------
        angle_to_matrix     : Converts axis angles to matrices.
        angle_to_quaternion : Converts axis angles to quaternions qi,qj,qk,qw.
        

        Examples
        --------
        >>> axis = randomVector(2)                 # make 2 random axis vectors
        >>> angles = randomEuler(2)[:,0]           # make 2 random angles
        >>> print (angle_to_euler(axis,angles))    # from axis angles from 2 lists
        >>> print (angle_to_euler(axis,angles[0])) # from axis angles from axis and first angle

    """        
    
    axis  = _setDimension(axis,2)
    angle = _setDimension(angle,1)    
    axes  = _setDimension(axes,1,dtype=np.int32)
    axis, angle,axes = _matchDepth(axis, angle, axes)
    
    M = _axisAngleToMatrix(axis, angle)
    return _matrixToEuler(M, axes)

