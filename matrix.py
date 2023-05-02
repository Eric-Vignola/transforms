import numpy as np

from ._transforms import _eulerToMatrix, _matrixIdentity, _matrixInverse, _matrixMultiply, _matrixNormalize
from ._transforms import _matrixPointMultiply, _matrixToEuler, _matrixToQuaternion, _quaternionSlerp
from ._transforms import _quaternionToMatrix, _vectorLerp

from ._utils import _setDimension, _matchDepth

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


#----------------------------------------------- MATRIX MATH -----------------------------------------------#


def identity(count):
    """ Creates 4x4 identity matrices
    """
    return _matrixIdentity(count)


def to_euler(matrix, axes=XYZ):
    """ Converts an euler angle to a 4x4 transform matrix
    
        >>> m = random(2)             # make 2 random matrices
        >>> print (to_euler(m[0],0))        # from 1 matrix   makes 1 euler angle  with xyz rotate order
        >>> print (to_euler(m,0))           # from 2 matrices makes 2 euler angles with xyz rotate order
        >>> print (to_euler(m,[2,0]))       # from 2 matrices makes 2 euler angles with zxy and xyz rotate orders
        >>> print (to_euler(m[0],range(6))) # from 1 matrix makes 6 euler angles with all possible 6 rotate orders
    """
    matrix = _setDimension(matrix,3, reshape_matrix=True)
    axes   = _setDimension(axes,1,dtype=np.int32)
    
    matrix, axes = _matchDepth(matrix, axes)
    return _matrixToEuler(matrix,axes)


def to_quaternion(matrix):
    """ Converts 4x4 matrix to transform matrices
    
        >>> m = random(2)                     # make 2 random matrices
        >>> print (repr(to_quaternion(m[0]))) # from 1 matrix make 1 quaternion
        >>> print (repr(to_quaternion(m)))    # from 2 matrices make 2 quaternions
    """
    matrix = _setDimension(matrix, 3, reshape_matrix=True)
    return _matrixToQuaternion(matrix)



def normalize(matrix):
    """ Normalizes the rotation component of transform matrices
    """
    matrix = _setDimension(matrix, 3, reshape_matrix=True)
    return _matrixNormalize(matrix)



def inverse(matrix):
    """ Inverse a list of transform matrices
    """
    matrix = _setDimension(matrix, 3, reshape_matrix=True)
    return _matrixInverse(matrix)
    
   
def point(point, matrix):
    """ Point * Matrix multiplication
    """
    point  = _setDimension(point,  2)
    matrix = _setDimension(matrix, 3, reshape_matrix=True)
    
    point, matrix = _matchDepth(point, matrix)
    
    return _matrixPointMultiply(point[:, :3], matrix)       

    
def multiply(matrix0, matrix1):
    """ Matrix * Matrix multiplication
    """
    matrix0 = _setDimension(matrix0, 3, reshape_matrix=True)
    matrix1 = _setDimension(matrix1, 3, reshape_matrix=True)
    matrix0, matrix1 = _matchDepth(matrix0, matrix1)

    return _matrixMultiply(matrix0,matrix1)


def slerp(matrix0, matrix1, weight=0.5):
    """
    slerp(matrix0, matrix1, weight=0.5)
    
        Performs a spherical interpolation between two lists of transform matrices.
        Translation component will be ignored.

        Parameters
        ----------
        matrix0 : *[4x4 float]* or *[[4x4 float],...]* or *[16 float]* or *[[16 float],...]
                  a single, or list of matrices which correspond to weight=0
            
        matrix1 : *[float, float, float, float]* or *[[float, float, float, float],...]*
                  a single, or list of matrices which correspond to weight=1
            
        weight  : *float* or *[float,...]*
                 weight values to interpolate between quat0 and quat1. default = 0.5
            
            
        Returns
        -------
        matrices : np.array(n,4,4)
                   a list of interpolated 4x4 transform matrices
            

        See Also
        --------
        euler.slerp     : Performs a spherical interpolation between two lists of euler angles.
        quaternion.lerp : Performs a spherical interpolation between two lists of quaternions.
        vector.slerp    : Performs a spherical interpolation between two lists of vectors.
        

        Examples
        --------
        >>> matrix0 = random(100)                   # init quat0
        >>> matrix1 = random(100)                   # init quat1
        >>> print (repr(slerp(matrix0 ,matrix1))    # get the halfway point between the two lists
        >>> print (repr(slerp(matrix0, matrix1[0])) # get the halfway point between the all items of matrix0 and the first item of matrix1

    """    
    
    matrix0 = _setDimension(matrix0,3,reshape_matrix=True)
    matrix1 = _setDimension(matrix1,3,reshape_matrix=True)
    weight  = _setDimension(weight,1)
    
    matrix0, matrix1, weight = _matchDepth(matrix0, matrix1, weight)

    q0 = _matrixToQuaternion(matrix0)
    q1 = _matrixToQuaternion(matrix1)
    q  = _quaternionSlerp(q0, q1, weight)
    return _quaternionToMatrix(q)


def interpolate(matrix0, matrix1, weight=0.5):
    """
    interpolate(matrix0, matrix1, weight=0.5)
    
        Performs interpolates scale, rotation and position between two lists of transform matrices.
        

        Parameters
        ----------
        matrix0 : *[4x4 float]* or *[[4x4 float],...]* or *[16 float]* or *[[16 float],...]
                  a single, or list of matrices which correspond to weight=0
            
        matrix1 : *[float, float, float, float]* or *[[float, float, float, float],...]*
                  a single, or list of matrices which correspond to weight=1
            
        weight : *float* or *[float,...]*
                 weight values to interpolate between quat0 and quat1. default = 0.5
            
            
        Returns
        -------
        matrices : np.array(n,4,4)
                   a list of interpolated 4x4 transform matrices
            

        Examples
        --------
        >>> matrix0 = random(100)                                                 # init matrix0
        >>> matrix1 = random(100)                                                 # init matrix1
        >>> print (repr(interpolate(matrix0, matrix1))                            # get the halfway point between the two lists
        >>> print (repr(interpolate(matrix0, matrix1[0]))                         # get the halfway point between the all items of matrix0 and the first item of matrix1
        >>> print (repr(interpolate(matrix0[0], matrix1[0], weight=[.25,.5,.75])) # get the 1/4, 1/2 an 3/4 points between matrix0[0] and matrix1[0]
    """ 
    
    # Set expected dimensions
    matrix0 = _setDimension(matrix0,3,reshape_matrix=True)
    matrix1 = _setDimension(matrix1,3,reshape_matrix=True)
    weight  = _setDimension(weight,1)
    
    matrix0, matrix1, weight = _matchDepth(matrix0, matrix1, weight)
    
    # Grab scale components
    scale0 = np.einsum('...i,...i', matrix0[:,:3,:3], matrix0[:,:3,:3]) ** 0.5
    scale1 = np.einsum('...i,...i', matrix1[:,:3,:3], matrix1[:,:3,:3]) ** 0.5
    
    # Normalize matrices and interpolate rotation
    matrix0[:,0,:3] /= scale0[:,0][:,None]
    matrix0[:,1,:3] /= scale0[:,1][:,None]
    matrix0[:,2,:3] /= scale0[:,2][:,None]
    
    matrix1[:,0,:3] /= scale1[:,0][:,None]
    matrix1[:,1,:3] /= scale1[:,1][:,None]
    matrix1[:,2,:3] /= scale1[:,2][:,None]      
    
    q0 = _matrixToQuaternion(matrix0)
    q1 = _matrixToQuaternion(matrix1)
    matrix = _quaternionToMatrix(_quaternionSlerp(q0, q1, weight))
    
    
    # Interpolate scale
    scale = _vectorLerp(scale0,scale1,weight)
    
    
    # Scale interpolated matrices
    matrix[:,0,:3] *= scale[:,0][:,None]
    matrix[:,1,:3] *= scale[:,1][:,None]
    matrix[:,2,:3] *= scale[:,2][:,None]
    
    
    # Interpolate position
    matrix[:,3,:3] = _vectorLerp(matrix0[:,3,:3], matrix1[:,3,:3], weight)    
    
    return matrix


def local(matrix, parent_matrix):
    """ Returns the local matrix to a parent matrix
    """
    
    matrix        = _setDimension(matrix, 3, reshape_matrix=True)
    parent_matrix = _setDimension(parent_matrix, 3, reshape_matrix=True)
    
    matrix0, parent_matrix = _matchDepth(matrix0, parent_matrix)

    return _matrixMultiply(matrix, _matrixInverse(parent_matrix))


def random(n, seed=None, random_position=False):
    """ Computes a list of random 4x4 rotation matrices
    """
    euler = np.random.seed(seed)
    euler = np.radians(360 - np.random.random((n,3))*720)
    
    M = _eulerToMatrix(euler, np.zeros(euler.shape[0], dtype=np.int32))
    
    if random_position:
        M[:,3,:3] = 1 - np.random.random((n, 3)) * 2

    return M


