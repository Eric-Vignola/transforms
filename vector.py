"""
BSD 3-Clause License:
Copyright (c)  2023, Eric Vignola
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:


1. Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

3. Neither the name of copyright holders nor the names of its
    contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""



import numpy as np

from ._transforms import (
    _matrixToEuler,
    _matrixToQuaternion,
    _quaternionToMatrix,
    _vectorArc,
    _vectorArcToQuaternion,
    _vectorCross,
    _vectorDot,
    _vectorLerp,
    _vectorMagnitude,
    _vectorNormalize,
    _vectorSlerp,
    _vectorToMatrix,
)

from ._utils import _matchDepth, _setDimension

from .quaternion import to_euler as quaternionToEuler


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


#----------------------------------------------- VECTOR MATH -----------------------------------------------#

def to_matrix(vector0, vector1, aim_axis=0, up_axis=1, extrapolate=False):
    """
    to_matrix(vector0, vector1, aim_axis=0, up_axis=1, extrapolate=False)
    
        Converts an array of aim (vector0) and up (vector1) vectors to 4x4 transform matrices.

        Parameters
        ----------
        vector0  : *[float, float, float]* or *[[float, float, float],...]*
                   a single, or list of aim direction vectors
            
        vector1  : *[float, float, float]* or *[[float, float, float],...]*
                   a single, or list of up direction vectors
            
        aim_axis : *X, Y, Z* or *int* or *[int,...]*
                   defines the aim axis
            
        up_axis  : *X, Y, Z* or *int* or *[int,...]*
                   defines the up axis
            
        extrapolate : *bool*
                      uses the up axes from the previous solution in the list
            
        Returns
        -------
        matrices: np.array(n,4,4)
                  a list of 4x4 transform matrices
            

        See Also
        --------
        vectorToEuler : Converts an array of aim (vector0) and up (vector1) vectors to euler angles.
        vectorToQuaternion: Converts an array of aim (vector0) and up (vector1) vectors to quaternions qi,qj,qk,qw.
        

        Examples
        --------
        >>> vector0 = randomVector(100)                                    # random aim vector array
        >>> vector1 = randomVector(100)                                    # random up vector array
        >>> print (vectorToMatrix(vector0, vector1))                       # computes quaternions qi,qj,qk,qw from aim and up vector arrays
        >>> print (vectorToMatrix(vector0, vector1[0], extrapolate=False)) # computes quaternions qi,qj,qk,qw from aim vector array and a single up vector
        >>> print (vectorToMatrix(vector0, vector1[0], extrapolate=True))  # computes quaternions qi,qj,qk,qw from aim vector array and a single up vector and extrapolate next up vector from previous solution
    """          
    
    vector0  = _setDimension(vector0,2)
    vector1  = _setDimension(vector1,2)
    aim_axis = _setDimension(aim_axis,1,dtype=np.int32) % 3
    up_axis  = _setDimension(up_axis,1,dtype=np.int32) % 3
    
    vector0, vector1, aim_axis, up_axis = _matchDepth(vector0, vector1, aim_axis, up_axis)
    
    return _vectorToMatrix(vector0, vector1, aim_axis, up_axis)


def to_quaternion(vector0, vector1, aim_axis=X, up_axis=Y):
    
    """
    to_quaternion(vector0, vector1, aim_axis=0, up_axis=1)
    
        Converts an array of aim (vector0) and up (vector1) vectors to quaternions qi,qj,qk,qw.

        Parameters
        ----------
        vector0 : *[float, float, float]* or *[[float, float, float],...]*
                  a single, or list of aim direction vectors
            
        vector1 : *[float, float, float]* or *[[float, float, float],...]*
                  a single, or list of up direction vectors
            
        aim_axis : *X, Y, Z* or *int* or *[int,...]*
                   defines the aim axis
            
        up_axis : *X, Y, Z* or *int* or *[int,...]*
                  defines the up axis
            
        extrapolate : *bool*
                      uses the up axes from the previous solution in the list
            
        Returns
        -------
        quaternion: np.array(n,[qi,qj,qk,qw]) 
                    a list of quaternions
            

        See Also
        --------
        vectorToEuler: Converts an array of aim (vector0) and up (vector1) vectors to euler angles.
        vectorToMatrix : Converts an array of aim (vector0) and up (vector1) vectors to transform matrices.
        

        Examples
        --------
        >>> vector0 = random(100)                                         # random aim vector array
        >>> vector1 = random(100)                                         # random up vector array
        >>> print (to_quaternion(vector0, vector1))                       # computes quaternions qi,qj,qk,qw from aim and up vector arrays
        >>> print (to_quaternion(vector0, vector1[0], extrapolate=False)) # computes quaternions qi,qj,qk,qw from aim vector array and a single up vector
        >>> print (to_quaternion(vector0, vector1[0], extrapolate=True))  # computes quaternions qi,qj,qk,qw from aim vector array and a single up vector and extrapolate next up vector from previous solution
    """           
    vector0  = _setDimension(vector0,2)
    vector1  = _setDimension(vector1,2)
    aim_axis = _setDimension(aim_axis,1,dtype=np.int32) % 3
    up_axis  = _setDimension(up_axis,1,dtype=np.int32) % 3
    
    vector0, vector1, aim_axis, up_axis = _matchDepth(vector0, vector1, aim_axis, up_axis)
    
    M = _vectorToMatrix(vector0, vector1, aim_axis, up_axis)
    return _matrixToQuaternion(M)



def to_euler(vector0, vector1, aim_axis=0, up_axis=1, axes=XYZ, extrapolate=False):
    
    """
    to_euler(vector0, vector1, aim_axis=0, up_axis=1, extrapolate=False)
    
        Converts an array of aim (vector0) and up (vector1) vectors to euler angles.

        Parameters
        ----------
        vector0 : *[float, float, float]* or *[[float, float, float],...]*
                  a single, or list of aim direction vectors
            
        vector1 : *[float, float, float]* or *[[float, float, float],...]*
                  a single, or list of up direction vectors
            
        aim_axis : *X, Y, Z* or *int* or *[int,...]*
                   defines the aim axis
            
        up_axis : *X, Y, Z* or *int* or *[int,...]*
                  defines the up axis
            
        extrapolate : *bool*
                      uses the up axes from the previous solution in the list
            
        Returns
        -------
        quaternion: np.array(n,3) 
                    a list of euler angles
            

        See Also
        --------
        vectorToQuaternion : Converts an array of aim (vector0) and up (vector1) vectors to quaternions qi,qj,qk,qw.
        vectorToMatrix     : Converts an array of aim (vector0) and up (vector1) vectors to transform matrices.
        

        Examples
        --------
        >>> vector0 = random(100)                                    # random aim vector array
        >>> vector1 = random(100)                                    # random up vector array
        >>> print (to_euler(vector0, vector1))                       # computes quaternions qi,qj,qk,qw from aim and up vector arrays
        >>> print (to_euler(vector0, vector1[0], extrapolate=False)) # computes quaternions qi,qj,qk,qw from aim vector array and a single up vector
        >>> print (to_euler(vector0, vector1[0], extrapolate=True))  # computes quaternions qi,qj,qk,qw from aim vector array and a single up vector and extrapolate next up vector from previous solution
    """          
    
    vector0  = _setDimension(vector0,2)
    vector1  = _setDimension(vector1,2)
    aim_axis = _setDimension(aim_axis,1,dtype=np.int32) % 3
    up_axis  = _setDimension(up_axis,1,dtype=np.int32) % 3
    axes     = _setDimension(axes,1,dtype=np.int32)
    
    vector0, vector1, aim_axis, up_axis, axes = _matchDepth(vector0, vector1, aim_axis, up_axis, axes)
    
    return _matrixToEuler(_vectorToMatrix(vector0, vector1, aim_axis, up_axis), axes)



def cross(vector0, vector1):

    """
    cross(vector0, vector1)
    
        Computes the cross product between 2 lists of vectors.

        Parameters
        ----------
        vector0 : *[float, float, float]* or *[[float, float, float],...]*
                  a single, or list of vectors
            
        vector1 : *[float, float, float]* or *[[float, float, float],...]*
                  a single, or list of vectors
            
            
        Returns
        -------
        vector: np.array(n,3) 
                a list of vectors
            

        See Also
        --------
        dot       : Computes the dot product between 2 lists of vectors.
        magnitude : Computes the magnitude of a list of vectors.
        normalize : Normalizes a list of vectors.
        
        
        Examples
        --------
        >>> vector0 = random(100)              # random vector array
        >>> vector1 = random(100)              # random vector array
        >>> print (cross(vector0, vector1))    # computes the cross product between 2 lists of vectors
        >>> print (cross(vector0, vector1[0])) # computes the cross product between a lists of vectors and a single vector
    """         
    vector0 = _setDimension(vector0,2)
    vector1 = _setDimension(vector1,2)
    vector0, vector1 = _matchDepth(vector0, vector1)
            
    return _vectorCross(vector0, vector1)


def dot(vector0, vector1):
    """
    dot(vector0, vector1)
    
        Computes the dot product between 2 lists of vectors.

        Parameters
        ----------
        vector0 : *[float, float, float]* or *[[float, float, float],...]*
            a single, or list of vectors
            
        vector1 : *[float, float, float]* or *[[float, float, float],...]*
            a single, or list of vectors
            
            
        Returns
        -------
        float: np.array(n) 
            a list of floats
            

        See Also
        --------
        cross     : Computes the cross product between 2 lists of vectors.
        magnitude : Computes the magnitude of a list of vectors.
        normalize : Normalizes a list of vectors.
        
        
        Examples
        --------
        >>> vector0 = random(100)            # random vector array
        >>> vector1 = random(100)            # random vector array
        >>> print (dot(vector0, vector1))    # computes the dot product between 2 lists of vectors
        >>> print (dot(vector0, vector1[0])) # computes the dot product between a lists of vectors and a single vector
    """  
    
    vector0 = _setDimension(vector0,2)
    vector1 = _setDimension(vector1,2)
    vector0, vector1 = _matchDepth(vector0, vector1)
            
    return _vectorDot(vector0, vector1)


def magnitude(vector):
    """
    magnitude(vector)
    
        Computes the magnitude of a list of vectors.

        Parameters
        ----------
        vector : *[float, float, float]* or *[[float, float, float],...]*
                 a single, or list of vectors

            
        Returns
        -------
        float: np.array(n) 
               a list of floats
            

        See Also
        --------
        cross     : Computes the cross product between 2 lists of vectors.
        dot       : Computes the dot product of a list of vectors.
        normalize : Normalizes a list of vectors.
        
        
        Examples
        --------
        >>> vector = random(100)      # random vector array
        >>> print (magnitude(vector)) # computes the magnitude of a list vectors
        """      
    
    vector = _setDimension(vector,2)
    
    return _vectorMagnitude(vector)



def normalize(vector):
    """
    normalize(vector)
    
        Normalizes a list of vectors.

        Parameters
        ----------
        vector : *[float, float, float]* or *[[float, float, float],...]*
                 a single, or list of vectors

            
        Returns
        -------
        vector: np.array(n,3) 
                a list of normalized vectors
            

        See Also
        --------
        cross     : Computes the cross product between 2 lists of vectors.
        dot       : Computes the dot product of a list of vectors.
        magnitude : Computes the magnitude of a list of vectors.
        
        
        Examples
        --------
        >>> vector = random(100)      # random vector array
        >>> print (normalize(vector)) # computes the magnitude of a list vectors
        """      
    
    vector = _setDimension(vector,2)

    return _vectorNormalize(vector)



def slerp(vector0, vector1, weight=0.5):
    """
    slerp(vector0, vector1, weight=0.5)
    
        Spherical interpolation between 2 lists of vectors

        Parameters
        ----------
        vector0 : *[float, float, float]* or *[[float, float, float],...]*
                  a single, or list of vectors
        
        vector1 : *[float, float, float]* or *[[float, float, float],...]*
                  a single, or list of vectors
            
        weight  : *[float,...]* or *float*
                  a single, or list of interpolation weights (default=0.5)

            
        Returns
        -------
        vector: np.array(n,3) 
                a list of spherically interpolated vectors
            

        See Also
        --------
        lerp: Linear interpolation between 2 lists of vectors.
        
        
        Examples
        --------
        >>> vector0 = random(100)                         # random vector array
        >>> vector1 = random.random(100)                  # random float weight array
        >>> print (slerp(vector0, vector1, weight))       # slerp 2 lists of vectors by list of floats
        >>> print (slerp(vector0, vector1, weight[0]))    # slerp 2 lists of vectors by same weight value
        >>> print (slerp(vector0[0], vector1[0], weight)) # slerp 2 vectors by list of weights
        """     
    
    vector0 = _setDimension(vector0,2)
    vector1 = _setDimension(vector1,2)
    weight  = _setDimension(weight,1)
    
    vector0, vector1, weight = _matchDepth(vector0, vector1, weight)
    
    return _vectorSlerp(vector0,vector1,weight)



def lerp(vector0, vector1, weight=0.5):
    """
    lerp(vector0, vector1, weight=0.5)
    
        Linear interpolation between 2 lists of vectors

        Parameters
        ----------
        vector0 : *[float, float, float]* or *[[float, float, float],...]*
                  a single, or list of vectors
        
        vector1 : *[float, float, float]* or *[[float, float, float],...]*
                  a single, or list of vectors
            
        weight  : *[float,...]* or *float*
                  a single, or list of interpolation weights (default=0.5)
            
        
        Returns
        -------
        vector: np.array(n,3) 
                a list of linearly interpolated vectors
            

        See Also
        --------
        slerp: Linear interpolation between 2 lists of vectors.
        
        
        Examples
        --------
        >>> vector0 = random(100)                         # random vector array
        >>> vector1 = random(100)                         # random vector array
        >>> weight  = np.random.random(100)               # random float weight array
        >>> print (lerp(vector0, vector1, weight))        # slerp 2 lists of vectors by list of floats
        >>> print (lerp(vector0, vector1, weight[0]))     # slerp 2 lists of vectors by same weight value
        >>> print ((lerp(vector0[0], vector1[0], weight)) # slerp 2 vectors by list of weights
        """      
    
    vector0 = _setDimension(vector0,2)
    vector1 = _setDimension(vector1,2)
    weight  = _setDimension(weight,1)
    
    vector0, vector1, weight = _matchDepth(vector0, vector1, weight)
    
    return np.nan_to_num(_vectorLerp(vector0, vector1, weight))



def arc_to_quaternion(vector0, vector1):
    """
    arc_to_quaternion(vector0, vector1)
    
        Computes the arc between 2 lists of vectors in the form of quaternions qi,qj,qk,qw

        Parameters
        ----------
        vector0 : *[float, float, float]* or *[[float, float, float],...]*
                  a single, or list of vectors
        
        vector1 : *[float, float, float]* or *[[float, float, float],...]*
                  a single, or list of vectors
            
            
        Returns
        -------
        quaternion: np.array(n,4)
                    a list of quaternions qi,qj,qk,qw.
            

        See Also
        --------
        arc_to_matrix: Computes the arc between 2 lists of vectors in the form of 4x4 transform matrices.
        arc_to_euler:  Computes the arc between 2 lists of vectors in the form of euler angles.
     
        
        Examples
        --------
        >>> vector0 = random(100)                          # random vector array
        >>> vector1 = random(100)                          # random vector array
        >>> print (arc_to_quaternion(vector0, vector1)     # arc between 2 lists of vectors
        >>> print (arc_to_quaternion(vector0, vector1[0])) # arc between a list of vectors and a single vector
        """          
    
    
    vector0 = _setDimension(vector0,2)
    vector1 = _setDimension(vector1,2)
    
    vector0, vector1 = _matchDepth(vector0, vector1)
            
    return _vectorArcToQuaternion(vector0, vector1)



def arc_to_matrix(vector0, vector1):
    """
    arc_to_matrix(vector0, vector1)
    
        Computes the arc between 2 lists of vectors in the form of 4x4 transformation matrices

        Parameters
        ----------
        vector0 : *[float, float, float]* or *[[float, float, float],...]*
                  a single, or list of vectors
        
        vector1 : *[float, float, float]* or *[[float, float, float],...]*
                  a single, or list of vectors
            
            
        Returns
        -------
        matrices: np.array(n,4,4)
                  a list of 4x4 transform matrices
            

        See Also
        --------
        arc_to_matrix: Computes the arc between 2 lists of vectors in the form of 4x4 transform matrices.
        arc_to_euler:  Computes the arc between 2 lists of vectors in the form of euler angles.
     
        
        Examples
        --------
        >>> vector0 = random(100)                      # random vector array
        >>> vector1 = random(100)                      # random vector array
        >>> print (arc_to_matrix(vector0, vector1))    # arc between 2 lists of vectors
        >>> print (arc_to_matrix(vector0, vector1[0])) # arc between a list of vectors and a single vector
        """
    
    vector0 = _setDimension(vector0,2)
    vector1 = _setDimension(vector1,2)
    
    vector0, vector1 = _matchDepth(vector0, vector1)
    
    return _quaternionToMatrix(_vectorArcToQuaternion(vector0, vector1))




def arc_to_euler(vector0, vector1, axes=XYZ):
    """
    arc_to_euler(vector0, vector1, axes=XYZ)
    
        Computes the arc between 2 lists of vectors in the form of euler angles.

        Parameters
        ----------
        vector0 : *[float, float, float]* or *[[float, float, float],...]*
            a single, or list of vectors
        
        vector1 : *[float, float, float]* or *[[float, float, float],...]*
            a single, or list of vectors
            
        axes : *XYZ, YZX, ZXY, XZY, YXZ, ZYX*, *int* or *[int,...]*
            rotate order of the resulting arc. default = XYZ
            
            
            
        Returns
        -------
        euler angle: np.array(n,3)
            a list of interpolated euler angles
            
            
            
        See Also
        --------
        vectorArcToMatrix: Computes the arc between 2 lists of vectors in the form of 4x4 transform matrices.
        vectorArcToEuler:  Computes the arc between 2 lists of vectors in the form of euler angles.
     
        
        Examples
        --------
        >>> vector0 = random(100)                     # random vector array
        >>> vector1 = random(100)                     # random vector array
        >>> print (arc_to_euler(vector0, vector1)     # arc between 2 lists of vectors
        >>> print (arc_to_euler(vector0, vector1[0])) # arc between a list of vectors and a single vector
        """
    
    vector0 = _setDimension(vector0,2)
    vector1 = _setDimension(vector1,2)
    axes    = _setDimension(axes,1,dtype=np.int32)
    
    vector0, vector1, axes = _matchDepth(vector0, vector1, axes)
    
    return quaternionToEuler(_vectorArcToQuaternion(vector0,vector1), axes)  



def angle(vector0, vector1):
    """
    angle(vector0, vector1, axes=XYZ)
    
        Computes the arc angle between 2 lists of vectors.

        Parameters
        ----------
        vector0 : *[float, float, float]* or *[[float, float, float],...]*
            a single, or list of vectors
        
        vector1 : *[float, float, float]* or *[[float, float, float],...]*
            a single, or list of vectors

            
        Returns
        -------
        angle: np.array(n)
            a list of angles in radians
            
            
        Examples
        --------
        >>> vector0 = random(100)              # random vector array
        >>> vector1 = random(100)              # random vector array
        >>> print (angle(vector0, vector1)     # arc angle between 2 lists of vectors
        >>> print (angle(vector0, vector1[0])) # arc angle between a list of vectors and a single vector
        """
    
    vector0 = _setDimension(vector0,2)
    vector1 = _setDimension(vector1,2)
    
    return _vectorArc(vector0, vector1)


def random(n, seed=None, normalize=False):
    """ Computes a list of random vectors
    """
    if seed is not None:
        np.random.seed(seed)
        
    if normalize:
        return _vectorNormalize(1 - np.random.random((n,3))*2)
    
    return 1 - np.random.random((n,3)) * 2
        
