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

from ._transforms import _eulerToMatrix, _eulerToQuaternion, _matrixToEuler
from ._transforms import _quaternionSlerp, _quaternionToMatrix
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



#----------------------------------------------- EULER MATH -----------------------------------------------#

def to_matrix(euler, axes=XYZ):
    """
    to_matrix(euler, axes=XYZ)
    
        Converts euler angles to a 4x4 transform matrices.

        Parameters
        ----------
        euler : *[float, float, float]* or *[[float, float, float],...]*
                a single, or list of euler angles
            
        axes : *XYZ, YZX, ZXY, XZY, YXZ, ZYX*, *int* or *[int,...]*
               corresponding rotate orders. default = XYZ
            
            
        Returns
        -------
        matrices: np.array(n,4,4)
                  a list of 4x4 transform matrices
            

        See Also
        --------
        eulerToEuler :      Converts euler angles from one rotate order to another.
        eulerToQuaternion : Converts euler angles to quaternions qi,qj,qk,qw.
        

        Examples
        --------
        >>> euler = random(2)                # make 2 random Maya euler angles
        >>> print (to_matrix(euler[0], XYZ)) # from 1 euler angle  with xyz rotate order make 1 matrix
        >>> print (to_matrix(euler, XYZ))    # from 2 euler angles with xyz rotate order make 2 matrices
        >>> print (to_matrix(euler, [2,0]))  # from 2 euler angles with zxy and xyz rotate orders make 2 matrices
    """    
    
    euler = _setDimension(euler,2)
    axes  = _setDimension(axes, 1, dtype=np.int32)
    euler, axes = _matchDepth(euler, axes)
    
    return _eulerToMatrix(euler,axes)



def to_quaternion(euler, axes=XYZ):
    """
    to_quaternion(euler,axes=XYZ)
    
        Converts euler angles to quaternions qi,qj,qk,qw.
    
    
        Parameters
        ----------
        euler : *[float, float, float]* or *[[float, float, float],...]*
                a single, or list of euler angles
            
        axes  : *XYZ, YZX, ZXY, XZY, YXZ, ZYX*, *int* or *[int,...]*
                corresponding rotate orders. default = XYZ
            
            
        Returns
        -------
        quaternion: np.array(n,4)
            a list of quaternions qi,qj,qk,qw.
            
    
        See Also
        --------
        to_euler  : Converts euler angles from one rotate order to another.
        to_matrix : Converts euler angles to a 4x4 transform matrices.
        
    
        Examples
        --------
        >>> euler = randomEuler(2)                  # make 2 random Maya euler angles
        >>> print (to_quaternion(euler[0], XYZ))    # from 1 euler angle  with xyz rotate order make 1 quaternion
        >>> print (to_quaternion(euler, XYZ))       # from 2 euler angles with xyz rotate order make 2 quaternions
        >>> print (to_quaternion(euler, [ZXY,YZX])) # from 2 euler angles with zxy and yzx rotate orders make 2 quaternions
    """
    euler = _setDimension(euler,2)
    axes  = _setDimension(axes,1,dtype=np.int32)
    euler, axes = _matchDepth(euler, axes)
    
    return _eulerToQuaternion(euler,axes)




def slerp(euler0, euler1, weight=0.5, axes0=XYZ, axes1=XYZ, axes=XYZ):
    """
    slerp(euler0, euler1, weight=0.5, axes0=XYZ, axes1=XYZ, axes=XYZ)
    
        Performs a spherical interpolation between two lists of euler angles.

        Parameters
        ----------
        euler0 : *[float, float, float]* or *[[float, float, float],...]*
                 a single, or list of euler angles which correspond to weight=0
            
        euler1 : *[float, float, float]* or *[[float, float, float],...]*
                 a single, or list of euler angles which correspond to weight=1
            
        weight : *float* or *[float,...]*
                 weight values to interpolate between euler0 and euler1. default = 0.5
            
        axes0  : *XYZ, YZX, ZXY, XZY, YXZ, ZYX*, *int* or *[int,...]*
                 corresponding rotate orders for euler0. default = XYZ
            
        axes1  : *XYZ, YZX, ZXY, XZY, YXZ, ZYX*, *int* or *[int,...]*
                 corresponding rotate orders for euler1. default = XYZ
            
        axes   : *XYZ, YZX, ZXY, XZY, YXZ, ZYX*, *int* or *[int,...]*
                 corresponding rotate orders the output interpolated euler angles. default = XYZ
            
        Returns
        -------
        euler angle: np.array(n,3)
                     a list of interpolated euler angles
            

        See Also
        --------
        matrix.slerp     : Performs a spherical interpolation between two lists of 4x4 matrices.
        quaternion.slerp : Performs a spherical interpolation between two lists of quaternions.
        vector.slerp     : Performs a spherical interpolation between two lists of vectors.
        

        Examples
        --------
        >>> euler0 = random(100)                                    # random euler angle array
        >>> euler1 = random(100)                                    # random euler angle array
        >>> print (slerp(euler0, euler1)                            # get the halfway point between the two lists
        >>> print (slerp(euler0, euler1[0])                         # get the halfway point between the all items of euler0 and the first item of euler1
        >>> print (slerp(euler0[0], euler1[0], weight=[.25,.5,.75]) # get the 1/4, 1/2 an 3/4 points between euler0[0] and euler1[0]
    """
    
    
    euler0 = _setDimension(euler0,2)
    euler1 = _setDimension(euler1,2)
    weight = _setDimension(weight,1)
    
    axes0  = _setDimension(axes0,1,dtype=np.int32)    
    axes1  = _setDimension(axes1,1,dtype=np.int32)    
    axes   = _setDimension(axes ,1,dtype=np.int32)
    
    euler0, euler0, weight, axes0, axes1, axes = _matchDepth(euler0, euler0, weight, axes0, axes1, axes)
    
    q0 = _eulerToQuaternion(euler0,axes0)
    q1 = _eulerToQuaternion(euler1,axes1)
    q  = _quaternionSlerp(q0, q1, weight)
    
    return _matrixToEuler(_quaternionToMatrix(q), axes)



def to_euler(euler, from_axes, to_axes):
    """
    to_euler(euler, from_axes, to_axes)
    
        Converts euler angles from one rotate order to another.
        
        
        Parameters
        ----------
        euler     : *[float, float, float]* or *[[float, float, float],...]*
                    a single, or list of euler angles which correspond to weight=0
            
        from_axes : *XYZ, YZX, ZXY, XZY, YXZ, ZYX*, *int* or *[int,...]*
                    corresponding rotate orders for from_axes. (ex: from_axes=XYZ)
            
        to_axes   : *XYZ, YZX, ZXY, XZY, YXZ, ZYX*, *int* or *[int,...]*
                    corresponding rotate orders for to_axes. (ex: to_axes=ZXY)
                 
        Returns
        -------
        euler angle: np.array(n,3)
                     a list of converted euler angles
            
        See Also
        --------
        to_matrix:     Converts euler angles to a 4x4 transform matrices.
        to_quaternion: Converts euler angles to quaternions qi,qj,qk,qw.
        
        
        Examples
        --------
        >>> euler = random(100)                                # random euler angle array
        >>> print (to_euler(euler, from_axes=XYZ, to_axes=ZXY) # convert array from xyz rotate order to zxy

    """            
    
    euler     = _setDimension(euler,     2)    
    from_axes = _setDimension(from_axes, 1, dtype=np.int32)    
    to_axes   = _setDimension(to_axes,   1, dtype=np.int32)
    
    euler, from_axes, to_axes = _matchDepth(euler, from_axes, to_axes)
    
    M = _eulerToMatrix(euler, from_axes)
    return _matrixToEuler(M, to_axes)    


def random(n, seed=None):
    """ Computes a list of random euler angles
    """
    np.random.seed(seed)
    return np.radians(360 - np.random.random((n,3))*720)


