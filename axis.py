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

from ._transforms import _axisAngleToMatrix, _axisAngleToQuaternion, _matrixToEuler
from ._utils import _matchDepth, _setDimension


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

