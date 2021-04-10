import numpy as np
from numpy.core.umath_tests import inner1d

from _transforms import _axisAngleToEuler, _axisAngleToMatrix, _axisAngleToQuaternion
from _transforms import _eulerSlerp, _eulerToEuler, _eulerToMatrix, _eulerToQuaternion
from _transforms import _matrixIdentity, _matrixInverse, _matrixLocal, _matrixMultiply, _matrixNormalize, _matrixPointMultiply, _matrixSlerp, _matrixToEuler, _matrixToQuaternion
from _transforms import _quaternionAdd, _quaternionConjugate, _quaternionDot, _quaternionInverse, _quaternionMultiply, _quaternionNegate, _quaternionNormalize, _quaternionSlerp, _quaternionSub, _quaternionToEuler, _quaternionToMatrix
from _transforms import _vectorArcToEuler, _vectorArcToMatrix, _vectorArcToQuaternion, _vectorCross, _vectorDot, _vectorLerp, _vectorMagnitude, _vectorNormalize, _vectorSlerp, _vectorToEuler, _vectorToMatrix, _vectorToQuaternion


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
    
    data = np.asarray(data).astype(dtype)
    
    while data.ndim < ndim:
        data = data[np.newaxis]
        
    # For when matrices are given as lists of 16 floats
    if reshape_matrix:
        if data.shape[-1] == 16:
            data = data.reshape(-1,4,4)

    return data




#----------------------------------------------- EULER MATH -----------------------------------------------#

def eulerToMatrix(euler, axes=XYZ):
    """
    eulerToMatrix(euler,axes=XYZ)
    
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
        eulerToEuler : Converts euler angles from one rotate order to another.
        eulerToQuaternion : Converts euler angles to quaternions qi,qj,qk,qw.
        

        Examples
        --------
        >>> euler = randomEuler(2)                           # make 2 random Maya euler angles
        >>> print repr(eulerToMatrix(euler[0],XYZ))          # from 1 euler angle  with xyz rotate order make 1 matrix
        >>> print repr(eulerToMatrix(euler,XYZ))             # from 2 euler angles with xyz rotate order make 2 matrices
        >>> print repr(eulerToMatrix(euler,[2,0]))           # from 2 euler angles with zxy and xyz rotate orders make 2 matrices
    """    
    
    euler = _setDimension(euler,2)
    axes  = _setDimension(axes,1,dtype=np.int64)
    
    return _eulerToMatrix(euler,axes)



def eulerToQuaternion(euler, axes=XYZ):
    """
    eulerToQuaternion(euler,axes=XYZ)
    
        Converts euler angles to quaternions qi,qj,qk,qw.
    
    
        Parameters
        ----------
        euler : *[float, float, float]* or *[[float, float, float],...]*
            a single, or list of euler angles
            
        axes : *XYZ, YZX, ZXY, XZY, YXZ, ZYX*, *int* or *[int,...]*
            corresponding rotate orders. default = XYZ
            
            
        Returns
        -------
        quaternion: np.array(n,4)
            a list of quaternions qi,qj,qk,qw.
            
    
        See Also
        --------
        eulerToEuler : Converts euler angles from one rotate order to another.
        eulerToMatrix : Converts euler angles to a 4x4 transform matrices.
        
    
        Examples
        --------
        >>> euler = randomEuler(2)                               # make 2 random Maya euler angles
        >>> print repr(eulerToQuaternion(euler[0],XYZ))          # from 1 euler angle  with xyz rotate order make 1 quaternion
        >>> print repr(eulerToQuaternion(euler,XYZ))             # from 2 euler angles with xyz rotate order make 2 quaternions
        >>> print repr(eulerToQuaternion(euler,[ZXY,YZX]))       # from 2 euler angles with zxy and yzx rotate orders make 2 quaternions
    """
    euler = _setDimension(euler,2)
    axes  = _setDimension(axes,1,dtype=np.int64)
    
    return _eulerToQuaternion(euler,axes)




def eulerSlerp(euler0, euler1, weight=0.5, axes0=XYZ, axes1=XYZ, axes=XYZ):
    """
    eulerSlerp(euler0, euler1, weight=0.5, axes0=XYZ, axes1=XYZ, axes=XYZ)
    
        Performs a spherical interpolation between two lists of euler angles.

        Parameters
        ----------
        euler0 : *[float, float, float]* or *[[float, float, float],...]*
            a single, or list of euler angles which correspond to weight=0
            
        euler1 : *[float, float, float]* or *[[float, float, float],...]*
            a single, or list of euler angles which correspond to weight=1
            
        weight : *float* or *[float,...]*
            weight values to interpolate between euler0 and euler1. default = 0.5
            
        axes0 : *XYZ, YZX, ZXY, XZY, YXZ, ZYX*, *int* or *[int,...]*
            corresponding rotate orders for euler0. default = XYZ
            
        axes1 : *XYZ, YZX, ZXY, XZY, YXZ, ZYX*, *int* or *[int,...]*
            corresponding rotate orders for euler1. default = XYZ
            
        axes : *XYZ, YZX, ZXY, XZY, YXZ, ZYX*, *int* or *[int,...]*
            corresponding rotate orders the output interpolated euler angles. default = XYZ
            
        Returns
        -------
        euler angle: np.array(n,3)
            a list of interpolated euler angles
            

        See Also
        --------
        matrixSlerp: Performs a spherical interpolation between two lists of 4x4 matrices.
        quaternionSlerp : Performs a spherical interpolation between two lists of quaternions.
        vectorSlerp : Performs a spherical interpolation between two lists of vectors.
        

        Examples
        --------
        >>> euler0 = randomEuler(100)                                      # random euler angle array
        >>> euler1 = randomEuler(100)                                      # random euler angle array
        >>> print repr(eulerSlerp(euler0,euler1)                           # get the halfway point between the two lists
        >>> print repr(eulerSlerp(euler0,euler1[0])                        # get the halfway point between the all items of euler0 and the first item of euler1
        >>> print repr(eulerSlerp(euler0[0],euler1[0],weight=[.25,.5,.75]) # get the 1/4, 1/2 an 3/4 points between euler0[0] and euler1[0]
    """        
    
    
    
    euler0 = _setDimension(euler0,2)
    euler1 = _setDimension(euler1,2)
    
    weight = _setDimension(weight,2)
    
    axes0  = _setDimension(axes0,1,dtype=np.int64)    
    axes1  = _setDimension(axes1,1,dtype=np.int64)    
    axes   = _setDimension(axes ,1,dtype=np.int64)    
    
    return _eulerSlerp(euler0,euler1,weight,axes0,axes1,axes,weight)



def eulerToEuler(euler, from_axes, to_axes):
    """
    eulerToEuler(euler, from_axes, to_axes)
    
        Converts euler angles from one rotate order to another.
        
        
        Parameters
        ----------
        euler : *[float, float, float]* or *[[float, float, float],...]*
            a single, or list of euler angles which correspond to weight=0
            
        from_axes : *XYZ, YZX, ZXY, XZY, YXZ, ZYX*, *int* or *[int,...]*
            corresponding rotate orders for from_axes. (ex: from_axes=XYZ)
            
        to_axes : *XYZ, YZX, ZXY, XZY, YXZ, ZYX*, *int* or *[int,...]*
            corresponding rotate orders for to_axes. (ex: to_axes=ZXY)
                 
        Returns
        -------
        euler angle: np.array(n,3)
            a list of converted euler angles
            
        See Also
        --------
        eulerToMatrix: Converts euler angles to a 4x4 transform matrices.
        eulerToQuaternion: Converts euler angles to quaternions qi,qj,qk,qw.
        
        
        Examples
        --------
        >>> euler = randomEuler(100)                                 # random euler angle array
        >>> print repr(eulerToEuler(euler,from_axes=XYZ,to_axes=ZXY) # convert array from xyz rotate order to zxy

    """            
    
    euler     = _setDimension(euler,2)    
    from_axes = _setDimension(from_axes,1,dtype=np.int64)    
    to_axes   = _setDimension(to_axes,1,dtype=np.int64)   
    
    return _eulerToEuler(euler, from_axes, to_axes)




#----------------------------------------------- VECTOR MATH -----------------------------------------------#

def vectorToMatrix(vector0, vector1, aim_axis=0, up_axis=1, extrapolate=False):
    """
    vectorToMatrix(vector0, vector1, aim_axis=0, up_axis=1, extrapolate=False)
    
        Converts an array of aim (vector0) and up (vector1) vectors to 4x4 transform matrices.

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
        matrices: np.array(n,4,4)
            a list of 4x4 transform matrices
            

        See Also
        --------
        vectorToEuler : Converts an array of aim (vector0) and up (vector1) vectors to euler angles.
        vectorToQuaternion: Converts an array of aim (vector0) and up (vector1) vectors to quaternions qi,qj,qk,qw.
        

        Examples
        --------
        >>> vector0 = randomVector(100)                                  # random aim vector array
        >>> vector1 = randomVector(100)                                  # random up vector array
        >>> print vectorToMatrix(vector0, vector1)                       # computes quaternions qi,qj,qk,qw from aim and up vector arrays
        >>> print vectorToMatrix(vector0, vector1[0], extrapolate=False) # computes quaternions qi,qj,qk,qw from aim vector array and a single up vector
        >>> print vectorToMatrix(vector0, vector1[0], extrapolate=True)  # computes quaternions qi,qj,qk,qw from aim vector array and a single up vector and extrapolate next up vector from previous solution
    """          
    
    vector0  = _setDimension(vector0,2)
    vector1  = _setDimension(vector1,2)
    aim_axis = _setDimension(aim_axis,1,dtype=np.int64) % 3
    up_axis  = _setDimension(up_axis,1,dtype=np.int64) % 3
    
    return _vectorToMatrix(vector0,vector1,aim_axis,up_axis,extrapolate)


def vectorToQuaternion(vector0, vector1, aim_axis=X, up_axis=Y, extrapolate=False):
    
    """
    vectorToQuaternion(vector0, vector1, aim_axis=0, up_axis=1, extrapolate=False)
    
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
        >>> vector0 = randomVector(100)                                      # random aim vector array
        >>> vector1 = randomVector(100)                                      # random up vector array
        >>> print vectorToQuaternion(vector0, vector1)                       # computes quaternions qi,qj,qk,qw from aim and up vector arrays
        >>> print vectorToQuaternion(vector0, vector1[0], extrapolate=False) # computes quaternions qi,qj,qk,qw from aim vector array and a single up vector
        >>> print vectorToQuaternion(vector0, vector1[0], extrapolate=True)  # computes quaternions qi,qj,qk,qw from aim vector array and a single up vector and extrapolate next up vector from previous solution
    """           
    
    
    vector0  = _setDimension(vector0,2)
    vector1  = _setDimension(vector1,2)
    aim_axis = _setDimension(aim_axis,1,dtype=np.int64) % 3
    up_axis  = _setDimension(up_axis,1,dtype=np.int64) % 3
    
    return _vectorToQuaternion(vector0,vector1,aim_axis,up_axis,extrapolate)


def vectorToEuler(vector0, vector1, aim_axis=0, up_axis=1, axes=XYZ, extrapolate=False):
    
    """
    vectorToEuler(vector0, vector1, aim_axis=0, up_axis=1, extrapolate=False)
    
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
        vectorToQuaternion: Converts an array of aim (vector0) and up (vector1) vectors to quaternions qi,qj,qk,qw.
        vectorToMatrix : Converts an array of aim (vector0) and up (vector1) vectors to transform matrices.
        

        Examples
        --------
        >>> vector0 = randomVector(100)                                 # random aim vector array
        >>> vector1 = randomVector(100)                                 # random up vector array
        >>> print vectorToEuler(vector0, vector1)                       # computes quaternions qi,qj,qk,qw from aim and up vector arrays
        >>> print vectorToEuler(vector0, vector1[0], extrapolate=False) # computes quaternions qi,qj,qk,qw from aim vector array and a single up vector
        >>> print vectorToEuler(vector0, vector1[0], extrapolate=True)  # computes quaternions qi,qj,qk,qw from aim vector array and a single up vector and extrapolate next up vector from previous solution
    """          
    
    vector0  = _setDimension(vector0,2)
    vector1  = _setDimension(vector1,2)
    aim_axis = _setDimension(aim_axis,1,dtype=np.int64) % 3
    up_axis  = _setDimension(up_axis,1,dtype=np.int64) % 3
    axes     = _setDimension(axes,1,dtype=np.int64)
    
    return _vectorToEuler(vector0,vector1,aim_axis,up_axis,axes,extrapolate)



def vectorCross(vector0, vector1):

    """
    vectorCross(vector0, vector1)
    
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
        vectorDot: Computes the dot product between 2 lists of vectors.
        vectorMagnitude : Computes the magnitude of a list of vectors.
        vectorNormalize : Normalizes a list of vectors.
        
        
        Examples
        --------
        >>> vector0 = randomVector(100)            # random vector array
        >>> vector1 = randomVector(100)            # random vector array
        >>> print vectorCross(vector0, vector1)    # computes the cross product between 2 lists of vectors
        >>> print vectorCross(vector0, vector1[0]) # computes the cross product between a lists of vectors and a single vector
    """         
    
    
    
    vector0 = _setDimension(vector0,2)
    vector1 = _setDimension(vector1,2)
            
    return _vectorCross(vector0, vector1)



def vectorDot(vector0, vector1):
    """
    vectorDot(vector0, vector1)
    
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
        vectorCross: Computes the cross product between 2 lists of vectors.
        vectorMagnitude : Computes the magnitude of a list of vectors.
        vectorNormalize : Normalizes a list of vectors.
        
        
        Examples
        --------
        >>> vector0 = randomVector(100)          # random vector array
        >>> vector1 = randomVector(100)          # random vector array
        >>> print vectorDot(vector0, vector1)    # computes the dot product between 2 lists of vectors
        >>> print vectorDot(vector0, vector1[0]) # computes the dot product between a lists of vectors and a single vector
    """  
    
    vector0 = _setDimension(vector0,2)
    vector1 = _setDimension(vector1,2)
            
    return _vectorDot(vector0, vector1)



def vectorMagnitude(vector):
    """
    vectorMagnitude(vector)
    
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
        vectorCross: Computes the cross product between 2 lists of vectors.
        vectorDot : Computes the dot product of a list of vectors.
        vectorNormalize : Normalizes a list of vectors.
        
        
        Examples
        --------
        >>> vector = randomVector(100)      # random vector array
        >>> print vectorMagnitude(vector)   # computes the magnitude of a list vectors
        """      
    
    vector = _setDimension(vector,2)
    
    return _vectorMagnitude(vector)



def vectorNormalize(vector):
    """
    vectorNormalize(vector)
    
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
        vectorCross: Computes the cross product between 2 lists of vectors.
        vectorDot : Computes the dot product of a list of vectors.
        vectorMagnitude : Computes the magnitude of a list of vectors.
        
        
        Examples
        --------
        >>> vector = randomVector(100)      # random vector array
        >>> print vectorNormalize(vector)   # computes the magnitude of a list vectors
        """      
    
    vector = _setDimension(vector,2)

    return _vectorNormalize(vector)



def vectorSlerp(vector0, vector1, weight=0.5):
    """
    vectorSlerp(vector0, vector1, weight=0.5)
    
        Spherical interpolation between 2 lists of vectors

        Parameters
        ----------
        vector0 : *[float, float, float]* or *[[float, float, float],...]*
            a single, or list of vectors
        
        vector1 : *[float, float, float]* or *[[float, float, float],...]*
            a single, or list of vectors
            
        weight : *[float,...]* or *float*
            a single, or list of interpolation weights (default=0.5)

            
        Returns
        -------
        vector: np.array(n,3) 
            a list of spherically interpolated vectors
            

        See Also
        --------
        vectorLerp: Linear interpolation between 2 lists of vectors.
        
        
        Examples
        --------
        >>> vector0 = randomVector(100)      # random vector array
        >>> vector1 = randomVector(100)      # random vector array
        >>> weight  = np.random.random(100)  # random float weight array
        >>> print repr(vectorSlerp(vector0, vector1, weight))    # slerp 2 lists of vectors by list of floats
        >>> print repr(vectorSlerp(vector0, vector1, weight[0])) # slerp 2 lists of vectors by same weight value
        >>> print repr(vectorSlerp(vector0[0], vector1[0], weight)) # slerp 2 vectors by list of weights
        """     
    
    vector0 = _setDimension(vector0,2)
    vector1 = _setDimension(vector1,2)
    weight  = _setDimension(weight,1)
    
    return _vectorSlerp(vector0,vector1,weight)



def vectorLerp(vector0, vector1, weight=0.5, method='linear', power=1., inverse=0.):
    """
    vectorLerp(vector0, vector1, weight=0.5, method='linear')
    
        Linear interpolation between 2 lists of vectors

        Parameters
        ----------
        vector0 : *[float, float, float]* or *[[float, float, float],...]*
            a single, or list of vectors
        
        vector1 : *[float, float, float]* or *[[float, float, float],...]*
            a single, or list of vectors
            
        weight : *[float,...]* or *float*
            a single, or list of interpolation weights (default=0.5)
            
        method : 'linear', 'multiplied' or 'cosine'
            linear, multiplied or sine interpolation
            multiplied is generally used for scale interpolation.
            cosine is used to ease in and out.
            
        power : when using the 'multiplied' method, this value is used to
            modulate the weights
            
        inverse : when using the 'multiplied' method, this value is used to
            blend between regular and inversed exponent flows
            


        Returns
        -------
        vector: np.array(n,3) 
            a list of linearly interpolated vectors
            

        See Also
        --------
        vectorSlerp: Linear interpolation between 2 lists of vectors.
        
        
        Examples
        --------
        >>> vector0 = randomVector(100)      # random vector array
        >>> vector1 = randomVector(100)      # random vector array
        >>> weight  = np.random.random(100)  # random float weight array
        >>> print repr(vectorLerp(vector0, vector1, weight))       # slerp 2 lists of vectors by list of floats
        >>> print repr(vectorLerp(vector0, vector1, weight[0]))    # slerp 2 lists of vectors by same weight value
        >>> print repr(vectorLerp(vector0[0], vector1[0], weight)) # slerp 2 vectors by list of weights
        """      
    
    vector0 = _setDimension(vector0,2)
    vector1 = _setDimension(vector1,2)
    weight  = _setDimension(weight,1)
    
    method = {'linear':0, 'multiplied':1, 'cosine':2}[method]
    if method == 2:
        method = 0
        t = np.where((weight>0.) & (weight<1.))[0]
        if weight.size:
            weight[t] = np.radians(weight[t]*180)
            weight[t] = 1-(np.cos(weight[t])+1)/2  
    
    
    elif method == 1:
        w0 = weight**abs(power)        # normal weight flow 
        w1 = 1-w0[::-1]                # reversed weight flow
        weight  = w0 + inverse*(w1-w0) # lerp between flows
        
    return np.nan_to_num(_vectorLerp(vector0,vector1,weight,method))



def vectorArcToQuaternion(vector0, vector1):
    """
    vectorArcToQuaternion(vector0, vector1)
    
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
        vectorArcToMatrix: Computes the arc between 2 lists of vectors in the form of 4x4 transform matrices.
        vectorArcToEuler:  Computes the arc between 2 lists of vectors in the form of euler angles.
     
        
        Examples
        --------
        >>> vector0 = randomVector(100)      # random vector array
        >>> vector1 = randomVector(100)      # random vector array
        >>> print repr(vectorArcToQuaternion(vector0, vector1)      # arc between 2 lists of vectors
        >>> print repr(vectorArcToQuaternion(vector0, vector1[0]))  # arc between a list of vectors and a single vector
        """          
    
    
    vector0 = _setDimension(vector0,2)
    vector1 = _setDimension(vector1,2)
            
    return _vectorArcToQuaternion(vector0, vector1)



def vectorArcToMatrix(vector0, vector1):
    """
    vectorArcToMatrix(vector0, vector1)
    
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
        vectorArcToMatrix: Computes the arc between 2 lists of vectors in the form of 4x4 transform matrices.
        vectorArcToEuler:  Computes the arc between 2 lists of vectors in the form of euler angles.
     
        
        Examples
        --------
        >>> vector0 = randomVector(100)      # random vector array
        >>> vector1 = randomVector(100)      # random vector array
        >>> print repr(vectorArcToMatrix(vector0, vector1)      # arc between 2 lists of vectors
        >>> print repr(vectorArcToMatrix(vector0, vector1[0]))  # arc between a list of vectors and a single vector
        """
    
    vector0 = _setDimension(vector0,2)
    vector1 = _setDimension(vector1,2)
            
    return _vectorArcToMatrix(vector0, vector1)



def vectorArcToEuler(vector0, vector1, axes=XYZ):
    """
    vectorArcToEuler(vector0, vector1, axes=XYZ)
    
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
        >>> vector0 = randomVector(100)      # random vector array
        >>> vector1 = randomVector(100)      # random vector array
        >>> print repr(vectorArcToEuler(vector0, vector1)      # arc between 2 lists of vectors
        >>> print repr(vectorArcToEuler(vector0, vector1[0]))  # arc between a list of vectors and a single vector
        """
    
    vector0 = _setDimension(vector0,2)
    vector1 = _setDimension(vector1,2)
    axes  = _setDimension(axes,1,dtype=np.int64)
            
    return _vectorArcToEuler(vector0, vector1, axes)




#----------------------------------------------- MATRIX MATH -----------------------------------------------#


def matrixIdentity(count):
    """ Creulertes 4x4 identity matrices
    """
    return _matrixIdentity(count)



def matrixToEuler(matrix, axes=XYZ):
    """ Converts an euler angle to a 4x4 transform matrix
    
        >>> m = randomMatrix(2)                      # make 2 random matrices
        >>> print repr(matrixToEuler(m[0],0))        # from 1 matrix   makes 1 euler angle  with xyz rotate order
        >>> print repr(matrixToEuler(m,0))           # from 2 matrices makes 2 euler angles with xyz rotate order
        >>> print repr(matrixToEuler(m,[2,0]))       # from 2 matrices makes 2 euler angles with zxy and xyz rotate orders
        >>> print repr(matrixToEuler(m[0],range(6))) # from 1 matrix makes 6 euler angles with all possible 6 rotate orders
    """
    matrix = _setDimension(matrix,3,reshape_matrix=True)
    axes = _setDimension(axes,1,dtype=np.int64)
    
    return _matrixToEuler(matrix,axes)



def matrixToQuaternion(matrix):
    """ Converts 4x4 matrix to transform matrices
    
        >>> m = randomMatrix(2)                  # make 2 random matrices
        >>> print repr(matrixToQuaternion(m[0])) # from 1 matrix make 1 quaternion
        >>> print repr(matrixToQuaternion(m))    # from 2 matrices make 2 quaternions
    """
    matrix = _setDimension(matrix,3,reshape_matrix=True)
    return _matrixToQuaternion(matrix)



def matrixNormalize(matrix):
    """ Normalizes the rotation component of transform matrices
    """
    matrix = _setDimension(matrix,3,reshape_matrix=True)
    return _matrixNormalize(matrix)



def matrixInverse(matrix):
    """ Inverse a list of transform matrices
    """
    matrix = _setDimension(matrix,3,reshape_matrix=True)
    
    return _matrixInverse(matrix)
    
   
    
def matrixMultiply(matrix0, matrix1):
    """ Multiplies 2 lists of matrices
    """
    
    matrix0 = _setDimension(matrix0,3,reshape_matrix=True)
    matrix1 = _setDimension(matrix1,3,reshape_matrix=True)

    return _matrixMultiply(matrix0,matrix1)



def matrixSlerp(matrix0, matrix1, weight=0.5):
    """
    matrixSlerp(matrix0, matrix1, weight=0.5)
    
        Performs a spherical interpolation between two lists of transform matrices

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
            

        See Also
        --------
        eulerSlerp : Performs a spherical interpolation between two lists of euler angles.
        quaternionSlerp: Performs a spherical interpolation between two lists of quaternions.
        vectorSlerp : Performs a spherical interpolation between two lists of vectors.
        

        Examples
        --------
        >>> matrix0 = randomMatrix(100)                # init quat0
        >>> matrix1 = randomMatrix(100)                # init quat1
        >>> print repr(matrixSlerp(matrix0,matrix1)    # get the halfway point between the two lists
        >>> print repr(matrixSlerp(matrix0,matrix1[0]) # get the halfway point between the all items of matrix0 and the first item of matrix1

    """    
    
    matrix0 = _setDimension(matrix0,3,reshape_matrix=True)
    matrix1 = _setDimension(matrix1,3,reshape_matrix=True)
    weight  = _setDimension(weight,1)

    return _matrixSlerp(matrix0,matrix1,weight)



def matrixInterpolate(matrix0, matrix1, weight=0.5):
    """
    matrixInterpolate(matrix0, matrix1, weight=0.5)
    
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
        >>> matrix0 = randomMatrix(100)                # init matrix0
        >>> matrix1 = randomMatrix(100)                # init matrix1
        >>> print repr(matrixSlerp(matrix0,matrix1)    # get the halfway point between the two lists
        >>> print repr(matrixSlerp(matrix0,matrix1[0]) # get the halfway point between the all items of matrix0 and the first item of matrix1

    """    
    
    # Set expected dimensions
    matrix0 = _setDimension(matrix0,3,reshape_matrix=True)
    matrix1 = _setDimension(matrix1,3,reshape_matrix=True)
    weight  = _setDimension(weight,1)
    
    # Grab scale components
    scale0 = inner1d(matrix0[:,:3,:3],matrix0[:,:3,:3]) ** 0.5
    scale1 = inner1d(matrix1[:,:3,:3],matrix1[:,:3,:3]) ** 0.5
    
    # Normalize matrices and interpolate rotation
    matrix0[:,0,:3] /= scale0[:,0][:,None]
    matrix0[:,1,:3] /= scale0[:,1][:,None]
    matrix0[:,2,:3] /= scale0[:,2][:,None]
    
    matrix1[:,0,:3] /= scale1[:,0][:,None]
    matrix1[:,1,:3] /= scale1[:,1][:,None]
    matrix1[:,2,:3] /= scale1[:,2][:,None]      
    
    matrix = _matrixSlerp(matrix0,matrix1,weight)    
    
    
    # Interpolate scale
    scale = _vectorLerp(scale0,scale1,weight,1)
    
    
    # Scale interpolated matrices
    matrix[:,0,:3] *= scale[:,0][:,None]
    matrix[:,1,:3] *= scale[:,1][:,None]
    matrix[:,2,:3] *= scale[:,2][:,None]
    
    
    # Interpolate position
    matrix[:,3,:3] = _vectorLerp(matrix0[:,3,:3],matrix1[:,3,:3],weight,0)    
    

    return matrix



def matrixLocal(matrix, parent_matrix):
    """ Multiplies 2 lists of matrices
    """
    
    matrix        = _setDimension(matrix,3,reshape_matrix=True)
    parent_matrix = _setDimension(parent_matrix,3,reshape_matrix=True)

    return _matrixLocal(matrix,parent_matrix)



def matrixPointMultiply(point,matrix):
    """ Transforms a list of points by a list of matrices
    """
    
    point  = _setDimension(point,2)
    matrix = _setDimension(matrix,3,reshape_matrix=True)
    return _matrixPointMultiply(point,matrix)



#----------------------------------------------- QUATERNION MATH -----------------------------------------------#


def quaternionSlerp(quat0, quat1, weight=0.5):
    """
    quaternionSlerp(quat0, quat1, weight=0.5)
    
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
        eulerSlerp : Performs a spherical interpolation between two lists of euler angles.
        matrixSlerp: Performs a spherical interpolation between two lists of 4x4 matrices.
        vectorSlerp : Performs a spherical interpolation between two lists of vectors.
        

        Examples
        --------
        >>> quat0 = randomQuaternion(100)               # init quat0
        >>> quat1 = randomQuaternion(100)               # init quat1
        >>> print repr(quaternionSlerp(quat0,quat1)    # get the halfway point between the two lists
        >>> print repr(quaternionSlerp(quat0,quat1[0]) # get the halfway point between the all items of quat0 and the first item of quat1

    """        
    quat0  = _setDimension(quat0,2)
    quat1  = _setDimension(quat1,2)
    weight = _setDimension(weight,1)

    return _quaternionSlerp(quat0,quat1,weight)



def quaternionDot(quat0, quat1):
    """ Calculates dot product between two quaternions
    """
    quat0  = _setDimension(quat0,2)
    quat1  = _setDimension(quat1,2)
    
    return _quaternionDot(quat0,quat1)



def quaternionConjugate(quat):
    """ Calculates dot product between two quaternions
    """
    quat  = _setDimension(quat,2)
    
    return _quaternionConjugate(quat)



def quaternionInverse(quat):
    """ Calculates dot product between two quaternions
    """
    quat  = _setDimension(quat,2)

    return _quaternionInverse(quat)



def quaternionNegate(quat):
    """ Calculates dot product between two quaternions
    """
    quat  = _setDimension(quat,2)

    return _quaternionNegate(quat)



def quaternionMultiply(quat0, quat1):
    """ Multiplies two quaternions
    """
    quat0  = _setDimension(quat0,2)
    quat1  = _setDimension(quat1,2)
    
    return _quaternionMultiply(quat0,quat1)



def quaternionAdd(quat0, quat1):
    """ Adds two quaternions
    """
    quat0  = _setDimension(quat0,2)
    quat1  = _setDimension(quat1,2)
    
    return _quaternionAdd(quat0,quat1)



def quaternionSub(quat0, quat1):
    """ Subtracts two quaternions
    """
    quat0  = _setDimension(quat0,2)
    quat1  = _setDimension(quat1,2)
    
    return _quaternionSub(quat0,quat1)



def quaternionToMatrix(quat):
    """ Converts list of quaternions qi,qj,qk,qw to 4x4 matrices
    
        >>> q = randomQuaternion(2)              # make 2 random quaternions
        >>> print repr(quaternionToMatrix(q[0])) # from 1 quaternion make matrix
        >>> print repr(quaternionToMatrix(q))    # from 2 quaternions make matrices
    """
    quat = _setDimension(quat,2)
    return _quaternionToMatrix(quat)
    
    
    
def quaternionNormalize(quat):
    """ Normalizes a quaternion
    """
    quat = _setDimension(quat,2)
    return _quaternionNormalize(quat)

    
    
def quaternionToEuler(quat, axes=XYZ):
    """ Converts quaternions qi,qj,qk,qw to euler angles
    
        >>> q = randomEuler(2)                   # make 2 random quaternions
        >>> print repr(quaternionToMatrix(q[0])) # from 1 quaternion make 1 matrix
        >>> print repr(quaternionToMatrix(q))    # from 2 quaternions make 2 matrices
    """
    quat = _setDimension(quat,2)
    axes = _setDimension(axes,1,dtype=np.int64)
    
    return _matrixToEuler(_quaternionToMatrix(quat), axes)



def axisAngleToQuaternion(axis, angle=0.):
    """
    axisAngleToQuaternion(axis, angle=0.)
    
        Computes a list of quaternions qi,qj,qk,qw from lists of axes and angles

        Parameters
        ----------
        axis : *[float, float, float]* or *[[float, float, float],...]*
            a single, or list vector axis.
            
        angle : *float* or *[float, ...]*
            a single, or list of angles to rotate about the given axis.
            
        axes : *XYZ, YZX, ZXY, XZY, YXZ, ZYX*, *int* or *[int,...]*
            corresponding rotate orders. default = XYZ
            
            
        Returns
        -------
        euler: np.array(n,3)
            a list of euler angles
            

        See Also
        --------
        axisAngleToMatrix : Converts axis angles to matrices.
        axisAngleToQuaternion : Converts axis angles to quaternions qi,qj,qk,qw.
        

        Examples
        --------
        >>> axis = randomVector(2)                      # make 2 random axis vectors
        >>> angles = randomEuler(2)[:,0]                # make 2 random angles
        >>> print repr(axisAngleToEuler(axis,angles)    # from axis angles from 2 lists
        >>> print repr(axisAngleToEuler(axis,angles[0]) # from axis angles from axis and first angle

    """            
    
    axis  = _setDimension(axis,2)
    angle = _setDimension(angle,1)    
    
    return _axisAngleToQuaternion(axis,angle)



def axisAngleToMatrix(axis, angle=0.):
    """ Computes a list of quaternions qi,qj,qk,qw from lists of axes and angles
    """
    axis  = _setDimension(axis,2)
    angle = _setDimension(angle,1)    
    
    return _axisAngleToMatrix(axis,angle)



def axisAngleToEuler(axis, angle=0., axes=XYZ):
    """
    axisAngleToEuler(euler,axes=XYZ)
    
        Computes a list of quaternions qi,qj,qk,qw from lists of axes and angles.

        Parameters
        ----------
        axis : *[float, float, float]* or *[[float, float, float],...]*
            a single, or list vector axis.
            
        angle : *float* or *[float, ...]*
            a single, or list of angles to rotate about the given axis.
            
        axes : *XYZ, YZX, ZXY, XZY, YXZ, ZYX*, *int* or *[int,...]*
            corresponding rotate orders. default = XYZ
            
            
        Returns
        -------
        euler: np.array(n,3)
            a list of euler angles
            

        See Also
        --------
        axisAngleToMatrix : Converts axis angles to matrices.
        axisAngleToQuaternion : Converts axis angles to quaternions qi,qj,qk,qw.
        

        Examples
        --------
        >>> axis = randomVector(2)                      # make 2 random axis vectors
        >>> angles = randomEuler(2)[:,0]                # make 2 random angles
        >>> print repr(axisAngleToEuler(axis,angles)    # from axis angles from 2 lists
        >>> print repr(axisAngleToEuler(axis,angles[0]) # from axis angles from axis and first angle

    """        
    
    axis  = _setDimension(axis,2)
    angle = _setDimension(angle,1)    
    axes  = _setDimension(axes,1,dtype=np.int64)
    
    return _axisAngleToEuler(axis,angle,axes)




#----------------------------------------------- RANDOM GENERATORS -----------------------------------------------#


def randomVector(n, seed=None, normalize=True):
    """ Computes a list of random normalized xyz axes
    """
    np.random.seed(seed)
    if normalize:
        return _vectorNormalize(1 - np.random.random((n,3,))*2)
    
    return 1 - np.random.random((n,3,))*2
        


def randomAngle(n, seed=None):
    """ Computes a list of random angles
    """
    np.random.seed(seed)
    return np.radians(360 - np.random.random(n)*720)



def randomEuler(n, seed=None):
    """ Computes a list of random euler angles
    """
    np.random.seed(seed)
    return np.radians(360 - np.random.random((n,3))*720)



def randomQuaternion(n, seed=None):
    """ Computes a list of random quaternions qi,qj,qk,qw
    """
    return _eulerToQuaternion(randomEuler(n=n, seed=seed),np.array([0]))



def randomMatrix(n, seed=None, random_position=False):
    """ Computes a list of random 4x4 rotation matrices
    """
    M = _eulerToMatrix(randomEuler(n=n, seed=seed),np.array([0]))
    if random_position:
        M[:,3,:3] = 1 - np.random.random((n,3,))*2

    return M


