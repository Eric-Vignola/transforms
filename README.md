# Transforms
A Numeric Transform operations toolset

## About
This module provides a toolset to do fast matrix/quaternion/vector operations.
For practical reasons, most functions are written to accept asymmetrical inputs,
meaning that calculations will be made element wise even if array sizes differ,
and reuse the last entry of the smallest input to complete the calculation.

## Requirements
Numpy, Scipy and Numba python modules.

## Author
* **Eric Vignola** (eric.vignola@gmail.com)

## Example
```
import numpy as np
from transforms import vector

# declare two arrays of 10 million vectors
V0 = vector.random((10**7,3))
V1 = vector.random((10**7,3))

# slerp 10 million vectors 1:1 element wise, half way
slerp = vector.slerp(V0, V1, 0.5)

# slerp 10 million vectors to a common vector, half way
slerp = vector.slerp(V0, V1[0], 0.5)

# slerp 10 million vectors 1:1 element wise, with a random ratio for each
blend = np.random.random(10**7)
slerp = vector.slerp(V0, V1, blend)
```


## Supported Functions
```
axisAngleToEuler       quaternionAdd        vectorArcToEuler
axisAngleToMatrix      quaternionConjugate  vectorArcToMatrix
axisAngleToQuaternion  quaternionDot        vectorArcToQuaternion
eulerSlerp             quaternionInverse    vectorCross
eulerToEuler           quaternionMultiply   vectorDot
eulerToMatrix          quaternionNegate     vectorLerp
eulerToQuaternion      quaternionNormalize  vectorMagnitude
matrixIdentity         quaternionSlerp      vectorNormalize
matrixInterpolate      quaternionSub        vectorSlerp
matrixInverse          quaternionToEuler    vectorToEuler
matrixLocal            quaternionToMatrix   vectorToMatrix
matrixMultiply         randomAngle          vectorToQuaternion
matrixNormalize        randomEuler
matrixPointMultiply    randomMatrix
matrixSlerp            randomQuaternion
matrixToEuler          randomVector
matrixToQuaternion
```

## License
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

