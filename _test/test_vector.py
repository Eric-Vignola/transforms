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

import unittest
import numpy as np
from .. import vector


class TestVector(unittest.TestCase):

    def testRandom(self):
        V = vector.random(10**6, normalize=True)
        mag = np.einsum('...i,...i', V, V) ** 0.5
        ones = np.ones(10**6)
        
        self.assertEqual(np.allclose(mag, ones), True)
        self.assertEqual(not np.allclose(V[0], V[1]), True)    


    def testNormalize(self):
        V = vector.random(10**6, normalize=False)
        mag = np.einsum('...i,...i', V, V) ** 0.5
        ones = np.ones(10**6)
        
        self.assertEqual(np.allclose(mag, ones), False)
        
        V = vector.normalize(V)
        mag = np.einsum('...i,...i', V, V) ** 0.5
        self.assertEqual(np.allclose(mag, ones), True)
        
        
    def testMagnitude(self):
        V = vector.random(10**6, normalize=False)
        mag0 = np.einsum('...i,...i', V, V) ** 0.5
        mag1 = vector.magnitude(V)
        self.assertEqual(np.allclose(mag0, mag1), True)     


    def testLerp(self):
        V0 = vector.random(10**6, normalize=False)
        V1 = vector.random(10**6, normalize=False)
        w = np.random.random(10**6)
        
        V = vector.lerp(V0, V1, w)
        
        mag0 = vector.magnitude(V1-V0)
        mag1 = vector.magnitude(V-V0)
        ratio = mag1 / mag0
        
        self.assertEqual(np.allclose(w, ratio), True)     


    def testSlerp(self):
        V0 = vector.random(10**6, normalize=True, seed=0)
        V1 = vector.random(10**6, normalize=True)  
        w = np.random.random(10**6) * 0.1
        
        V0 = np.array([1, 0, 0])
        V1 = np.array([0, 1, 0])
        w = np.random.random(10**6)
        
        V = vector.slerp(V0, V1, w)
        
        ang0 = np.arccos(np.clip(np.einsum('...i,...i', V1, V0), -1.0, 1.0))
        ang1 = np.arccos(np.clip(np.einsum('...i,...i', V, V0), -1.0, 1.0))
        
        self.assertEqual(np.allclose((ang1/w), ang0), True)


    def testDot(self):
        V0 = vector.random(10**6, normalize=False)
        V1 = vector.random(10**6, normalize=True)

        dot0 = vector.dot(V0, V1)
        dot1 = np.einsum('...i,...i', V1, V0)

        self.assertEqual(np.allclose(dot0, dot1), True)


    def testCross(self):
        V0 = vector.random(10**6, normalize=True)
        V1 = vector.random(10**6, normalize=True)
        V0 = np.random.random((10**6, 3))
        V1 = np.random.random((10**6, 3))

        cross0 = vector.cross(V0, V1)
        cross1 = np.cross(V0, V1)

        self.assertEqual(np.allclose(cross0, cross1), True)
        
        
    def testAngle(self):
        V0 = vector.random(500, normalize=True)
        V1 = vector.random(500, normalize=True)
        
        V0 = [ 0.84041616, -0.38676315,  0.37962473]
        V1 = [-0.07168087,  0.24989775, -0.96561533]

        ang0 = vector.angle(V0, V1)
        ang1 = np.arccos(np.clip(np.einsum('...i,...i', V0, V1), -1.0, 1.0))

        self.assertEqual(np.allclose(ang0, ang1), True)      


    def testArcToEuler(self):
        arc = np.degrees(vector.arc_to_euler([1, 0, 0], [0, 1, 0]))
        self.assertEqual(np.allclose(arc, [0, 0, 90]), True)      
        
        
    def testToEuler(self):
        x = np.array([1, 0, 0])
        y = np.array([0, 1, 0])
        eu = vector.to_euler(x, y)
        self.assertEqual(np.allclose(eu, [0, 0, 0]), True)
        
    def testToMatrix(self):
        x = np.array([1, 0, 0])
        y = np.array([0, 1, 0])
        M = vector.to_matrix(x, y)
        
        self.assertEqual(np.allclose(M, [[[1., 0., 0., 0.],
                                          [0., 1., 0., 0.],
                                          [0., 0., 1., 0.],
                                          [0., 0., 0., 1.]]]), True)
        
        
    def testToQuaternion(self):
        x = np.array([1, 0, 0])
        y = np.array([0, 1, 0])
        Q = vector.to_quaternion(x, y)
        
        self.assertEqual(np.allclose(Q, [[0., 0., 0., 1.]]), True)
        
        
        
    def testArcToEuler(self):
        x = np.array([1, 0, 0])
        y = np.array([0, 1, 0])
        angle = np.degrees(vector.arc_to_euler(x, y))

        self.assertEqual(np.allclose(angle, [[0, 0, 90]]), True)
        
        
    def testArcToMatrix(self):
        x = np.array([1, 0, 0])
        y = np.array([1, 0, 0])
        M = vector.arc_to_matrix(x, y)

        self.assertEqual(np.allclose(M, [[[1., 0., 0., 0.],
                                          [0., 1., 0., 0.],
                                          [0., 0., 1., 0.],
                                          [0., 0., 0., 1.]]]), True)
        
        
    def testArcToQuaternion(self):
        x = np.array([1, 0, 0])
        y = np.array([1, 0, 0])
        Q = vector.arc_to_quaternion(x, y)
        
        self.assertEqual(np.allclose(Q, [[0., 0., 0., 1.]]), True)     