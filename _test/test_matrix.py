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
from .. import matrix, euler, quaternion


class TestMatrix(unittest.TestCase):
    
    def testRandom(self):
        M = matrix.random(10**6)
        x = np.einsum('...i,...i', M[:, 0], M[:, 0]) ** 0.5
        y = np.einsum('...i,...i', M[:, 1], M[:, 1]) ** 0.5
        z = np.einsum('...i,...i', M[:, 2], M[:, 2]) ** 0.5
        ones = np.ones(10**6)

        self.assertEqual(np.allclose(x, ones), True)
        self.assertEqual(np.allclose(y, ones), True)
        self.assertEqual(np.allclose(z, ones), True)
  

    def testInterpolate(self):
        M0 = matrix.random(10**6)
        M1 = matrix.random(10**6)
        w = np.random.random(10**6)
    
        # inerp M0 to M1
        forward = matrix.interpolate(M0, M1, w)
    
        # do the opposite
        backward = matrix.interpolate(M1, M0, 1-w)
        
        self.assertEqual(np.allclose(forward, backward), True)
        
        
    def testToEuler(self):
        M = matrix.random(10**6)
        ea = matrix.to_euler(M)
        M_ = euler.to_matrix(ea)
        
        self.assertEqual(np.allclose(M, M_), True)
        
        
    def testToQuaternion(self):
        M = matrix.random(10**6)
        Q = matrix.to_quaternion(M)
        M_ = quaternion.to_matrix(Q)
        
        self.assertEqual(np.allclose(M, M_), True)    
        
        
    def testSlerp(self):
        M0 = matrix.random(10**6)
        M1 = matrix.random(10**6)
        w = np.random.random(10**6)
    
        # inerp M0 to M1
        forward = matrix.slerp(M0, M1, w)
    
        # do the opposite
        backward = matrix.slerp(M1, M0, 1-w)
        
        self.assertEqual(np.allclose(forward, backward), True)
        
        
    def testNormalize(self):
        M = matrix.random(10**6) * 0.1
        ones = np.ones(10**6)

        M_ = matrix.normalize(M)
        x = np.einsum('...i,...i', M_[:, 0], M_[:, 0]) ** 0.5
        y = np.einsum('...i,...i', M_[:, 1], M_[:, 1]) ** 0.5
        z = np.einsum('...i,...i', M_[:, 2], M_[:, 2]) ** 0.5
              
        self.assertEqual(np.allclose(x, ones), True)
        self.assertEqual(np.allclose(y, ones), True)
        self.assertEqual(np.allclose(z, ones), True)        
        
        
    def testLocal(self):
        M = matrix.random(10**6)
        M[:, 3, :3] = np.random.random((10**6, 3))
        
        P = matrix.identity(10**6)
        P[:, 3, :3] = np.random.random((10**6, 3))
        
        L = matrix.local(M, P)
        
        delta = M[:, 3, :3] - P[:, 3, :3] 

        self.assertEqual(np.allclose(L[:, 3, :3], delta), True)        
                
                
    def testMultiply(self):
        M0 = matrix.identity(10**6)
        M0[:, 3, :3] = np.random.random((10**6, 3))
        
        M1 = matrix.identity(10**6)
        M1[:, 3, :3] = np.random.random((10**6, 3))
        
        test = matrix.multiply(M1, M0)
        
        self.assertEqual(np.allclose(test[:, 3, :3], M0[:, 3, :3] + M1[:, 3, :3]), True)
        
        
    def testPoint(self):
        M0 = matrix.identity(10**6)
        M0[:, 3, :3] = np.random.random((10**6, 3))
        
        p = np.random.random((10**6, 3))
        test = matrix.point(p, M0)
        self.assertEqual(np.allclose(test, M0[:, 3, :3] + p), True)    