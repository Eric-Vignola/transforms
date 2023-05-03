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
from .. import euler, quaternion


class TestQuaternion(unittest.TestCase):
    
    def testRandom(self):
        Q = quaternion.random(10**6)
        M = quaternion.to_matrix(Q)

        x = np.einsum('...i,...i', M[:, 0], M[:, 0]) ** 0.5
        y = np.einsum('...i,...i', M[:, 1], M[:, 1]) ** 0.5
        z = np.einsum('...i,...i', M[:, 2], M[:, 2]) ** 0.5
        ones = np.ones(10**6)

        self.assertEqual(np.allclose(x, ones), True)
        self.assertEqual(np.allclose(y, ones), True)
        self.assertEqual(np.allclose(z, ones), True)
        
        
  
    def testToEuler(self):
        Q = quaternion.random(10**6)
        M = quaternion.to_matrix(Q)
        
        ea = quaternion.to_euler(Q)
        M_ = euler.to_matrix(ea)
        self.assertEqual(np.allclose(M, M_), True)
        

    def testToMatrix(self):
        Q = quaternion.random(10**6)
        M = quaternion.to_matrix(Q)
        
        x = np.einsum('...i,...i', M[:, 0], M[:, 0]) ** 0.5
        y = np.einsum('...i,...i', M[:, 1], M[:, 1]) ** 0.5
        z = np.einsum('...i,...i', M[:, 2], M[:, 2]) ** 0.5
        ones = np.ones(10**6)

        self.assertEqual(np.allclose(x, ones), True)
        self.assertEqual(np.allclose(y, ones), True)
        self.assertEqual(np.allclose(z, ones), True)
        


    def testSlerp(self):
        Q0 = quaternion.random(10**6)
        Q1 = quaternion.random(10**6)
        w = np.random.random(10**6)

        # slerp ea0 to ea1
        forward = quaternion.slerp(Q0, Q1, w)
        
        # do the opposite
        backward = quaternion.slerp(Q1, Q0, 1-w)
        
        self.assertEqual(np.allclose(quaternion.to_matrix(forward), quaternion.to_matrix(backward)), True)
        
        
    def testNormalize(self):
        Q = quaternion.random(10**6) * 0.1
        Q_ = quaternion.normalize(Q)
        
        ones = np.ones(10**6)
        
        mag = np.einsum('...i,...i', Q_, Q_) ** 0.5
        self.assertEqual(np.allclose(mag, ones), True)
        
        
    def testNegate(self):
        Q = quaternion.random(10**6)
        Q_ = quaternion.negate(Q)

        self.assertEqual(np.allclose(Q_, -Q), True)
        
        
        
    def testConjugate(self):
        Q = quaternion.random(10**6)
        Q_ = quaternion.conjugate(Q)
        
        test = np.array(Q)
        test[:, :3] *= -1

        self.assertEqual(np.allclose(Q_, test), True)
        
        
    def testInverse(self):
        Q = quaternion.random(10**6)
        Q_ = quaternion.inverse(Q)
        
        test = np.array(Q)
        test[:, :3] *= -1
        test /= np.einsum('...i,...i', Q, Q)[:, None]

        self.assertEqual(np.allclose(Q_, test), True)    
        
        
    def testSub(self):
        Q0 = quaternion.random(10**6)
        Q1 = quaternion.random(10**6)
        Q_ = quaternion.sub(Q0, Q1)
        
        self.assertEqual(np.allclose(Q_, Q0-Q1), True)
        
        
    def testMultiply(self):
        Q0 = np.zeros((10**6, 4))
        Q0[:, 3] = 1.
        
        Q1 = quaternion.random(10**6)
        Q_ = quaternion.multiply(Q0, Q1)
        
        self.assertEqual(np.allclose(Q_, Q1), True)
        
        
    def testDot(self):
        Q0 = quaternion.random(10**6)
        Q1 = quaternion.random(10**6)
    
        dot = quaternion.dot(Q0, Q1)
        dot_ = np.einsum('...i,...i', Q0, Q1)
        
        self.assertEqual(np.allclose(dot, dot_), True)
        
        