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
from .. import euler
from .. import matrix
from .. import quaternion
from .. import vector
import pickle

class TestEuler(unittest.TestCase):

    def testToEuler(self):
        ea = euler.random(10**6)
        ea_ = euler.to_euler(ea, 0, 3)  # change to rotate order to 3
        ea_ = euler.to_euler(ea_, 3, 0) # bring back to rotate order to 0

        self.assertEqual(np.allclose(euler.to_matrix(ea, 0), euler.to_matrix(ea_, 0)), True)

        
    def testToMatrix(self):
        ea = euler.random(10**6)
        M  = euler.to_matrix(ea, 0)
        ea_ = matrix.to_euler(M, 0)
        
        self.assertEqual(np.allclose(euler.to_matrix(ea), euler.to_matrix(ea_)), True)
        
        
    def testToQuaternion(self):
        ea = euler.random(10**6)
        Q  = euler.to_quaternion(ea, 0)
        ea_ = quaternion.to_euler(Q, 0)
        
        self.assertEqual(np.allclose(euler.to_matrix(ea), euler.to_matrix(ea_)), True)
        
        
    def testSlerp(self):
        ea0 = euler.random(10**6)
        ea1 = euler.random(10**6)
        w = np.random.random(10**6)

        # slerp ea0 to ea1
        forward = euler.slerp(ea0, ea1, w)
        
        # do the opposite
        backward = euler.slerp(ea1, ea0, 1-w)
        
        
        self.assertEqual(np.allclose(euler.to_matrix(forward), euler.to_matrix(backward)), True)    