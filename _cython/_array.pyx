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



# Imports
import numpy as np
cimport cython
cimport numpy as np



# 1D array, example: floats [0,1,2,3,4,5]
@cython.boundscheck(False)
@cython.wraparound(False)
def convert1D(data):
    cdef unsigned int  I
    cdef unsigned int  i
    I = len(data)
    cdef np.ndarray[double,ndim=1] new = np.empty(I)
    
    for i in range(I):
        new[i] = data[i]

    return new
    

# 2D array, example: points [[0,0,0,1],
#                            [1,1,1,1],
#                            [2,2,2,1]]
@cython.boundscheck(False)
@cython.wraparound(False)
def convert2D(data):
    cdef unsigned int  I,J
    cdef unsigned int  i,j
    I = len(data)
    J = len(data[0])
    cdef np.ndarray[double,ndim=2] new = np.empty((I,J))

    for i in range(I):
        for j in range(J):
            new[i,j] = data[i][j]

    return new


# 3D array, example: matrices [[[0,0,0],
#                               [1,1,1],
#                               [2,2,2]],
#                              [[3,3,3],
#                               [4,4,4],
#                               [5,5,5]]]
@cython.boundscheck(False)
@cython.wraparound(False)
def convert3D(data):
    cdef unsigned int  I,J,K
    cdef unsigned int  i,j,k
    I = len(data)
    J = len(data[0])
    K = len(data[0][0])
    cdef np.ndarray[double,ndim=3] new = np.empty((I,J,K))
    
    for i in range(I):
        for j in range(J):
            for k in range(K):
                new[i,j,k] = data[i][j][k]

    return new



# why? why not!
@cython.boundscheck(False)
@cython.wraparound(False)
def convert4D(data):
    cdef unsigned int  I,J,K,L
    cdef unsigned int  i,j,k,l
    I = len(data)
    J = len(data[0])
    K = len(data[0][0])
    L = len(data[0][0][0])
    cdef np.ndarray[double,ndim=4] new = np.empty((I,J,K,L))
    
    for i in range(I):
        for j in range(J):
            for k in range(K):
                for l in range(L):
                    new[i,j,k,l] = data[i][j][k][l]

    return new


# sure, one more!
@cython.boundscheck(False)
@cython.wraparound(False)
def convert5D(data):
    cdef unsigned int  I,J,K,L,M
    cdef unsigned int  i,j,k,l,m
    I = len(data)
    J = len(data[0])
    K = len(data[0][0])
    L = len(data[0][0][0])
    M = len(data[0][0][0][0])
    cdef np.ndarray[double,ndim=5] new = np.empty((I,J,K,L,M))
    
    for i in range(I):
        for j in range(J):
            for k in range(K):
                for l in range(L):
                    for m in range(M):
                        new[i,j,k,l,m] = data[i][j][k][l][m]

    return new
    
    
# k that's enough!
@cython.boundscheck(False)
@cython.wraparound(False)
def convert6D(data):
    cdef unsigned int  I,J,K,L,M,N
    cdef unsigned int  i,j,k,l,m,n
    I = len(data)
    J = len(data[0])
    K = len(data[0][0])
    L = len(data[0][0][0])
    M = len(data[0][0][0][0])
    N = len(data[0][0][0][0][0])
    
    cdef np.ndarray[double,ndim=6] new = np.empty((I,J,K,L,M,N))
    
    for i in range(I):
        for j in range(J):
            for k in range(K):
                for l in range(L):
                    for m in range(M):
                        for n in range(N):
                            new[i,j,k,l,m,n] = data[i][j][k][l][m][n]

    return new