
import numpy as np
import sys
import cProfile
import pstats
import re
import statistics
import tempfile
import os

from io import StringIO
from inspect import getmembers, isbuiltin
from . import _array

from numba import njit, prange




# dynamically map what's available int the cythonized _array module
methods = dict(getmembers(sys.modules['_array'], isbuiltin))
CONVERSION_DEPTH_MAP = dict(zip(range(1, len(methods) + 1), methods.values()))



def _is_sequence(obj):
    
    """ tests if given input is a sequence """
    
    if isinstance(obj, str):
        return False
    try:
        len(obj)
        if isinstance(obj, dict):
            return False
    except Exception:
        return False
    
    return True



def _get_depth(obj, depth=0):
    
    """ returns the depth of a sequence and its data type """
    
    if _is_sequence(obj):
        depth += 1
        return _get_depth(obj[0], depth)
    
    return depth, type(obj)
    
    

def _to_numpy(data, dtype=None):
    
    """ converts to a numpy array via cython """
    
    # already a numpy array?
    if isinstance(data, (np.ndarray)):
        return np.asarray(data, dtype=dtype)
    
    # get the array depth and data type
    depth, data_type = _get_depth(data)
    
    # if no data type specified, inherit the type 
    # of the first entry of the leaf array element
    if dtype is None:
        dtype = data_type

    # if depth == 0, (depth 0)
    if depth == 0:
        return np.asarray(data, dtype=dtype)
    
    # convert the data as requested to the desired data type
    return np.asarray(CONVERSION_DEPTH_MAP[depth](data), dtype=dtype)



#----------------------------- ARRAY RESIZERS -----------------------------#

@njit(fastmath=True, parallel=True)
def resize1D(a, new_depth):
    delta     = new_depth - a.shape[0]
    new_array = np.empty((new_depth), a.dtype)

    # fill initial array
    for i in prange(min(a.shape[0], new_depth)):
        new_array[i] = a[i]

    # dplicate last entry for fill
    if delta > 0:
        for i in prange(a.shape[0], a.shape[0]+delta):
            new_array[i] = a[a.shape[0]-1]

    return new_array   



@njit(fastmath=True, parallel=True)
def resize2D(a, new_depth):
    delta     = new_depth - a.shape[0]
    new_array = np.empty((new_depth, a.shape[1]), a.dtype)

    # fill initial array
    for i in prange(min(a.shape[0], new_depth)):
        for j in range(a.shape[1]):
            new_array[i,j] = a[i,j]


    # dplicate last entry for fill
    if delta > 0:
        for i in prange(a.shape[0], a.shape[0]+delta):
            for j in range(a.shape[1]):
                new_array[i,j] = a[a.shape[0]-1,j]

    return new_array   


@njit(fastmath=True, parallel=True)
def resize3D(a, new_depth):
    delta     = new_depth - a.shape[0]
    new_array = np.empty((new_depth, a.shape[1], a.shape[2]), a.dtype)

    # fill initial array
    for i in prange(min(a.shape[0], new_depth)):
        for j in range(a.shape[1]):
            for k in range(a.shape[2]):
                new_array[i, j, k] = a[i, j, k]


    # dplicate last entry for fill
    if delta > 0:
        for i in prange(a.shape[0], a.shape[0]+delta):
            for j in range(a.shape[1]):
                for k in range(a.shape[2]):            
                    new_array[i, j, k] = a[a.shape[0]-1, j, k]

    return new_array   


RESIZE_MAP = {1: resize1D, 2: resize2D, 3: resize3D}



#----------------------------------------------- UTILS -----------------------------------------------#

def _setDimension(data, ndim=1, dtype=np.float64, reshape_matrix=False):
    """ Sets input data to expected dimension
    """
    #data = np.asarray(data, dtype=dtype)
    data = _to_numpy(data, dtype=dtype)
    
    while data.ndim < ndim:
        data = data[np.newaxis]
        
    # For when matrices are given as lists of 16 floats
    if reshape_matrix:
        if data.shape[-1] == 16:
            data = data.reshape(-1,4,4)

    return data


def _matchDepth(*data):
    """ Sets given data to the highest depth.
        It is assumed all entries are already numpy arrays.
    """
    count   = [len(d) for d in data]
    highest = max(count)
    matched = list(data)
    
    for i in range(len(count)):
        if count[i] > 0 and count[i] < highest:
            #matched[i] = np.concatenate((data[i],) + (np.repeat([data[i][-1]],highest-count[i],axis=0),))
            matched[i] = RESIZE_MAP[matched[i].ndim](matched[i], highest)
        
    return matched


def profile(cmd, n=100):
    """
    simple speed profiler
    """
    def _profile(cmd):
        tmp_dir = tempfile._get_default_tempdir()
        statsfile = os.path.join(tmp_dir, 'statsfile')
        
        cProfile.run(cmd, statsfile)
        
        stream = StringIO()
        stats = pstats.Stats(statsfile, stream=stream)
        stats.print_stats()
        stream = stream.getvalue()    
        values = re.findall(r'[\d\.\d]+', stream.splitlines()[2])
        return float(values[-1])
    
    vals = []
    for i in range(n):
        vals.append(_profile(cmd))
        
    return statistics.median(vals)
