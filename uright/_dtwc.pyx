from libc.math cimport sqrt
import numpy as np
cimport numpy as np
cimport cython

import inkutils

cdef int _X_IDX = inkutils.INK_STRUCT['X_IDX']
cdef int _Y_IDX = inkutils.INK_STRUCT['Y_IDX']
cdef int _DX_IDX = inkutils.INK_STRUCT['DX_IDX']
cdef int _DY_IDX = inkutils.INK_STRUCT['DY_IDX']
cdef int _PU_IDX = inkutils.INK_STRUCT['PU_IDX']

ctypedef np.float64_t dtype_t

cdef dtype_t _MIN_STDEV = 1e-5

@cython.boundscheck(False)
@cython.wraparound(False)
def _compute_pairwise_distance(np.ndarray[dtype_t, ndim=1] p not None, 
                               np.ndarray[dtype_t, ndim=1] q not None,
                               dtype_t alpha, 
                               dtype_t penup_penalty):
    cdef dtype_t location_d
    cdef dtype_t direction_d

    if p[_PU_IDX] > 0 and q[_PU_IDX] > 0:
        return 0.0
    elif p[_PU_IDX] > 0 or q[_PU_IDX] > 0:
        return penup_penalty
    else:
        location_d = (p[_X_IDX]-q[_X_IDX])**2 + (p[_Y_IDX]-q[_Y_IDX])**2
        direction_d = (p[_DX_IDX]-q[_DX_IDX])**2 + (p[_DY_IDX]-q[_DY_IDX])**2
        return (alpha*location_d + (1 - alpha)*direction_d)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _compute_combined_distance(np.ndarray[dtype_t, ndim=2] ink_array1 not None,
                               np.ndarray[dtype_t, ndim=2] ink_array2 not None,
                               dtype_t alpha, 
                               dtype_t penup_z):

    cdef np.ndarray[dtype_t, ndim=2] d_xy
    cdef np.ndarray[dtype_t, ndim=2] d_dir
    cdef np.ndarray[dtype_t, ndim=2] d_combined
    cdef np.ndarray[np.uint8_t, ndim=2] and_penup
    cdef np.ndarray[np.uint8_t, ndim=2] or_penup    
    cdef int i,j,n,m
    cdef dtype_t penup_penalty, stdev_xy, stdev_dir

    n = ink_array1.shape[0]
    m = ink_array2.shape[0]

    # pen-ups
    and_penup = np.zeros((n,m), dtype=np.uint8)
    or_penup = np.zeros((n,m), dtype=np.uint8)

    # inter-point square distance
    d_xy = np.zeros((n,m))
    d_dir = np.zeros((n,m))

    for i in xrange(n):
        for j in xrange(m):
            d_xy[i,j] = ((ink_array2[j,_X_IDX] - ink_array1[i,_X_IDX])**2 + 
                         (ink_array2[j,_Y_IDX] - ink_array1[i,_Y_IDX])**2)
            d_dir[i,j] = ((ink_array2[j,_DX_IDX] - ink_array1[i,_DX_IDX])**2 + 
                          (ink_array2[j,_DY_IDX] - ink_array1[i,_DY_IDX])**2)
            if ink_array2[j,_PU_IDX] > 0 and ink_array1[i,_PU_IDX] > 0:
                and_penup[i,j] = 1
            if ink_array2[j,_PU_IDX] > 0 or ink_array1[i,_PU_IDX] > 0:
                or_penup[i,j] = 1

    stdev_xy = np.std(d_xy[or_penup < 1])
    stdev_dir = np.std(d_dir[or_penup < 1])

    if stdev_xy < _MIN_STDEV: stdev_xy = _MIN_STDEV
    if stdev_dir < _MIN_STDEV: stdev_dir = _MIN_STDEV

    penup_penalty = max(penup_z * (stdev_xy * alpha + 
                                   stdev_dir * (1 - alpha)), 1.0)
    
    d_combined = ((alpha / stdev_xy ) * d_xy + 
                  ((1 - alpha) / stdev_dir) * d_dir)
    
    d_combined[or_penup > 0] = penup_penalty
    d_combined[and_penup > 0] = 0.0

    return d_combined

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _execute_dtw_in_c(np.ndarray[dtype_t, ndim=2] d_combined, 
                      int m, int n):

    cdef np.ndarray[dtype_t, ndim=2] dtw
    cdef np.ndarray[np.int_t, ndim=2] pl
    cdef int i,j
    cdef dtype_t t1,t2,t3
    
    dtw = np.zeros((n,m))
    pl = np.zeros((n,m), dtype=np.int)
    
    for i in xrange(1,n):
        dtw[i,0] = dtw[i-1,0] + d_combined[i,0]
        pl[i,0] = pl[i-1,0] + 1

    for j in xrange(1,m):
        dtw[0,j] = dtw[0,j-1] + d_combined[0,j]
        pl[0,j] = pl[0,j-1] + 1
        
    for i in xrange(1,n):
        for j in xrange(1,m):
            t1 = dtw[i-1,j]
            t2 = dtw[i-1,j-1]
            t3 = dtw[i,j-1]
            if t2 <= t1 and t2 <= t3: 
                dtw[i,j] = t2 + d_combined[i,j]
                pl[i,j] = pl[i-1,j-1] + 1
            elif t1 <= t2 and t1 <= t3:
                dtw[i,j] = t1 + d_combined[i,j]
                pl[i,j] = pl[i-1,j] + 1
            else: 
                dtw[i,j] = t3 + d_combined[i,j];
                pl[i,j] = pl[i,j-1] + 1

    return dtw[n-1,m-1] / pl[n-1,m-1]
