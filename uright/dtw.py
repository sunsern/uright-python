import numpy as np

from inkutils import INK_STRUCT
from _dtwc import (_execute_dtw_in_c,
                   _compute_combined_distance)

_PU_IDX = INK_STRUCT['PU_IDX']

def compute_dtw_distance(ink_array1, ink_array2, alpha=0.5, penup_z=10.0):
    """Returns the DTW distance between two trajectories."""
    n = ink_array1.shape[0]
    m = ink_array2.shape[0]

    if (n <= 0 or m <= 0):
        raise ValueError('Both input arrays must have length > 0')

    combined_d = _compute_combined_distance(ink_array1, 
                                            ink_array2, 
                                            alpha, 
                                            penup_z)

    return _execute_dtw_in_c(combined_d, m , n)

def compute_dtw_vector(center_ink, ink_array, alpha=0.5, penup_z=10.0):
    """Returns a vector representing the difference between 
    `center_ink` and time-warped `ink_array`. The output vector 
    always has the same length as `center_ink`."""
    n = center_ink.shape[0]
    m = ink_array.shape[0]

    if (n <= 0 or m <= 0):
        raise ValueError('Both input arrays must have length > 0')

    combined_d = _compute_combined_distance(center_ink, 
                                            ink_array, 
                                            alpha, 
                                            penup_z)
    dtw = np.zeros((n,m))
    path = np.zeros((n,m))
    _LEFT, _DIAG, _UP = 1, 2, 3

    for i in xrange(1,n):
        dtw[i,0] = dtw[i-1,0] + combined_d[i,0]
        path[i,0] = _UP

    for j in xrange(1,m):
        dtw[0,j] = dtw[0,j-1] + combined_d[0,j]
        path[0,j] = _LEFT

    for i in xrange(1,n):
        for j in xrange(1,m):
            dtw[i,j] = combined_d[i,j] + min(dtw[i-1,j], 
                                             min(dtw[i-1,j-1], 
                                                 dtw[i,j-1]))
            if (dtw[i-1,j-1] < dtw[i-1,j] and dtw[i-1,j-1] < dtw[i,j-1]):
                path[i,j] = _DIAG
            elif (dtw[i-1,j] < dtw[i-1,j-1] and dtw[i-1,j] < dtw[i,j-1]):
                path[i,j] = _UP
            else:
                path[i,j] = _LEFT

    # trace backwards
    i, j = n-1, m-1
    mapping = np.zeros(n, dtype=np.int)
    while i >= 0 and j >= 0:
        mapping[i] = j
        if path[i,j] == _UP: i -= 1
        elif path[i,j] == _LEFT: j -= 1
        else:
            i -= 1
            j -= 1

    # do not match pen-up
    for i in xrange(1,n):
        if (center_ink[i,_PU_IDX] < 1 and ink_array[mapping[i],_PU_IDX] > 0):
            # find a nearby point that is not a pen-up
            j = i - 1
            while ink_array[mapping[j],_PU_IDX] > 0: j = j - 1
            mapping[i] = mapping[j]

    diff_array = ink_array[mapping,:] - center_ink
    return diff_array.reshape(-1)
