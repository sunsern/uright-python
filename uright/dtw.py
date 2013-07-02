import numpy as np

import inkutils
from _dtwc import (_execute_dtw_in_c,
                   _compute_combined_distance)

def compute_dtw_distance(ink_array1, ink_array2, alpha=0.5, penup_z=10.0):
    n = ink_array1.shape[0]
    m = ink_array2.shape[0]

    if (n <= 0 or m <= 0):
        raise ValueError('both input arrays must have length > 0')

    combined_d = _compute_combined_distance(ink_array1, 
                                            ink_array2, 
                                            alpha, 
                                            penup_z)

    return _execute_dtw_in_c(combined_d, m , n)

def compute_dtw_vector(center_ink, ink_array, alpha=0.5, penup_z=10.0):
    """Warp `ink_array` to `center_ink`

    Returns
    -------
    array (1, center_ink.shape[0] x 5)
  
    """
    n = center_ink.shape[0]
    m = ink_array.shape[0]

    if (n <= 0 or m <= 0):
        raise ValueError('input arrays must have length > 1')

    combined_d = _compute_combined_distance(center_ink, 
                                            ink_array, 
                                            alpha, 
                                            penup_z)
    dtw = np.zeros((n,m))
    path = np.zeros((n,m))
    _LEFT, _DIAG, _UP = 1, 2, 3

    for i in range(1,n):
        dtw[i,0] = dtw[i-1,0] + combined_d[i,0]
        path[i,0] = _UP

    for j in range(1,m):
        dtw[0,j] = dtw[0,j-1] + combined_d[0,j]
        path[0,j] = _LEFT

    for i in range(1,n):
        for j in range(1,m):
            dtw[i,j] = combined_d[i,j] + min(dtw[i-1,j], 
                                             min(dtw[i-1,j-1], dtw[i,j-1]))
            if (dtw[i-1,j-1] < dtw[i-1,j] and dtw[i-1,j-1] < dtw[i,j-1]):
                path[i,j] = _DIAG
            elif (dtw[i-1,j] < dtw[i-1,j-1] and dtw[i-1,j] < dtw[i,j-1]):
                path[i,j] = _UP
            else:
                path[i,j] = _LEFT

    # trace backwards
    i = n-1
    j = m-1
    mapping = np.zeros(n)
    while i >= 0 and j >= 0:
        mapping[i] = j       
        if path[i,j] == _UP:
            i -= 1
        elif path[i,j] == _DIAG:
            i -= 1
            j -= 1
        else:
            j -= 1

    # do not match pen-up
    _PU_IDX = inkutils.INK_STRUCT['PU_IDX']
    for i in range(1,n):
        if (center_ink[i,_PU_IDX] < 1 and ink_array[mapping[i],_PU_IDX] > 0):
            mapping[i] = mapping[i-1]

    ret = np.zeros(center_ink.shape)
    for i in range(n):
        ret[i,:] = ink_array[mapping[i],:] - center_ink[i,:]
    
    return ret.reshape(-1)
