import numpy as np

from dtw import compute_dtw_distance
from _dtwc import _compute_combined_distance
import inkutils

_DX_IDX = inkutils.INK_STRUCT['DX_IDX']
_DY_IDX = inkutils.INK_STRUCT['DY_IDX']
_PU_IDX = inkutils.INK_STRUCT['PU_IDX']

__DEBUG=False

def _pathcount(template_ink, ink, alpha=0.5, penup_z=10.0):
    m = template_ink.shape[0]
    n = ink.shape[0]
    u = np.zeros(m)
    v = np.zeros(m)
    peak_count = np.zeros(m)
    d = _compute_combined_distance(ink, template_ink, alpha, penup_z)
    for i in xrange(n):
        new_u = np.zeros(m)
        new_v = np.zeros(m)
        new_u[0] = u[0] + d[i,0]
        new_v[0] = v[0] + 1
        for j in xrange(1,m):
            t1 = u[j]
            t2 = u[j-1]
            if (t2 < t1) and (t2 < new_u[j-1]):
                new_u[j] = t2 + d[i,j]
                new_v[j] = v[j-1] + 1
            elif (t1 < t2) and (t1 < new_u[j-1]):
                new_u[j] = t1 + d[i,j]
                new_v[j] = v[j] + 1
            else:
                new_u[j] = new_u[j-1] + d[i,j]
                new_v[j] = new_v[j-1] + 1
        max_ind = np.argmax(-new_u / new_v)
        peak_count[max_ind] += 1
        u = new_u
        v = new_v
    return peak_count

def _compute_fraction(prototype, test_ink, alpha=0.5, penup_z=10.0):
    fraction = np.zeros(prototype.shape[0])
    for ink in test_ink:
        fraction += _pathcount(prototype, ink)
    return fraction / len(test_ink)

def _merge_bad_states(prototype, path_fraction, threshold=0.50):
    path_fraction = np.asarray(path_fraction)
    plen = prototype.shape[0]
    minidx = np.argmin(path_fraction)
    if prototype[minidx,_PU_IDX] > 0:
        if __DEBUG: print "[merge] skip"
        return prototype

    t = np.amin(path_fraction)
    if t < threshold:
        li = minidx
        ui = minidx
        while (li > 1 and path_fraction[li-1] < threshold and
               prototype[li-1,_PU_IDX] < 1):
            li -= 1
        while (ui < plen - 1 and 
               path_fraction[ui+1] < threshold and
               prototype[ui+1,_PU_IDX] < 1):
            ui += 1

        if li == ui:
            if __DEBUG: print "[merge] delete %d"%li
            new_proto = np.delete(prototype,ui,axis=0)
            return new_proto
        elif li > 0 and ui < plen - 1:            
            if __DEBUG: print "[merge] merge %d %d"%(li,ui)
            avg_state = np.mean(prototype[li:ui+1,:],axis=0)
            # recompute the pen direction
            dx = avg_state[_DX_IDX]
            dy = avg_state[_DY_IDX]
            z = np.sqrt(dx*dx+dy*dy)
            dx = dx / max(z, 1e-10)
            dy = dy / max(z, 1e-10)
            avg_state[_DX_IDX] = dx
            avg_state[_DY_IDX] = dy
            # new prototype
            new_proto = np.delete(prototype,
                                  range(max(li,0),min(ui+1,plen)),
                                  axis=0)
            new_proto = np.insert(new_proto, li, avg_state, axis=0)
            return new_proto
    return prototype

def _remove_useless_states(prototype, path_fraction, threshold=0.01):
    path_fraction = np.asarray(path_fraction)
    # find indices of states to remove
    delete_idx = np.nonzero(path_fraction < threshold)
    new_prototype = np.delete(prototype, delete_idx, axis=0)
    new_path_fraction = np.delete(path_fraction, delete_idx)
    return (new_prototype, new_path_fraction)

def _state_reduction(prototype, test_ink, 
                     n_iter=30, merge_threshold=0.6, 
                     remove_threshold=0.01):
    """Reduce number of states of `prototype` 
    """
    current_prototype = prototype

    for it in range(n_iter):
        if __DEBUG: print "[state-reduction] iter %d"%it
        fraction = _compute_fraction(current_prototype, test_ink)
        
        # remove rarely states
        (new_prototype,fraction) = _remove_useless_states(
            current_prototype, fraction, threshold=remove_threshold)
        
        # merge similar states
        new_prototype = _merge_bad_states(new_prototype, fraction,
                                          threshold=merge_threshold)
        
        if __DEBUG: print " > %d -> %d"%(current_prototype.shape[0], 
                                         new_prototype.shape[0])
        # stop when no update
        if (current_prototype.shape[0] == new_prototype.shape[0]):
            break
        
        current_prototype = new_prototype

    return current_prototype
