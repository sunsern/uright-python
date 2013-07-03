import numpy as np

from dtw import compute_dtw_distance
from _dtwc import _compute_combined_distance
import inkutils

_DX_IDX = inkutils.INK_STRUCT['DX_IDX']
_DY_IDX = inkutils.INK_STRUCT['DY_IDX']
_PU_IDX = inkutils.INK_STRUCT['PU_IDX']

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
        max_ind = np.argmax(-np.sqrt(new_u) / new_v)
        peak_count[max_ind] += 1
        u = new_u
        v = new_v
    return peak_count


def _compute_path(prototype_dict, test_ink_dict, alpha=0.5, penup_z=10.0):

    def _closest_prototype(p_array, ink, alpha, penup_z):
        all_d = np.zeros(len(p_array))
        for i in range(len(p_array)):
            all_d[i] = compute_dtw_distance(ink, 
                                            p_array[i],
                                            alpha=alpha, 
                                            penup_z=penup_z)
        return np.argmin(all_d)

    path_info = []
    for key in prototype_dict.keys():
        p_array = prototype_dict[key]
        all_count = [ np.zeros(p.shape[0]) for p in p_array ]
        all_hit = np.zeros(len(p_array))
        for ink in test_ink_dict[key]:
            idx = _closest_prototype(p_array,ink,alpha,penup_z)
            peak_count = _pathcount(p_array[idx],ink)
            all_count[idx] += peak_count
            all_hit[idx] += 1
        for i in range(len(p_array)):
            path_info.append((key,
                              prototype_dict[key][i],
                              all_count[i],
                              all_hit[i]))
    return path_info


def _merge_bad_states(prototype, path_fraction, threshold=0.50, verbose=False):
    path_fraction = np.asarray(path_fraction)
    plen = prototype.shape[0]
    minidx = np.argmin(path_fraction)
    if prototype[minidx,_PU_IDX] > 0:
        if verbose: print "[merge] skip"
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
            if verbose: print "[merge] delete %d"%li
            new_proto = np.delete(prototype,ui,axis=0)
            return new_proto
        elif li > 0 and ui < plen - 1:            
            if verbose: print "[merge] merge %d %d"%(li,ui)
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


def _state_reduction(prototype_dict, test_ink_dict, 
                     n_iter=30, merge_threshold=0.5, 
                     remove_threshold=0.01, verbose=False):
    prototypes = prototype_dict.copy()
    for it in range(n_iter):
        path_info = _compute_path(prototypes, test_ink_dict)
        reduced_prototypes = {}
        old_total_length = 0
        new_total_length = 0
        if verbose: print "[state-reduction] iter %d"%it
        for label,old_proto,count,hit in path_info:
            if hit > 0:
                fraction = np.asarray(count) / hit 
                old_total_length += old_proto.shape[0]

                (new_proto,fraction) = _remove_useless_states(
                    old_proto, fraction, threshold=remove_threshold)

                new_proto = _merge_bad_states(new_proto,fraction,
                                              threshold=merge_threshold,
                                              verbose=verbose)
                new_total_length += new_proto.shape[0]
                reduced_prototypes.setdefault(label,[]).append(new_proto)
                if verbose: print " > %s: %d -> %d"%(label, 
                                                     old_proto.shape[0], 
                                                     new_proto.shape[0])

        prototypes = reduced_prototypes

        if (old_total_length == new_total_length):
            break

    return prototypes
