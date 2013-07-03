import numpy as np
import numpy.random as random
import json

# each point structure
INK_STRUCT = {'X_IDX' :0, 
              'Y_IDX' :1, 
              'DX_IDX':2, 
              'DY_IDX':3, 
              'PU_IDX':4,
              'T_IDX' :5}

_X_IDX  = INK_STRUCT['X_IDX']
_Y_IDX  = INK_STRUCT['Y_IDX']
_DX_IDX = INK_STRUCT['DX_IDX']
_DY_IDX = INK_STRUCT['DY_IDX']
_PU_IDX = INK_STRUCT['PU_IDX']
_T_IDX  = INK_STRUCT['T_IDX']

_EPS = 1e-10

def json2array(ink,timestamp=False):
    """Convert a json ink into a numpy array.

    Parameters
    ----------
    timestamp : bool
       If true, the output array also includes timestamps.

    Returns
    -------
    (N x D) array where N is the number of points. Each row is
    [x,y,dx,dy,pen-up, {t}]

    """
    all_points = []
    
    if timestamp:
        penup = np.empty(6)
    else:
        penup = np.empty(5)
    penup[:] = np.nan
    penup[_PU_IDX] = 1

    for stroke in ink['strokes']:
        previous_point = None
        for point in stroke:
            if timestamp:
                vec = np.zeros(6)                
                vec[_T_IDX] = point['t']
            else:
                vec = np.zeros(5)
            vec[_X_IDX] = float(point['x'])
            vec[_Y_IDX] = float(point['y'])
            dx = 0.0
            dy = 0.0
            if previous_point is not None:
                (px,py) = previous_point
                dx = vec[_X_IDX] - px 
                dy = vec[_Y_IDX] - py
                # normalize by the norm
                z = np.sqrt(dx*dx + dy*dy)
                dx = dx / max(z, _EPS)
                dy = dy / max(z, _EPS)
            vec[_DX_IDX] = dx
            vec[_DY_IDX] = dy
            vec[_PU_IDX] = 0
            previous_point = (vec[_X_IDX],vec[_Y_IDX])
            all_points.append(vec)
        all_points.append(penup)
    return np.asarray(all_points)

def center_ink(ink):
    """Center the ink at x=0"""
    ink_array = ink.copy()
    x = ink_array[:,_X_IDX]
    mean_x = np.mean(x[~np.isnan(x)])
    ink_array[:,_X_IDX] = ink_array[:,_X_IDX] - mean_x
    return ink_array
 
def scale_ink(ink):
    """Scale ink height to [-1..1]"""
    ink_array = ink.copy()
    y = ink_array[:,_Y_IDX]
    min_y = np.nanmin(y)
    max_y = np.nanmax(y)
    h = (max_y - min_y) / 2.0
    ink_array[:,_X_IDX] = ink_array[:,_X_IDX] / max(h, _EPS)
    ink_array[:,_Y_IDX] = -1.0 + ((ink_array[:,_Y_IDX] - min_y) / max(h, _EPS))
    return ink_array

def update_directions(ink):
    ink_array = ink.copy()
    for i in range(1,ink_array.shape[0]):
        if ink_array[i,_PU_IDX] < 1:
            dx = ink_array[i,_X_IDX] - ink_array[i-1,_X_IDX]
            dy = ink_array[i,_Y_IDX] - ink_array[i-1,_Y_IDX]
            z = np.sqrt(dx*dx + dy*dy)
            dx = dx / max(z, _EPS)
            dy = dy / max(z, _EPS)
            ink_array[i,_DX_IDX] = dx
            ink_array[i,_DY_IDX] = dy
    return ink_array
    
def add_noise(ink):
    ink_array = ink.copy()
    n = ink_array.shape[0]
    scaleX = random.uniform(0.8,1.2)
    scaleY = random.uniform(0.8,1.2)
    translateX = random.normal(scale=0.1)
    translateY = random.normal(scale=0.1)
    ink_array[:,_X_IDX] = (ink_array[:,_X_IDX] * scaleX) + translateX
    ink_array[:,_Y_IDX] = (ink_array[:,_Y_IDX] * scaleY) + translateY
    return update_directions(ink_array)

def normalize_ink(ink):
    """Normalize using bounding box"""
    return scale_ink(center_ink(ink))

def filter_bad_ink(ink_list, min_length=3):
    """Filter out bad ink"""
    all_ink = map(json2array, ink_list)
    filtered = [ink_list[i]
                for i in range(len(ink_list))
                if all_ink[i].shape[0] > min_length]
    return filtered

def user_normalized_ink(user_raw_ink):
    normalized_ink = {}
    for userid, raw_ink in user_raw_ink.iteritems():
        temp = {}
        for label, ink_list in raw_ink.iteritems():
            temp[label] = [np.nan_to_num(normalize_ink(json2array(ink))) 
                           for ink in filter_bad_ink(ink_list)]
        normalized_ink[userid] = temp
    return normalized_ink
