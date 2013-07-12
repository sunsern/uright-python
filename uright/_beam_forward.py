import numpy as np
import heapq as hp
from math import log, exp
from sklearn.utils.extmath import logsumexp 

from _dtwc import _compute_pairwise_distance
from inkutils import INK_STRUCT
from prototype import PrototypeDTW

_X_IDX = INK_STRUCT['X_IDX']
_Y_IDX = INK_STRUCT['Y_IDX']
_DX_IDX = INK_STRUCT['DX_IDX']
_DY_IDX = INK_STRUCT['DY_IDX']
_PU_IDX = INK_STRUCT['PU_IDX']

_EPS = 0.2
_BEAM_PQ_PUSH_THRESHOLD = -10

def _max(a,b):
    if a < b: return b
    else: return a

def _min(a,b):
    if a < b: return a
    else: return b
    
def _logaddexp(a,b):
    if a < b:
        return b + log(1 + exp(a-b))
    else:
        return a + log(1 + exp(b-a))

class BeamForward(object):
    """Beam-search forward algorithm
    
    TODO:
    * Test/debug online normalization
    
    """
    def __init__(self, trained_prototypes, 
                 normalization=False, 
                 skips=3,
                 alpha=0.5, 
                 beam_width=None, 
                 penup_penalty=10.0,
                 algorithm='sum'):

        if algorithm not in ('max', 'sum'):
            raise ValueError('Algorithm must be either max or sum.')

        if algorithm == 'max':
            self._combine = _max
        else:
            self._combine = _logaddexp

        self.algorithm = algorithm
        self.centers = [prot_obj.model.copy()
                        for prot_obj in trained_prototypes]

        num_obs = np.array([prot_obj.num_obs 
                            for prot_obj in trained_prototypes],
                           dtype=np.float64)

        self.log_priors = np.log(num_obs / np.sum(num_obs))
        self.normalization = normalization
        self.alpha = alpha
        self.penup_penalty = penup_penalty
        self.skips = skips

        if beam_width is None:
            self.beam_width = np.inf
        else:
            self.beam_width = beam_width

        # online normalization
        if (normalization):
            for proto in self.centers:
                proto_sum_x = 0.0
                proto_sum_x_sq = 0.0
                proto_sum_y = 0.0
                proto_sum_y_sq = 0.0
                proto_num_points = 0
                for j in range(proto.shape[0]):
                    x = proto[j,_X_IDX]
                    y = proto[j,_Y_IDX],
                    if proto[j,_PU_IDX] < 1:
                        if j > 0:
                            mean_y = proto_sum_y / proto_num_points
                            mean_x = proto_sum_x / proto_num_points
                            h = np.sqrt(proto_sum_y_sq / proto_num_points - 
                                        mean_y**2)
                            proto[j,_X_IDX] = (x - mean_x) / max(h, _EPS)
                            proto[j,_Y_IDX] = (y - mean_y) / max(h, _EPS)
                        else:
                            proto[j,_X_IDX] = 0.0
                            proto[j,_Y_IDX] = 0.0

                        # update normalizing info
                        proto_sum_x += x
                        proto_sum_x_sq += x * x
                        proto_sum_y += y
                        proto_sum_y_sq += y * y
                        proto_num_points += 1
        self.reset()

    def reset(self):
        self._sum_x = 0.0
        self._sum_x_sq = 0.0
        self._sum_y = 0.0
        self._sum_y_sq = 0.0
        self._num_points = 0
        self._active_states = []
        for i in range(len(self.centers)):
            hp.heappush(self._active_states,(-self.log_priors[i], 
                                              self.log_priors[i], i, -1))
            
    def add_point(self, org_point):
        # online normalize
        if (self.normalization and point[_PU_IDX] < 1):
            point = org_point.copy()
            if self._num_points == 0:
                point[_X_IDX] = 0.0
                point[_Y_IDX] = 0.0
            else:
                mean_x = self._sum_x / self._num_points
                mean_y = self._sum_y / self._num_points
                h = np.sqrt(self.sum_y_sq / self.num_points - 
                            mean_y**2)
                point[_X_IDX] = (point[_X_IDX] - mean_x) / max(h, _EPS)
                point[_Y_IDX] = (point[_Y_IDX] - mean_y) / max(h, _EPS)
        else:
            point = org_point

        cached = {}
        combine = self._combine
        _beam_alpha_ = self.alpha
        _beam_penup_ = self.penup_penalty
        state_count = 0
        while (self._active_states and state_count < self.beam_width):
            (_, logprob, i, j) = hp.heappop(self._active_states)
            proto = self.centers[i]
            for k in xrange(_max(j,0), _min(j+1+self.skips, proto.shape[0])):
                hashkey = (i,k)
                if hashkey in cached:
                    (log_d, log_alpha) = cached[hashkey]
                    cached[hashkey] = (log_d, combine(log_alpha, 
                                                      logprob + log_d))
                else:
                    log_d = -(_compute_pairwise_distance(point, 
                                                         proto[k,:],
                                                         _beam_alpha_,
                                                         _beam_penup_))
                    cached[hashkey] = (log_d, logprob + log_d)
                # only incur cost when not staying in the same state 
                if k > j: 
                    logprob += log_d

            state_count += 1
    
        # normalize probabilities
        if (self.algorithm == 'sum'):
            log_denom = logsumexp(
                np.array([v for _,(_,v) in cached.iteritems()]))
        else:
            log_denom = 0.0

        #print "log_denom=%f"%log_denom
        self._active_states = []
        for key, (_, log_alpha) in cached.iteritems():
            log_normalized = log_alpha - log_denom
            if (log_normalized > _BEAM_PQ_PUSH_THRESHOLD):
                (prot_idx, state_idx) = key
                #print "push: %f,%d,%d"%(log_normalized, prot_idx, state_idx)
                hp.heappush(self._active_states,(-log_normalized, 
                                                  log_normalized,
                                                  prot_idx, 
                                                  state_idx))

        # update normalization info
        if (self.normalization and org_point[_PU_IDX] < 1):
            self.sum_x += org_point[_X_IDX]
            self.sum_x_sq += org_point[_X_IDX]**2
            self.sum_y += org_point[_Y_IDX]
            self.sum_y_sq += org_point[_Y_IDX]**2
            self.num_points += 1

    def scores(self):
        logprobs = np.ones(len(self.centers)) * -np.inf
        for (_, u, i, j) in self._active_states:
            if j == self.centers[i].shape[0]-1:
                logprobs[i] = u
        return logprobs
        
    def loglikelihood(self):
        logsumprob = np.ones(len(self.centers)) * -np.inf
        for (_, u, i, j) in self._active_states:
            logsumprob[i] = _logaddexp(logsumprob[i], u)
        return logsumprob
