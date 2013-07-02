import numpy as np
from sklearn.hmm import normalize

from weightedhmm import WeightedGaussianHMM 
from dtw import (compute_dtw_distance, compute_dtw_vector)
from inkutils import update_directions

class _Prototype(object):
    """Base class for Prototype
    
    Attributes
    ---------
    num_obs : int
       Number of training observation.
    
    label : string
       Corresponding label.
    
    model : Object
       Underlying model.
  
    """
    def __init__(self, label):
        self.label = label
        self.num_obs = 0
        self.model = None
        
    def train(self):
        pass

    def score(self):
        pass

    def toJSON(self):
        info = {}
        info['label'] = self.label
        info['num_obs'] = self.num_obs
        return info

    def fromJSON(self,jsonObj):
        self.label = jsonObj['label']
        self.num_obs = jsonObj['num_obs']


class PrototypeDTW(_Prototype):
    """DTW-based prototype 

    Parameters
    ----------
    alpha : float
       Weighting between location distance and direction distance
       where alpha=1.0 ignores direction distance and alpha=0.0
       ignores location distance.

    """
    def __init__(self, label, alpha=0.5):
        _Prototype.__init__(self, label)
        self.alpha = alpha

    def train(self, obs, obs_weights=None, center_type='medoid'):

        def _find_medoid(obs, obs_weights, distmat):
            n = len(obs)
            weighted_distmat = distmat * np.tile(obs_weights,(n,1)).T
            sum_distmat = np.sum(weighted_distmat,axis=0) / np.sum(obs_weights)
            return (np.argmin(sum_distmat), np.amin(sum_distmat))

        def _find_centroid(obs, obs_weights, medoid):
            n_features = obs[0].shape[1]
            f = [compute_dtw_vector(medoid, ink) for ink in obs]
            feature_mat = np.vstack(f)
            feature_mat = np.nan_to_num(feature_mat)
            weighted_feature_mat = feature_mat * np.tile(obs_weights, 
                                                         (feature_mat.shape[1],1)).T
            mean_ink = np.sum(weighted_feature_mat,axis=0) / np.sum(obs_weights)
            mean_ink = np.reshape(mean_ink, (-1,n_features), order='C')
            mean_ink = mean_ink + medoid
            return update_directions(mean_ink)

        n = len(obs)

        if obs_weights is None:
            obs_weights = np.ones(n)
        else:
            obs_weights = np.asarray(obs_weights)

        if not center_type in ['medoid', 'centroid']:
            raise ValueError('bad center type')

        # calculate distance matrix
        distMat = np.zeros((n,n))
        for i in xrange(n):
            for j in xrange(i+1,n):
                distMat[i,j] = compute_dtw_distance(obs[i], 
                                                    obs[j], 
                                                    alpha=self.alpha)
                distMat[j,i] = distMat[i,j]

        (medoid_idx, avg_min_dist) = _find_medoid(obs, obs_weights, distMat)
        
        if center_type == 'centroid':
            self.model = _find_centroid(obs, obs_weights, obs[medoid_idx])
        else:
            self.model = obs[medoid_idx]

        self.num_obs = len(obs)

        return -avg_min_dist

    def score(self, obs):
        dist = compute_dtw_distance(self.model, obs, alpha=self.alpha)
        return -dist, None

    def toJSON(self):
        info = super(PrototypeDTW, self).toJSON()
        info['alpha'] = self.alpha
        info['center'] = self.model.astype(np.float16).tolist()
        return info

    def fromJSON(self,jsonObj):
        super(PrototypeDTW, self).fromJSON(jsonObj)
        self.alpha = jsonObj['alpha']
        self.model = np.asarray(jsonObj['center'])


class PrototypeHMM(_Prototype):
    """HMM-based prototype

    Parameters
    ----------
    num_states : float
       Number of hidden states. If num_states < 1, the number of states is 
       set proportionally to the average length of the observations. 
       For example, if num_states = 0.5, the number of states will 
       be set to 0.5 * average length of the observations.

    self_transprob : float
       Probability of staying in the same state.

    next_transprob : float
       Probability of moving to the adjacent state.

    skip_transprob : float
       Probability of moving to any other non-adjacent states.

    Attributes
    ----------
    N : int
       Number of hidden states in the model.

    """
    def __init__(self, label, num_states=0.5, self_transprob=0.8, 
                 next_transprob=0.2, skip_transprob=1e-6):
        _Prototype.__init__(self, label)
        self.num_states = num_states
        self.self_transprob = self_transprob
        self.next_transprob = next_transprob
        self.skip_transprob = skip_transprob
        
    def train(self, obs, obs_weights=None, max_N=15):
        """Train the HMM model
        """
        if obs_weights is None:
            obs_weights = np.ones(len(obs))
        else:
            obs_weights = np.asarray(obs_weights)

        # set the number of states
        if self.num_states >= 1.0:
            self.N = int(self.num_states)
        else:
            mean_length = np.mean([each_obs.shape[0] for each_obs in obs])
            self.N = min(int(self.num_states * mean_length), max_N)

        # transition prob: left-to-right
        self.transmat = np.zeros((self.N, self.N))
        for i in range(self.N):
            self.transmat[i,i] = self.self_transprob
            if i+1 < self.N:
                self.transmat[i,i+1] = self.next_transprob
            for j in range(i+2, self.N):
                self.transmat[i,j] = self.skip_transprob

        self.transmat = normalize(self.transmat,axis=1)

        # state prior prob: left-most only
        self.startprob = np.zeros(self.N)
        self.startprob[0] = 1.0
        
        self.model = WeightedGaussianHMM(self.N, 'diag', 
                                         self.startprob,
                                         self.transmat,
                                         algorithm='map',
                                         params='mc')
        self.num_obs = len(obs)

        return self.model.fit(obs, obs_weights=obs_weights)

    def score(self, obs):
        obs = np.asarray(obs)
        framelogprob = self.model._compute_log_likelihood(obs)
        logprob, fwdlattice = self.model._do_forward_pass(framelogprob)
        return logprob, fwdlattice

    def toJSON(self):
        info = super(PrototypeHMM, self).toJSON()
        info['n_components'] = int(self.model.n_components)
        info['n_features'] = int(self.model.n_features)
        info['transmat'] = self.model.transmat_.astype(np.float16).tolist()
        info['startprob'] = self.model.startprob_.astype(np.float16).tolist()
        info['means'] = self.model._means_.astype(np.float16).tolist()
        info['covars'] = self.model._covars_.astype(np.float16).tolist()
        info['N'] = self.N
        return info

    def fromJSON(self,jsonObj):
        super(PrototypeHMM, self).fromJSON(jsonObj)
        self.N = jsonObj['N']
        self.model = WeightedGaussianHMM(self.N, 'diag',
                                         algorithm='map',
                                         params='mc')
        self.model.n_features = jsonObj['n_features']
        self.model.transmat_ = normalize(jsonObj['transmat'],axis=1)
        self.model.startprob_ = normalize(jsonObj['startprob'],axis=0)
        self.model._means_ = np.asarray(jsonObj['means'])
        self.model._covars_ = np.asarray(jsonObj['covars'])


