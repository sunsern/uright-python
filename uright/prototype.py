# Prototype (Character model)
#
# Author: Sunsern Cheamanunkul (sunsern@gmail.com)

"""
This module implements different types of prototypes.
"""

import numpy as np

from sklearn.hmm import normalize

from dtw import compute_dtw_distance, compute_dtw_vector
from weightedhmm import WeightedGaussianHMM 
from inkutils import update_directions

class _Prototype(object):
    """Prototype base class.
    
    An abstract representation of a prototype.
    
    Attributes
    ----------
    num_obs : int
       Number of training observations associated with
       the prototype.
    
    label : string
       Label of the prototype.
    
    model : object 
       The underlying model.
  
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

    def fromJSON(self,json_dict):
        self.label = json_dict['label']
        self.num_obs = json_dict['num_obs']


class PrototypeDTW(_Prototype):
    """DTW-based prototype.

    The prototype is simply a sequence of points along the trajactory.
    The similarity between the prototype and an observation is measured 
    using the Dynamic Time Warping distance.

    Parameters
    ----------
    alpha : float
       Weighting between location distance and direction distance.
       If alpha=1.0, the direction distance is ignored. If alpha=0.0,
       the location distance is ignored.

    """
    def __init__(self, label, alpha=0.5):
        _Prototype.__init__(self, label)
        self.alpha = alpha

    def train(self, obs, obs_weights=None, center_type='medoid'):
        """Estimates the prototype from a set of observations."""

        def _find_medoid(obs, obs_weights, distmat):
            n = len(obs)
            weighted_distmat = distmat * np.tile(obs_weights,(n,1)).T
            sum_distmat = (np.sum(weighted_distmat,axis=0) / 
                           np.sum(obs_weights))
            return (np.argmin(sum_distmat), np.amin(sum_distmat))

        def _find_centroid(obs, obs_weights, medoid):
            n_features = obs[0].shape[1]
            f = [compute_dtw_vector(medoid, ink) for ink in obs]
            feature_mat = np.vstack(f)
            feature_mat = np.nan_to_num(feature_mat)
            weighted_feature_mat = feature_mat * np.tile(
                obs_weights, (feature_mat.shape[1],1)).T
            mean_ink = (np.sum(weighted_feature_mat,axis=0) / 
                        np.sum(obs_weights))
            mean_ink = np.reshape(mean_ink, (-1,n_features), order='C')
            mean_ink = mean_ink + medoid

            weighted_dists = [obs_weights[i] * 
                              compute_dtw_distance(mean_ink, obs[i], 
                                                   alpha=self.alpha) 
                              for i in xrange(len(obs))]
            avg_dist = np.sum(weighted_dists) / np.sum(obs_weights)

            return (update_directions(mean_ink), avg_dist)

        n = len(obs)
        self.num_obs = n

        if obs_weights is None:
            obs_weights = np.ones(n)
        else:
            obs_weights = np.asarray(obs_weights)

        if not center_type in ['medoid', 'centroid']:
            raise ValueError(
                'center_type should be either medoid or centroid.')

        # calculate distance matrix
        distMat = np.zeros((n,n))
        for i in xrange(n):
            for j in xrange(i+1,n):
                distMat[i,j] = compute_dtw_distance(obs[i], obs[j], 
                                                    alpha=self.alpha)
                distMat[j,i] = distMat[i,j]

        (medoid_idx, avg_min_dist) = _find_medoid(obs, obs_weights, distMat)
        
        if center_type == 'centroid':
            (self.model, avg_min_dist) = _find_centroid(obs, obs_weights, 
                                                        obs[medoid_idx])
        else:
            self.model = obs[medoid_idx].copy()

        return -avg_min_dist

    def score(self, obs):
        """Calculates the score of an observation.
        
        The score is defined as negative of the DTW distance.

        Returns
        -------
        (score, None)

        """
        dist = compute_dtw_distance(self.model, obs, alpha=self.alpha)
        return -dist, None

    def toJSON(self):
        """Returns a JSON dictionary representing the prototype."""
        info = super(PrototypeDTW, self).toJSON()
        info['alpha'] = self.alpha
        info['center'] = self.model.astype(np.float16).tolist()
        return info

    def fromJSON(self,jsonObj):
        """Initializes the prototype with a JSON dictionary.""" 
        super(PrototypeDTW, self).fromJSON(jsonObj)
        self.alpha = jsonObj['alpha']
        self.model = np.asarray(jsonObj['center'])


class PrototypeHMM(_Prototype):
    """HMM-based prototype.

    This class uses HMM as the underlying model. The similarity is defined
    in term of the log likelihood.

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
        """Estimates the prototype from a set of observations.
        
        Parameters
        ----------
        max_N : int
           The maximum lenght of the HMM.
      
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
        self.transmat = np.zeros((self.N,self.N))
        for i in range(self.N):
            self.transmat[i,i] = self.self_transprob
            if i+1 < self.N:
                self.transmat[i,i+1] = self.next_transprob
            for j in range(i+2, self.N):
                self.transmat[i,j] = self.skip_transprob

        self.transmat = normalize(self.transmat, axis=1)

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
        """Calculates the score of an observation.
        
        The score is defined as the log likelihood of the observation
        under the model.

        Returns
        -------
        (loglikelihood, fwdlattice)

        """
        obs = np.asarray(obs)
        framelogprob = self.model._compute_log_likelihood(obs)
        logprob, fwdlattice = self.model._do_forward_pass(framelogprob)
        return logprob, fwdlattice

    def toJSON(self):
        """Returns a JSON dictionary representing the prototype."""
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
        """Initializes the prototype with a JSON dictionary.""" 
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


