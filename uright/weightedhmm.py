# Adapted from sklearn.hmm 

import string
import numpy as np

from sklearn import _hmmc, cluster
from sklearn.utils.extmath import logsumexp 
from sklearn.hmm import _BaseHMM, normalize, decoder_algorithms
from sklearn.mixture import (
    GMM, log_multivariate_normal_density, sample_gaussian,
    distribute_covar_matrix_to_match_covariance_type, _validate_covars)

class _WeightedBaseHMM(_BaseHMM):
    """Base class for weighted HMM."""

    def fit(self, obs, obs_weights=None):
        """Perform EM training."""
        if obs_weights is None:
            obs_weights = np.ones(len(obs))

        if self.algorithm not in decoder_algorithms:
            self._algorithm = "viterbi"

        self._init(obs, self.init_params)

        logprob = []
        for i in range(self.n_iter):
            # Expectation step
            stats = self._initialize_sufficient_statistics()
            curr_logprob = 0
            for j in range(len(obs)):
                seq = obs[j]
                seq_weight = obs_weights[j]
                framelogprob = self._compute_log_likelihood(seq)
                lpr, fwdlattice = self._do_forward_pass(framelogprob)
                bwdlattice = self._do_backward_pass(framelogprob)
                gamma = fwdlattice + bwdlattice
                posteriors = np.exp(gamma.T - logsumexp(gamma, axis=1)).T
                # Weighted log prob
                curr_logprob += seq_weight * lpr
                self._accumulate_sufficient_statistics(
                    stats, seq, framelogprob, posteriors, fwdlattice,
                    bwdlattice, self.params, seq_weight)
            logprob.append(curr_logprob)

            # Check for convergence.
            if i > 0 and abs(logprob[-1] - logprob[-2]) < self.thresh:
                break

            # Maximization step
            self._do_mstep(stats, self.params)
        
        #print "log prob = %0.2f"%(curr_logprob)
        return logprob[-1]


    def _accumulate_sufficient_statistics(self, stats, seq, framelogprob,
                                          posteriors, fwdlattice, bwdlattice,
                                          params, seq_weight):
        stats['nobs'] += 1 * seq_weight
        if 's' in params:
            stats['start'] += posteriors[0] * seq_weight
        if 't' in params:
            n_observations, n_components = framelogprob.shape
            lneta = np.zeros((n_observations - 1, n_components, n_components))
            lnP = logsumexp(fwdlattice[-1])
            _hmmc._compute_lneta(n_observations, n_components, fwdlattice,
                                 self._log_transmat, bwdlattice, framelogprob,
                                 lnP, lneta)
            stats["trans"] += np.exp(logsumexp(lneta, 0)) * seq_weight

    def _do_mstep(self, stats, params):
        if self.startprob_prior is None:
            self.startprob_prior = 1.0
        if self.transmat_prior is None:
            self.transmat_prior = 1.0

        if 's' in params:
            self.startprob_ = normalize(
                np.maximum(self.startprob_prior - 1.0 + stats['start'], 1e-20))

        if 't' in params:
            self.transmat_ = normalize(
                np.maximum(self.transmat_prior - 1.0 + stats['trans'], 1e-20),
                axis=1)

class WeightedGaussianHMM(_WeightedBaseHMM):
    """Weighted Gaussian HMM"""
    def __init__(self, n_components=1, covariance_type='diag', startprob=None,
                 transmat=None, startprob_prior=None, transmat_prior=None,
                 algorithm="viterbi", means_prior=None, means_weight=1.0,
                 covars_prior=None, covars_weight=1.0,
                 random_state=None, n_iter=10, thresh=1e-2,
                 params=string.ascii_letters,
                 init_params=string.ascii_letters):
        _WeightedBaseHMM.__init__(self, n_components, startprob, transmat,
                                  startprob_prior=startprob_prior,
                                  transmat_prior=transmat_prior, 
                                  algorithm=algorithm,
                                  random_state=random_state, n_iter=n_iter,
                                  thresh=thresh, params=params,
                                  init_params=init_params)

        self._covariance_type = covariance_type
        if not covariance_type in ['spherical', 'tied', 'diag', 'full']:
            raise ValueError('bad covariance_type')

        self.means_prior = means_prior
        self.means_weight = means_weight

        self.covars_prior = covars_prior
        self.covars_weight = covars_weight

    @property
    def covariance_type(self):
        """Covariance type of the model.

        Must be one of 'spherical', 'tied', 'diag', 'full'.
        """
        return self._covariance_type

    def _get_means(self):
        """Mean parameters for each state."""
        return self._means_

    def _set_means(self, means):
        means = np.asarray(means)
        if (hasattr(self, 'n_features')
                and means.shape != (self.n_components, self.n_features)):
            raise ValueError('means must have shape '
                             '(n_components, n_features)')
        self._means_ = means.copy()
        self.n_features = self._means_.shape[1]

    means_ = property(_get_means, _set_means)

    def _get_covars(self):
        """Return covars as a full matrix."""
        if self._covariance_type == 'full':
            return self._covars_
        elif self._covariance_type == 'diag':
            return [np.diag(cov) for cov in self._covars_]
        elif self._covariance_type == 'tied':
            return [self._covars_] * self.n_components
        elif self._covariance_type == 'spherical':
            return [np.eye(self.n_features) * f for f in self._covars_]

    def _set_covars(self, covars):
        covars = np.asarray(covars)
        _validate_covars(covars, self._covariance_type, self.n_components)
        self._covars_ = covars.copy()

    covars_ = property(_get_covars, _set_covars)

    def _compute_log_likelihood(self, obs):
        return log_multivariate_normal_density(
            obs, self._means_, self._covars_, self._covariance_type)

    def _generate_sample_from_state(self, state, random_state=None):
        if self._covariance_type == 'tied':
            cv = self._covars_
        else:
            cv = self._covars_[state]
        return sample_gaussian(self._means_[state], cv, self._covariance_type,
                               random_state=random_state)

    def _init(self, obs, params='stmc'):
        super(WeightedGaussianHMM, self)._init(obs, params=params)

        if (hasattr(self, 'n_features')
                and self.n_features != obs[0].shape[1]):
            raise ValueError('Unexpected number of dimensions, got %s but '
                             'expected %s' % (obs[0].shape[1],
                                              self.n_features))

        self.n_features = obs[0].shape[1]

        if 'm' in params:
            # Evenly spaced states 
            indices = np.fix(
                np.linspace(0,obs[0].shape[0]-1,self.n_components)).astype(int)
            self._means_ = obs[0][indices,:]
        if 'c' in params:
            cv = np.cov(obs[0].T).clip(min=1e-3)
            if not cv.shape:
                cv.shape = (1, 1)
            self._covars_ = distribute_covar_matrix_to_match_covariance_type(
                cv, self._covariance_type, self.n_components)

    def _initialize_sufficient_statistics(self):
        stats = super(WeightedGaussianHMM, 
                      self)._initialize_sufficient_statistics()
        stats['post'] = np.zeros(self.n_components)
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        stats['obs**2'] = np.zeros((self.n_components, self.n_features))
        stats['obs*obs.T'] = np.zeros((self.n_components, self.n_features,
                                       self.n_features))
        return stats

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice,
                                          params, obs_weight):
        super(WeightedGaussianHMM, self)._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice,
            params, obs_weight)

        if 'm' in params or 'c' in params:
            stats['post'] += posteriors.sum(axis=0) * obs_weight
            stats['obs'] += np.dot(posteriors.T, obs) * obs_weight

        if 'c' in params:
            if self._covariance_type in ('spherical', 'diag'):
                stats['obs**2'] += np.dot(posteriors.T, obs ** 2) * obs_weight
            elif self._covariance_type in ('tied', 'full'):
                for t, o in enumerate(obs):
                    obsobsT = np.outer(o, o)
                    for c in range(self.n_components):
                        stats['obs*obs.T'][c] += (posteriors[t, c] * 
                                                  obsobsT) * obs_weight

    def _do_mstep(self, stats, params):
        super(WeightedGaussianHMM, self)._do_mstep(stats, params)
        
        denom = stats['post'][:, np.newaxis]
        EPS = 1e-10

        if 'c' in params:
            covars_prior = self.covars_prior
            covars_weight = self.covars_weight
            if covars_prior is None:
                covars_weight = 0
                covars_prior = 0

            means_prior = self.means_prior
            means_weight = self.means_weight
            if means_prior is None:
                means_weight = 0
                means_prior = 0

            meandiff = self._means_ - means_prior

            if self._covariance_type in ('spherical', 'diag'):
                cv_num = (means_weight * (meandiff) ** 2
                          + stats['obs**2']
                          - 2 * self._means_ * stats['obs']
                          + self._means_ ** 2 * denom)
                cv_den = max(covars_weight - 1, 0) + denom

                self._covars_ = ((covars_prior + cv_num + EPS) / 
                                 (cv_den + EPS)).clip(min=1e-3)
                if self._covariance_type == 'spherical':
                    self._covars_ = np.tile(
                        self._covars_.mean(1)[:, np.newaxis],
                        (1, self._covars_.shape[1]))
                    
            elif self._covariance_type in ('tied', 'full'):
                cvnum = np.empty((self.n_components, self.n_features,
                                  self.n_features))
                for c in range(self.n_components):
                    obsmean = np.outer(stats['obs'][c], self._means_[c])
                    cvnum[c] = (means_weight * np.outer(meandiff[c],
                                                        meandiff[c])
                                + stats['obs*obs.T'][c]
                                - obsmean - obsmean.T
                                + np.outer(self._means_[c], 
                                           self._means_[c])
                                * stats['post'][c])
                cvweight = max(covars_weight - self.n_features, 0)
                if self._covariance_type == 'tied':
                    self.covars_ = ((covars_prior + cvnum.sum(axis=0) + EPS) /
                                    (cvweight + stats['post'].sum() + EPS))
                elif self._covariance_type == 'full':
                    self.covars_ = ((covars_prior + cvnum + EPS) /
                                    (cvweight + stats['post'][:, None, None] + 
                                     EPS))
        if 'm' in params:
            prior = self.means_prior
            weight = self.means_weight
            if prior is None:
                weight = 0
                prior = 0
            self._means_ = (weight * prior + stats['obs']) / (weight + denom)

