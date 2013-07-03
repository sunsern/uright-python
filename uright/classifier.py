import numpy as np

from _state_reduction import _state_reduction
from _beam_dtw import BeamSearchDTW
from prototype import PrototypeHMM,PrototypeDTW

def _max_score(obs, prot_list, log_priors):
    """
    Returns the label of the prototype in `prot_list`
    that has the maximum score (or log likelihood).
    """
    all_scores = np.empty(len(prot_list))
    for i in xrange(len(prot_list)):
        prot_obj = prot_list[i]
        logprob,_ = prot_obj.score(obs)
        all_scores[i] = log_priors[i] + logprob
    return prot_list[np.argmax(all_scores)].label

class _Classifier(object):
    """Maximum likelihood classifier.
        
    Attributes
    ----------
    trained_prototypes : list
       List of trained prototype instances

    log_priors : numpy.ndarray
       Log prior probability of each prototype (offsets)
    """
    def __init__(self):
        self._trained_prototypes = []
        self.log_priors = None

    def _compute_log_priors(self):
        pass

    def _get_trained_prototypes(self):
        return self._trained_prototypes

    def _set_trained_prototypes(self, prototypes):
        self._trained_prototypes = list(prototypes)
        self._compute_log_priors()

    trained_prototypes = property(_get_trained_prototypes, 
                                  _set_trained_prototypes)
    
    def train(self):
        pass
        
    def test(self, label_ink_data, dview=None):
        """Calculates the percent classification accuracy.
    
        Parameters
        ----------
        label_ink_data : list
           List of label-ink pairs.

        dview : IPython.Directview
           For parallel processing.

        Returns
        -------
        (accuracy, true_label, predicted)
        """
        if len(self._trained_prototypes) == 0:
            raise ValueError('trained_prototypes is empty.'
                             'You must train classifer before calling test')

        n = len(label_ink_data)
        true_labels, all_ink = zip(*label_ink_data)

        if dview is None:
            predicted = map(_max_score,
                            all_ink,
                            [self._trained_prototypes] * n,
                            [self.log_priors] * n)
        else:
            # IPython parallel map
            predicted = dview.map_sync(_max_score,
                                       all_ink,
                                       [self._trained_prototypes] * n,
                                       [self.log_priors] * n)

        true_labels = np.asarray(true_labels)
        predicted = np.asarray(predicted)
        accuracy = 100.0 * np.sum(true_labels == predicted) / n
        return (accuracy, true_labels, predicted)


    def classify(self, obs):
        """Classify observation using the model."""
        if len(self._trained_prototypes) > 0:
            return _max_score(obs, 
                              self._trained_prototypes, 
                              self.log_priors)
        else:
            return None

    def toJSON(self):
        prototypes = []
        for i in range(len(self._trained_prototypes)):
            p = self._trained_prototypes[i]
            proto_info = p.toJSON()
            proto_info['prior'] = np.exp(self.log_priors[i]).astype(float16)
            prototypes.append(proto_info)
        return {'prototypes':prototypes}
    
    def fromJSON(self, json_obj):
        pass
        

class ClassifierDTW(_Classifier):
    """Classifier with DTW prototypes
    
    Attributes
    ----------
    min_cluster_size : int
      Minimum number of examples in a cluster.

    alpha : float
      Defult alpha for DTW algorithm      
    """
    def __init__(self, min_cluster_size=5, alpha=0.5):
        _Classifier.__init__(self)
        self.min_cluster_size = min_cluster_size
        self.alpha = alpha

    def _compute_log_priors(self):
        self.log_priors = np.zeros(len(self._trained_prototypes))

    def train(self, clustered_ink_data, center_type='medoid', verbose=False):
        """Trains the classifier.

        Parameters
        ----------
        clustered_ink_data : dictionary
           Training data with the following structure.
           clustered_ink_data[label][cluster_id] = [ink1, ink2, ... inkN]

        center_type : {'medoid', 'centroid'}
           Type of the DTW center.
       
        """
        self._trained_prototypes = []
        alldist = []
        for label in clustered_ink_data.keys():
            for ink_list in clustered_ink_data[label]:
                if len(ink_list) > self.min_cluster_size:
                    ink_data, weights = zip(*ink_list)
                    proto = PrototypeDTW(label, alpha=self.alpha)
                    avgdist = proto.train(ink_data, 
                                          obs_weights=weights,
                                          center_type=center_type)
                    self._trained_prototypes.append(proto)
                    alldist.append(avgdist)
                    if verbose:
                        print ("Prototype for "
                               "%s (%d instances, avg.dist = %0.2f)"%(
                                label, len(ink_list), avgdist))
        self._compute_log_priors()
        
    def state_reduction(self, test_ink, n_iter=30):
        """Performs state reduction on the trained prototypes.

        Parameters
        ----------
        test_ink : list
           List of observations.

        n_iter : int
           Number of iterations to perfrom reduction.

        """
        test_ink_dict = {}
        for label, ink in test_ink:
            test_ink_dict.setdefault(label,[]).append(ink)

        prototype_dict = {}
        num_obs_dict = {}
        for prot_obj in self._trained_prototypes:
            prototype_dict.setdefault(prot_obj.label,
                                      []).append(prot_obj.model)
            num_obs_dict.setdefault(prot_obj.label,
                                    []).append(prot_obj.num_obs)

        # run the state reduction algorithm
        reduced_prototypes = _state_reduction(prototype_dict, test_ink_dict)
        
        # unpack the reduced prototypes
        trained_prototypes = []
        for label in reduced_prototypes.keys():
            for i,p in enumerate(reduced_prototypes[label]):
                prot_obj = PrototypeDTW(label, alpha=self.alpha)
                prot_obj.model = p
                prot_obj.num_obs = num_obs_dict[label][i]
                trained_prototypes.append(prot_obj)

        new_c = ClassifierDTW(min_cluster_size=self.min_cluster_size,
                              alpha=self.alpha)
        new_c.trained_prototypes = trained_prototypes
        return new_c

    def toJSON(self):
        info = super(ClassifierHMM,self).toJSON()
        info['prototype_type'] = 'DTW'
        return info


class ClassifierBeamDTW(ClassifierDTW):
    """Beam-search DTW classifier
    
    Parameters
    ----------
    min_cluster_size : int
      Minimum number of examples in a cluster.

    alpha : float
      Alpha for DTW algorithm (used in training)
        
    beam_width : int, None
      Number of states kept in memeory at each time step.

    beam_alpha : float
      Alpha for the beam decoding (used in classify) 

    """
    def __init__(self, min_cluster_size=5, alpha=0.5, 
                 beam_width=None, beam_alpha=0.5):
        ClassifierDTW.__init__(self, 
                               min_cluster_size=min_cluster_size, 
                               alpha=alpha)
        self.beam = None
        self.beam_width = beam_width
        self.beam_alpha = beam_alpha

    def train(self, clustered_ink_data, center_type='medoid'):
        super(ClassifierBeamDTW, self).train(clustered_ink_data, 
                                             center_type=center_type)
        self.beam = BeamSearchDTW(self.trained_prototypes,
                                  max_states=self.beam_width,
                                  alpha=self.beam_alpha)
    
    def test(self, label_ink_data):
        if len(self._trained_prototypes) == 0:
            raise ValueError('No prototypes in the classifier.')

        n = len(label_ink_data)
        true_labels, all_ink = zip(*label_ink_data)
        true_labels = np.asarray(true_labels)
        predicted = map(self.classify, all_ink)
        predicted = np.asarray(predicted)
        accuracy = 100.0 * np.sum(true_labels == predicted) / n
        return (accuracy, true_labels, predicted)

    def classify(self, obs):
        self.beam.reset()
        for i in range(obs.shape[0]):
            self.beam.add_point(obs[i,:])
        ll = self.beam.score()
        return self._trained_prototypes[np.argmax(ll)].label


    def state_reduction(self, test_ink, n_iter=30):
        """
        Returns a new instance of ClassifierDTW
        """
        test_ink_dict = {}
        for label, ink in test_ink:
            test_ink_dict.setdefault(label,[]).append(ink)

        prototype_dict = {}
        num_obs_dict = {}
        for prot_obj in self._trained_prototypes:
            prototype_dict.setdefault(prot_obj.label,
                                      []).append(prot_obj.model)
            num_obs_dict.setdefault(prot_obj.label,
                                    []).append(prot_obj.num_obs)

        # run the state reduction algorithm
        reduced_prototypes = _state_reduction(prototype_dict, test_ink_dict)
        
        # unpack the reduced prototypes
        trained_prototypes = []
        for label in reduced_prototypes.keys():
            for i,p in enumerate(reduced_prototypes[label]):
                prot_obj = PrototypeDTW(label, alpha=self.alpha)
                prot_obj.model = p
                prot_obj.num_obs = num_obs_dict[label][i]
                trained_prototypes.append(prot_obj)

        new_c = ClassifierBeamDTW(min_cluster_size=self.min_cluster_size,
                                  alpha=self.alpha,
                                  beam_width=self.beam_width,
                                  beam_alpha=self.beam_alpha)
        new_c.trained_prototypes = trained_prototypes
        new_c.beam = BeamSearchDTW(new_c.trained_prototypes,
                                   max_states=new_c.beam_width,
                                   alpha=new_c.beam_alpha)
        return new_c


    
class ClassifierHMM(_Classifier):
    """Classifier with HMM-based prototypes
    
    Attributes
    ----------
    min_cluster_size : int
       Minimum number of examples in a cluster
    
    """
    def __init__(self, min_cluster_size=10):
        _Classifier.__init__(self)
        self.min_cluster_size = min_cluster_size

    def _compute_log_priors(self):
        num_examples = np.array([float(proto.num_obs) 
                                 for proto in self._trained_prototypes])
        self.log_priors = np.log(num_examples / np.sum(num_examples))

    def train(self, clustered_ink_data, verbose=False):
        """Train the classifier from clustered_data
        
        Parameters
        ----------
        clustered_ink_data : dictionary
          clustered_ink_data[label] = [ [(ink,weight) from each cluster] ]
        """
        self._trained_prototypes = []
        for label in clustered_ink_data.keys():
            for ink_list in clustered_ink_data[label]:
                if len(ink_list) > self.min_cluster_size:
                    ink_data, weights = zip(*ink_list)
                    proto = PrototypeHMM(label)
                    loglike = proto.train(ink_data, obs_weights=weights)
                    self._trained_prototypes.append(proto)
                    if verbose:
                        print ("Prototype for "
                               "%s (%d instances, avg_ll = %0.1f)"%(
                                label, len(ink_list), loglike/len(ink_list)))
        self._compute_log_priors()
        
    def toJSON(self):
        info = super(ClassifierHMM,self).toJSON()
        info['prototype_type'] = 'HMM'
        return info
