import numpy as np

from classifier import _Classifier
from prototype import PrototypeHMM
    
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
        for label in clustered_ink_data:
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
