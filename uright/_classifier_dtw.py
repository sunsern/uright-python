import numpy as np

from classifier import _Classifier
from prototype import PrototypeDTW

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
        # We do not use the priors.
        self.log_priors = np.zeros(len(self._trained_prototypes))

    def train(self, clustered_ink_data, center_type='centroid', 
              verbose=False, state_reduction=False):
        """Trains the classifier.

        Parameters
        ----------
        clustered_ink_data : dictionary
           Training data with the following structure.
           clustered_ink_data[label][cluster_id] = [ink1, ink2, ... inkN]

        center_type : {'medoid', 'centroid'}
           Type of the DTW center.
           
        state_reduction : bool
           If state_reduction=True, state-reduction procedure is performed
           after each prototype is trained.
       
        """
        self._trained_prototypes = []
        for label in clustered_ink_data:
            for ink_list in clustered_ink_data[label]:
                if len(ink_list) > self.min_cluster_size:
                    ink_data, weights = zip(*ink_list)
                    proto = PrototypeDTW(label, alpha=self.alpha)
                    avg_score = proto.train(ink_data, 
                                            obs_weights=weights,
                                            center_type=center_type,
                                            state_reduction=state_reduction)
                    self._trained_prototypes.append(proto)
                    if verbose:
                        print ("Prototype for "
                               "%s (%d instances, avg.score = %0.2f)"%(
                                label, len(ink_list), avg_score))

        self._compute_log_priors()

    def toJSON(self):
        info = super(ClassifierDTW,self).toJSON()
        info['prototype_type'] = 'DTW'
        return info
