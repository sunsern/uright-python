import numpy as np

def _max_score(obs, prot_list, offsets):
    """
    Returns the label of the prototype in `prot_list`
    with the maximum score.
    """
    all_scores = np.empty(len(prot_list))
    for i, prot_obj in enumerate(prot_list):
        score = prot_obj.score(obs)
        all_scores[i] = offsets[i] + score
    return prot_list[all_scores.argmax()].label

class _Classifier(object):
    """MAP classifier.
        
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
            proto_info['prior'] = float(np.exp(self.log_priors[i]))
            prototypes.append(proto_info)
        return {'prototypes':prototypes}
    
    def fromJSON(self, json_obj):
        pass
        
from _classifier_dtw import ClassifierDTW
from _classifier_hmm import ClassifierHMM
from _classifier_beam import ClassifierBeam
