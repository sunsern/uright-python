import numpy as np

from _beam_forward import BeamForward
from _classifier_dtw import ClassifierDTW

class ClassifierBeam(ClassifierDTW):
    """Classifier based on Beam-search forward algorithm
    
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

    def train(self, clustered_ink_data, 
              center_type='centroid', state_reduction=False):

        super(ClassifierBeam, self).train(clustered_ink_data, 
                                          center_type=center_type,
                                          state_reduction=state_reduction)
        self.labels = sorted(set([prot.label 
                                  for prot in self._trained_prototypes]))
        self.beam = BeamForward(self._trained_prototypes,
                                beam_width=self.beam_width,
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
        for i in xrange(obs.shape[0]):
            self.beam.add_point(obs[i,:])
        ll = self.beam.loglikelihood()
        return self._trained_prototypes[ll.argmax()].label

    def posterior(self, obs):

        def _label_prob(ll):
            probdict = {}
            for i,prot in enumerate(self._trained_prototypes):
                prob = probdict.get(prot.label, 0)
                probdict[prot.label] = prob + np.exp(ll[i])
            v = np.zeros(len(self.labels))
            for i,label in enumerate(self.labels):
                v[i] = probdict[label]
            return v / v.sum()

        post = np.zeros((obs.shape[0], len(self.labels)))
        self.beam.reset()
        for i in xrange(obs.shape[0]):
            self.beam.add_point(obs[i,:])
            post[i,:] = _label_prob(self.beam.loglikelihood())
        return post
