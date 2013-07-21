import os.path
import numpy as np
import unittest
import pickle
import random

from inkutils import (json2array,
                      normalize_ink,
                      filter_bad_ink)

from classifier import (ClassifierDTW,
                        ClassifierHMM,
                        ClassifierBeam)

VERBOSE=False

class _BaseTest(unittest.TestCase):
    def setUp(self):
        fn = os.path.join(os.path.dirname(__file__), 
                          'fixtures/clustered_data.p')
        candidate_proto = pickle.load(open(fn,"rb"))
        
        # fix random seed
        random.seed(12345)

        max_examples = 20
        clustered_data = {}
        label_ink_pairs = []
        for label in ['a','u','v']:
            clustered_data[label] = []
            for _,examples in candidate_proto[label]:
                data = [np.nan_to_num(normalize_ink(json2array(ink)))
                        for ink in filter_bad_ink(examples)]

                if len(data) > max_examples:
                    sampled_data = random.sample(data, max_examples)
                else:
                    sampled_data = data

                weights = [1] * len(sampled_data)
                clustered_data[label].append(zip(sampled_data, weights))
                label_ink_pairs += [(label,ink) for ink in sampled_data]

        self.clustered_data = clustered_data
        self.label_ink_pairs = label_ink_pairs

class TestClassifierDTW(_BaseTest):
    def setUp(self):
        super(TestClassifierDTW,self).setUp()

    def test_medoid(self):
        cDTW = ClassifierDTW(alpha=0.5,min_cluster_size=10)
        cDTW.train(self.clustered_data,center_type='medoid')
        accuracy,_,_ = cDTW.test(self.label_ink_pairs)
        if VERBOSE: print accuracy, 86
        self.assertGreater(accuracy, 86.0)

    def test_centroid(self):
        cDTW = ClassifierDTW(alpha=0.5,min_cluster_size=10)
        cDTW.train(self.clustered_data,center_type='centroid')
        accuracy,_,_ = cDTW.test(self.label_ink_pairs)
        if VERBOSE: print accuracy, 87
        self.assertGreater(accuracy, 87.0)

    def test_state_reduction(self):
        cDTW = ClassifierDTW(alpha=0.5,min_cluster_size=10)
        cDTW.train(self.clustered_data,center_type='medoid',
                   state_reduction=True)
        accuracy,_,_ = cDTW.test(self.label_ink_pairs)
        if VERBOSE: print accuracy, 87
        self.assertGreater(accuracy, 87.0)

class TestClassifierHMM(_BaseTest):
    def setUp(self):
        super(TestClassifierHMM,self).setUp()

    def test_simple(self):
        cHMM = ClassifierHMM(min_cluster_size=10)
        cHMM.train(self.clustered_data)
        accuracy,_,_ = cHMM.test(self.label_ink_pairs)
        if VERBOSE: print accuracy, 92
        self.assertGreater(accuracy, 92.0)
        
class TestClassifierBeam(_BaseTest):
    def setUp(self):
        super(TestClassifierBeam,self).setUp()

    def test_simple(self):
        beam = ClassifierBeam(beam_width=500, 
                              beam_alpha=0.1)
        beam.train(self.clustered_data)
        accuracy,_,_ = beam.test(self.label_ink_pairs)
        if VERBOSE: print accuracy, 87
        self.assertGreater(accuracy, 87.0)

    def test_state_reduction(self):
        beam = ClassifierBeam(beam_width=500, 
                              beam_alpha=0.1)
        beam.train(self.clustered_data, state_reduction=True)
        accuracy,_,_ = beam.test(self.label_ink_pairs)
        if VERBOSE: print accuracy, 90
        self.assertGreater(accuracy, 90.0)
    
if __name__ == "__main__":
    unittest.main()
