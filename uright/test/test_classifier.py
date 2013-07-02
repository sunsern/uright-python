import numpy as np
import unittest
import pickle
import random

from inkutils import (json2array,
                      normalize_ink,
                      filter_bad_ink)

from classifier import (ClassifierDTW,
                        ClassifierHMM,
                        ClassifierBeamDTW)

class _BaseTest(unittest.TestCase):
    def setUp(self):
        random.seed(12345)

        filename = "test/fixtures/candidate_prototypes_1368819209.p"
        candidate_proto = pickle.load(open(filename,"rb"))

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
        cDTW = ClassifierDTW(alpha=0.5)
        cDTW.train(self.clustered_data,center_type='medoid')
        accuracy,_,_ = cDTW.test(self.label_ink_pairs)
        print accuracy
        self.assertGreater(accuracy, 85.0)

    def test_centroid(self):
        cDTW = ClassifierDTW(alpha=0.5)
        cDTW.train(self.clustered_data,center_type='centroid')
        accuracy,_,_ = cDTW.test(self.label_ink_pairs)
        print accuracy
        self.assertGreater(accuracy, 87.0)

    def test_state_reduction(self):
        cDTW = ClassifierDTW(alpha=0.5)
        cDTW.train(self.clustered_data,center_type='medoid')
        before = cDTW.trained_prototypes[0].model.shape
        reduced_cDTW = cDTW.state_reduction(self.label_ink_pairs,
                                            n_iter=10)
        after = reduced_cDTW.trained_prototypes[0].model.shape
        accuracy,_,_ = reduced_cDTW.test(self.label_ink_pairs)
        print accuracy
        self.assertGreater(accuracy, 88.0)
        self.assertGreaterEqual(before, after)

class TestClassifierHMM(_BaseTest):
    def setUp(self):
        super(TestClassifierHMM,self).setUp()

    def test_basic(self):
        cHMM = ClassifierHMM()
        cHMM.train(self.clustered_data)
        accuracy,_,_ = cHMM.test(self.label_ink_pairs)
        print accuracy
        self.assertGreater(accuracy, 92.0)
        
class TestClassifierBeamDTW(_BaseTest):
    def setUp(self):
        super(TestClassifierBeamDTW,self).setUp()

    def test_basic(self):
        beam = ClassifierBeamDTW()
        beam.train(self.clustered_data)
        accuracy,_,_ = beam.test(self.label_ink_pairs)
        print accuracy
        self.assertGreater(accuracy, 85.0)
    
if __name__ == "__main__":
    unittest.main()
