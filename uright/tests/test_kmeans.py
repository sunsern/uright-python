import os.path
import numpy as np
import unittest
import pickle
import random

from inkutils import (json2array,
                      normalize_ink,
                      filter_bad_ink)

from classifier import ClassifierHMM, ClassifierDTW
from clustering import ClusterKMeans

class _BaseTest(unittest.TestCase):
    def setUp(self):
        fn = os.path.join(os.path.dirname(__file__), 'raw_ink.p')
        user_raw_ink = pickle.load(open(fn,"rb"))
        
        # fix random seed
        random.seed(12345)

        all_users = ['user_1', 'user_32', 'user_6', 'user_29']
        max_examples = 10

        user_ink_data = {}
        label_ink_pairs = []
        for userid in all_users:
            raw_ink = user_raw_ink[userid]
            normalized_ink = {}
            for label in ['a','u','v']: 
                temp = [np.nan_to_num(normalize_ink(json2array(ink)))
                        for ink in filter_bad_ink(raw_ink[label])]
                if len(temp) > max_examples:
                    sampled_data = random.sample(temp, max_examples)
                else:
                    sampled_data = temp
                normalized_ink[label] = sampled_data
                label_ink_pairs += [(label,ink) for ink in sampled_data]
            user_ink_data[userid] = normalized_ink

        self.user_ink_data = user_ink_data
        self.label_ink_pairs = label_ink_pairs

class TestKMeansDTW(_BaseTest):
    def setUp(self):
        super(TestKMeansDTW,self).setUp()

    def test_simple(self):
        km = ClusterKMeans(self.user_ink_data,algorithm='dtw')
        clustered_data = km.clustered_data()
        cDTW = ClassifierDTW()
        cDTW.train(clustered_data, center_type='centroid')
        accuracy,_,_ = cDTW.test(self.label_ink_pairs)
        self.assertGreater(accuracy, 93.0)

    def test_optimize(self):
        km = ClusterKMeans(self.user_ink_data,algorithm='dtw')
        km.optimize_cluster_num(self.label_ink_pairs, verbose=False)
        clustered_data = km.clustered_data()
        cDTW = ClassifierDTW()
        cDTW.train(clustered_data, center_type='centroid')
        accuracy,_,_ = cDTW.test(self.label_ink_pairs)
        self.assertGreater(accuracy, 93.0)

    def test_specific_target(self):
        km = ClusterKMeans(self.user_ink_data,target_user_id='user_1',
                           algorithm='dtw')
        clustered_data = km.clustered_data()
        cDTW = ClassifierDTW()
        cDTW.train(clustered_data, center_type='centroid')
        accuracy,_,_ = cDTW.test(self.label_ink_pairs)
        self.assertGreater(accuracy, 93.0)


class TestKMeansHMM(_BaseTest):
    def setUp(self):
        super(TestKMeansHMM,self).setUp()

    def test_simple(self):
        km = ClusterKMeans(self.user_ink_data,
                           min_cluster_size=10)
        clustered_data = km.clustered_data()
        chmm = ClassifierHMM()
        chmm.train(clustered_data)
        accuracy,_,_ = chmm.test(self.label_ink_pairs)
        self.assertGreater(accuracy, 94.0)

    def test_optimize(self):
        km = ClusterKMeans(self.user_ink_data,
                           min_cluster_size=10)
        km.optimize_cluster_num(self.label_ink_pairs, verbose=False)
        clustered_data = km.clustered_data()
        chmm = ClassifierHMM()
        chmm.train(clustered_data)
        accuracy,_,_ = chmm.test(self.label_ink_pairs)
        self.assertGreater(accuracy, 94.0)

    def test_specific_target(self):
        km = ClusterKMeans(self.user_ink_data, 
                           target_user_id='user_1',
                           min_cluster_size=10)
        clustered_data = km.clustered_data()
        chmm = ClassifierHMM()
        chmm.train(clustered_data)
        accuracy,_,_ = chmm.test(self.label_ink_pairs)
        self.assertGreater(accuracy, 95.0)

if __name__ == "__main__":
    unittest.main()
