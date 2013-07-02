import numpy as np
import unittest
import pickle
import random

from inkutils import (json2array,
                      normalize_ink,
                      filter_bad_ink)

from classifier import ClassifierHMM
from clustering import ClusterKMeans

class _BaseTest(unittest.TestCase):
    def setUp(self):
        random.seed(12345)
        filename = "test/fixtures/rawink_1_1371107932.p"
        user_raw_ink = pickle.load(open(filename,"rb"))

        all_users = ['user_1', 'user_32', 'user_6', 'user_29']
        max_examples = 20

        user_ink_data = {}
        label_ink_pairs = []
        for userid in all_users:
            raw_ink = user_raw_ink[userid]
            normalized_ink = {}
            for label in ['a','u','v']: #raw_ink.keys():
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

class TestKMeans(_BaseTest):
    def setUp(self):
        super(TestKMeans,self).setUp()

    def test_basic(self):
        km = ClusterKMeans(self.user_ink_data)
        clustered_data = km.clustered_data()
        chmm = ClassifierHMM()
        chmm.train(clustered_data)
        accuracy,_,_ = chmm.test(self.label_ink_pairs)
        print accuracy
        self.assertGreater(accuracy, 90.0)

    def test_optimize(self):
        km = ClusterKMeans(self.user_ink_data)
        km.optimize_cluster_num(self.label_ink_pairs)
        clustered_data = km.clustered_data()
        chmm = ClassifierHMM()
        chmm.train(clustered_data)
        accuracy,_,_ = chmm.test(self.label_ink_pairs)
        print accuracy
        self.assertGreater(accuracy, 90.0)

if __name__ == "__main__":
    unittest.main()
