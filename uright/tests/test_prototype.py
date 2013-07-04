import os.path
import numpy as np
import unittest
import pickle

from inkutils import (json2array,
                      normalize_ink,
                      filter_bad_ink)

from prototype import (PrototypeDTW,
                       PrototypeHMM)

class _BaseTest(unittest.TestCase):
    def setUp(self):
        fn = os.path.join(os.path.dirname(__file__), 'clustered_data.p')
        candidate_proto = pickle.load(open(fn,"rb"))
        # set test label
        self.label = 'u'
        clustered_data = []
        for _,examples in candidate_proto[self.label]:
            data = [np.nan_to_num(normalize_ink(json2array(ink)))
                    for ink in filter_bad_ink(examples)]
            clustered_data.append(data)
        # set test cluster
        self.ink_data = clustered_data[1]

class TestPrototypeDTW(_BaseTest):
    def setUp(self):
        super(TestPrototypeDTW,self).setUp()
        self.p = PrototypeDTW(self.label, alpha=0.5)
        self.p.train(self.ink_data)
        
    def test_serialization(self):
        p_data = self.p.toJSON()
        q = PrototypeDTW(None)
        q.fromJSON(p_data)
        self.assertEqual(p_data, q.toJSON())

    def test_score(self):
        score0,_ = self.p.score(self.ink_data[0])
        self.assertAlmostEqual(score0, -0.065, delta=1e-3)
        score1,_ = self.p.score(self.ink_data[1])
        self.assertAlmostEqual(score1, -0.060, delta=1e-3)
        
    def test_centroid(self):
        q = PrototypeDTW('u', alpha=0.5)
        q.train(self.ink_data, center_type="centroid")
        score0,_ = q.score(self.ink_data[0])
        self.assertAlmostEqual(score0, -0.063, delta=1e-3)


class TestPrototypeHMM(_BaseTest):
    def setUp(self):
        super(TestPrototypeHMM,self).setUp()
        self.p = PrototypeHMM(self.label, 
                              num_states=0.5, 
                              self_transprob=0.8, 
                              next_transprob=0.2, 
                              skip_transprob=1e-6)
        self.p.train(self.ink_data, max_N=15)
        
    def test_serialization(self):
        p_data = self.p.toJSON()
        q = PrototypeHMM(None)
        q.fromJSON(p_data)
        self.assertEqual(p_data, q.toJSON())

    def test_score(self):
        score0,_ = self.p.score(self.ink_data[0])
        self.assertAlmostEqual(score0, 47.772, delta=1e-3)
        score1,_ = self.p.score(self.ink_data[1])
        self.assertAlmostEqual(score1, 53.725, delta=1e-3)
        
if __name__ == "__main__":
    unittest.main()
