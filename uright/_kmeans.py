import numpy as np
import numpy.random

import clustering
from prototype import PrototypeHMM, PrototypeDTW
from classifier import ClassifierHMM, ClassifierDTW

def _closest_prototype(obs, prot_list):
    scores = np.ones(len(prot_list)) * -np.inf
    for i in range(len(prot_list)):
        prot_obj = prot_list[i]
        if prot_obj is not None:
            scores[i],_ = prot_obj.score(obs)
    return (scores.argmax(), scores.max())

def _should_terminate(partitions, threshold=2):
    N = partitions[0].shape[0]
    # cyclic condition
    for part in partitions[:-1]:
        if np.all(partitions[-1] == part):
            return True
    # number of label change is small
    return (np.sum(partitions[-1] == partitions[-2]) > N - threshold)


def _perform_kmeans(algorithm, label, obs, obs_weights=None, 
                    K=4, n_iter=10, init_partition=None,
                    min_cluster_size=10):

    def _rep(s, m):
        a, b = divmod(m, len(s))
        return s * a + s[:b]

    def _train(algorithm, label, obs, obs_weights):
        if algorithm == 'hmm':
            prot = PrototypeHMM(label)
            prot.train(obs, obs_weights=obs_weights)
        else:
            prot = PrototypeDTW(label)
            prot.train(obs, obs_weights=obs_weights, 
                       center_type='centroid')
        return prot

    N = len(obs)

    if obs_weights is None:
        obs_weights = np.ones(N)
    else:
        obs_weights = np.asarray(obs_weights)

    # randomly assign each example to a cluster
    if init_partition is None:
        init_partition = _rep(range(K), N)
        np.random.shuffle(init_partition)
        init_partition = np.array(init_partition, dtype=np.int)
        
    partitions = [ init_partition ]
    print partitions[-1]
    for it in range(n_iter):
        # Train prototypes using current cluster assignment
        trained_prototypes = []
        for i in range(K):
            clustered_obs = [obs[j] 
                             for j in range(N) 
                             if partitions[-1][j] == i]
            clustered_weights = obs_weights[partitions[-1] == i]
            
            if len(clustered_obs) > min_cluster_size:
                prot = _train(algorithm, label, clustered_obs, 
                              clustered_weights)
            else:
                prot = None
 
            trained_prototypes.append(prot)

        # compute a new cluster assignment
        new_partition = np.zeros(N, dtype=np.int)
        curr_ll = 0
        for i in range(N):
            (new_partition[i], ll) = _closest_prototype(obs[i], 
                                                        trained_prototypes)
            curr_ll += ll
            
        partitions.append(new_partition)
        print "ll = %0.2f"%(curr_ll)
        
        # check stop conditions
        if _should_terminate(partitions):
            break

    print (partitions[-1]+1)
    return (curr_ll, (partitions[-1] + 1), trained_prototypes)

def _partition_subset(packed_data):
    ink_data, weights = zip(*packed_data[0])
    label = packed_data[1]
    K = packed_data[2]
    min_cluster_size = packed_data[3]
    algorithm = packed_data[4]
    (_,partition,prots) = _perform_kmeans(algorithm,
                                          label, 
                                          ink_data,  
                                          obs_weights=weights, 
                                          K=K, 
                                          min_cluster_size=min_cluster_size)
    return (partition,prots)

class ClusterKMeans(clustering._BaseClusterer):
    """Cluster data with K-means algorithm"""
    def __init__(self, user_ink_data, target_user_id=None, 
                 min_cluster_size=10, maxclust=4, 
                 random_state=None, algorithm='hmm'):
        clustering._BaseClusterer.__init__(self, 
                                           user_ink_data, 
                                           target_user_id, 
                                           min_cluster_size, 
                                           maxclust)
        np.random.seed(random_state)
        
        if algorithm not in ('hmm','dtw'):
            raise ValueError('algorithm must be hmm or dtw.')
        
        self.algorithm = algorithm    
        
    def optimize_cluster_num(self, test_data, n_iter=30, 
                             threshold=0.001, dview=None):

        if self.algorithm == 'hmm':
            classifier_class = ClassifierHMM
        else:
            classifier_class = ClassifierDTW            

        # start with 1 prototype for each label
        n = len(self.labels)
        temp_n_clusters = np.ones(n, dtype=np.int) 
        cluster_info = self._partition_data(temp_n_clusters.tolist(),
                                            dview=dview)

        trained_prototypes = []
        for _,prots in cluster_info:
            for prot_obj in prots:
                if prot_obj is not None:
                    trained_prototypes.append(prot_obj)
                  
        curr_classifier = classifier_class()
        curr_classifier.trained_prototypes =  trained_prototypes
        (accuracy,_,_) = curr_classifier.test(test_data,dview=dview)
        error_rates = [1.0 - accuracy / 100.0]
        n_clusters = [temp_n_clusters.copy()]
        added_labels = []
        print error_rates, n_clusters[-1]

        # compute all candidates
        candidates = []
        for i in range(n):
            _,prots = _partition_subset((self.weighted_ink_data[i], 
                                         self.labels[i], 
                                         temp_n_clusters[i]+1,
                                         self.min_cluster_size,
                                         self.algorithm))
            # remove None from prots
            prots = [p for p in prots if p is not None]
            candidates.append(prots)        

        for it in range(n_iter):
            print "Iteration %d"%it
            # find the most beneficial class to increse prototype by 1
            min_error = 1.0
            min_idx = None
            best_classifier = None
            for i in range(len(self.labels)):
                if len(candidates[i]) < temp_n_clusters[i]+1:
                    # degenerated case. dont need to test
                    error = 1.0
                elif temp_n_clusters[i]+1 > self.maxclust:
                    # maxclust reached
                    error = 1.0
                else:
                    test_classifier = classifier_class()
                    prototypes = curr_classifier.trained_prototypes
                    # filter out all prototype with the label
                    prototypes = [p 
                                  for p in prototypes 
                                  if p.label != self.labels[i]]
                    prototypes += candidates[i]
                    test_classifier.trained_prototypes = prototypes
                    (accuracy,_,_) = test_classifier.test(test_data,dview=dview)
                    error = 1.0 - accuracy / 100.0
                    print "> %f"%error

                if (error < min_error):
                    min_error = error
                    min_idx = i
                    best_classifier = test_classifier
                    
            # Stop if no update found or error reduction is small
            if (min_idx is None) or (error_rates[-1] - min_error < threshold):
                print "no improvement. done."
                break
            
            # choose the min_idx
            temp_n_clusters[min_idx] += 1
            error_rates.append(min_error)
            added_labels.append(self.labels[min_idx])
            n_clusters.append(temp_n_clusters.copy())
            curr_classifier = best_classifier

            print error_rates
            print n_clusters
            print "choosing %s"%self.labels[min_idx]

            # replace the candidate if needed
            if (temp_n_clusters[min_idx]+1 <= self.maxclust):
                _,prots = _partition_subset((self.weighted_ink_data[min_idx], 
                                             self.labels[min_idx], 
                                             temp_n_clusters[min_idx]+1,
                                             self.min_cluster_size,
                                             self.algorithm))
                # remove None from prots
                prots = [p for p in prots if p is not None]
                candidates[min_idx] = prots

        self.n_clusters = n_clusters[-1]
        return (error_rates, added_labels)

    def _partition_data(self, n_clusters, dview=None):
        n = len(self.labels)
        packed_data = zip(self.weighted_ink_data, 
                          self.labels, 
                          n_clusters,
                          [self.min_cluster_size]*n,
                          [self.algorithm]*n)

        if dview is None:
            cluster_info = map(_partition_subset,packed_data)
        else:
            cluster_info = dview.map_sync(_partition_subset,packed_data)

        return cluster_info
    
    def clustered_data(self, dview=None):
        cluster_info = self._partition_data(self.n_clusters, dview=dview)
        data = {}
        for i in range(len(self.labels)):
            label = self.labels[i]            
            data[label] = []
            for k in range(self.n_clusters[i]):
                clustered_obs = [self.weighted_ink_data[i][j]
                                 for j in range(len(self.weighted_ink_data[i])) 
                                 if cluster_info[i][0][j] == (k+1)]
                data[label].append(clustered_obs)
        return data
