import numpy as np
import scipy.cluster as sp_cluster
import scipy.spatial.distance as distance

import clustering
from prototype import PrototypeDTW
from classifier import ClassifierDTW
from dtw import compute_dtw_distance

def _perform_linkage(obs, max_cluster, distmat=None, 
                     alpha=0.5, penup_z=10.0, 
                     algorithm='complete',
                     verbose=False):
    n = len(obs)
    
    if max_cluster == 1:
        return (np.ones(len(obs),dtype=np.int), None)

    if algorithm not in ('average','single','complete'):
        raise ValueError('algorithm must be either average, single, '
                         'or complte.')

    # calculate distance matrix
    if distmat is None:
        if verbose: print "calculating distmat"
        distmat = np.zeros((n,n))
        for i in range(n):
            for j in range(i+1,n):
                distmat[i,j] = compute_dtw_distance(obs[i],
                                                    obs[j],
                                                    alpha=alpha,
                                                    penup_z=penup_z)
                distmat[j,i] = distmat[i,j]

    z_avg = sp_cluster.hierarchy.linkage(distance.squareform(distmat), 
                                         method=algorithm)
    fc = sp_cluster.hierarchy.fcluster(z_avg,t=max_cluster,
                                       criterion='maxclust')
    return (np.asarray(fc), distmat)
    
def _partition_subset(packed_data, distmat=None, verbose=False):
    ink_data, _ = zip(*packed_data[0])
    label = packed_data[1]
    n_cluster = packed_data[2]
    (partition, distmat) = _perform_linkage(ink_data, n_cluster, distmat)
    if verbose:
        print "clustering %s"%label
        print partition
    return (partition, distmat)

def _train_prototypes(weighted_ink, partition, label, 
                      min_cluster_size=5, center_type='medoid'):
    obs, obs_weights = zip(*weighted_ink)
    obs_weights = np.asarray(obs_weights)
    trained_prototypes = []
    for k in range(np.max(partition)):
        clustered_obs = [obs[j] 
                         for j in range(len(obs))
                         if partition[j] == (k+1)]
        clustered_weights = obs_weights[partition == (k+1)]
        if len(clustered_obs) > min_cluster_size:
            p_dtw = PrototypeDTW(label)
            p_dtw.train(clustered_obs, 
                        obs_weights=clustered_weights, 
                        center_type=center_type)
            trained_prototypes.append(p_dtw)
    return trained_prototypes

def _train_all_prototypes(weighted_ink_list, cluster_info, labels, 
                          min_cluster_size=5, center_type='medoid'):
    trained_prototypes = []
    for i in range(len(labels)):
        trained_prototypes += _train_prototypes(
            weighted_ink_list[i],
            cluster_info[i][0],
            labels[i],
            min_cluster_size=min_cluster_size,
            center_type=center_type)
    return trained_prototypes


class ClusterLinkage(clustering._BaseClusterer):
    """Cluster data with Linkage algorithm"""
    def optimize_cluster_num(self, test_data, n_iter=30, 
                             threshold=0.001, dview=None, verbose=False):
        # start with 1 prototype for each label
        temp_n_clusters = np.ones(len(self.labels),dtype=np.int) 
        cluster_info = self._partition_data(temp_n_clusters.tolist(),
                                            dview=dview)
        distmats = [distmat for _,distmat in cluster_info]
        trained_prototypes = _train_all_prototypes(self.weighted_ink_data,
                                                   cluster_info, 
                                                   self.labels)
        curr_classifier = ClassifierDTW()
        curr_classifier.trained_prototypes = trained_prototypes

        (accuracy,_,_) = curr_classifier.test(test_data,dview=dview)
        error_rates = [1.0 - accuracy / 100.0]
        n_clusters = [temp_n_clusters.copy()]
        added_labels = []
        if verbose: print error_rates

        # compute all candidates
        candidates = []
        for i in range(len(self.labels)):
            partition,_ = _partition_subset((self.weighted_ink_data[i], 
                                             self.labels[i], 
                                             temp_n_clusters[i]+1),
                                            distmat=distmats[i],
                                            verbose=verbose)
            prototypes = _train_prototypes(self.weighted_ink_data[i], 
                                           partition, 
                                           self.labels[i])
            if verbose: print "candidate for %s has length %d"%(
                self.labels[i], len(prototypes))
            candidates.append(prototypes)        

        for it in range(n_iter):
            if verbose: print "Iteration %d"%it
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
                    test_classifier = ClassifierDTW()
                    prototypes = curr_classifier.trained_prototypes
                    # filter out all prototypes with the label
                    prototypes = [p 
                                  for p in prototypes 
                                  if p.label != self.labels[i]]
                    prototypes += candidates[i]
                    test_classifier.trained_prototypes = prototypes
                    (accuracy,_,_) = test_classifier.test(test_data, 
                                                          dview=dview)
                    error = 1.0 - accuracy / 100.0
                    if verbose: print "> %f"%error

                if (error < min_error):
                    min_error = error
                    min_idx = i
                    best_classifier = test_classifier
                    
            # Stop if no update found or error reduction is small
            if (min_idx is None) or (error_rates[-1] - min_error < threshold):
                if verbose: print "no improvement. done."
                break
            
            # choose the min_idx
            temp_n_clusters[min_idx] += 1
            error_rates.append(min_error)
            added_labels.append(self.labels[min_idx])
            n_clusters.append(temp_n_clusters.copy())
            curr_classifier = best_classifier

            if verbose:
                print error_rates
                print n_clusters
                print "choosing %s"%self.labels[min_idx]

            # replace the candidate if needed
            if (temp_n_clusters[min_idx]+1 <= self.maxclust):
                partition,_ = _partition_subset(
                    (self.weighted_ink_data[min_idx], 
                     self.labels[min_idx], 
                     temp_n_clusters[min_idx]+1),
                    distmat=distmats[min_idx], 
                    verbose=verbose)
                prototypes = _train_prototypes(
                    self.weighted_ink_data[min_idx],
                    partition, 
                    self.labels[min_idx])
                if verbose:
                    print "candidate for %s has length %d"%(
                        self.labels[min_idx], len(prototypes))
                candidates[min_idx] = prototypes

        self.n_clusters = n_clusters[-1]
        return (error_rates, added_labels)
        
    def _partition_data(self, n_clusters, dview=None):
        packed_ink_data = zip(self.weighted_ink_data, 
                              self.labels, 
                              n_clusters)
        if dview is None:
            cluster_info = map(_partition_subset,packed_ink_data)
        else:
            cluster_info = dview.map_sync(_partition_subset,packed_ink_data)

        return cluster_info


    def clustered_data(self, dview=None):
        cluster_info = self._partition_data(self.n_clusters, dview=dview)
        data = {}
        for i in range(len(self.labels)):
            label = self.labels[i]            
            data[label] = []
            for k in range(self.n_clusters[i]):
                clustered_obs = [
                    self.weighted_ink_data[i][j]
                    for j in range(len(self.weighted_ink_data[i])) 
                    if cluster_info[i][0][j] == (k+1)]
                data[label].append(clustered_obs)
        return data
