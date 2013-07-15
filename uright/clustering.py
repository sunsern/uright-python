import numpy as np

def _target_weight(user_ink_data, target_id, label):
    target_count = len(user_ink_data[target_id][label])
    nontarget_count = np.sum([len(user_ink_data[uid][label]) 
                              for uid in user_ink_data 
                              if uid != target_id])
    return float(nontarget_count) / target_count

class _BaseClusterer(object):
    """Base class for clustering algorithm

    Parameters
    ----------
    user_ink_data : dictionary
       user_ink_data[user_id][label] = [ink_1, ..., ink_n]

    target_user_id : string
       A specific key of `user_ink_data`
     
    min_cluster_size : int
       Minimum number of examples per cluster
 
    maxclust : int
       Maximum number of clusters per label

    Attributes
    ----------
    labels : list
       List of all labels

    n_clusters : array, shape=(1,len(labels))
       Number of clusteres of each label

    weighted_ink_data : list of (ink, weight)

    """
    def __init__(self, user_ink_data, target_user_id=None, 
                 min_cluster_size=10, maxclust=4, 
                 target_weight_multiplier=1.0,
                 equal_total_weight=True):
        self.user_ink_data = user_ink_data
        self.target_user_id = target_user_id
        self.min_cluster_size = min_cluster_size
        self.maxclust = maxclust
   
        # extract unique labels
        all_labels = []
        for userid in user_ink_data:
            all_labels += user_ink_data[userid].keys()
        self.labels = sorted(set(all_labels))
        
        # initialize the number of cluster to 'maxclust' for all labels
        self.n_clusters = np.ones(len(self.labels),dtype=np.int) * maxclust

        # compute the weighted ink data
        self.weighted_ink_data = []
        for label in self.labels:
            if target_user_id is None:
                target_weight = 1.0
            elif equal_total_weight:
                target_weight = _target_weight(user_ink_data, 
                                               target_user_id,
                                               label)
            else:
                target_weight = 1.0

            weighted_ink = []
            for userid in user_ink_data:
                if userid == target_user_id:
                    w = target_weight * target_weight_multiplier
                else:
                    w = 1.0
                weighted_ink += [(ink, w) 
                                 for ink in user_ink_data[userid][label]]
            self.weighted_ink_data.append(weighted_ink)

    def optimize_cluster_num(self):
        pass
    
    def clustered_data(self):
        pass

from _kmeans import ClusterKMeans
from _linkage import ClusterLinkage
