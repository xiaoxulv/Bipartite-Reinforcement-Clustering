__author__ = 'Ariel'


import time

import numpy as np
import newEval

from sklearn.cluster import KMeans



X = M

k_means = KMeans(init='k-means++', n_clusters=200, n_init=10)
t0 = time.time()
k_means.fit(X)
t_batch = time.time() - t0
k_means_labels = k_means.labels_
k_means_cluster_centers = k_means.cluster_centers_
k_means_labels_unique = np.unique(k_means_labels)

with open("dev_docCluster_scikit.txt", 'w') as dc:
    for idx, item in enumerate(k_means_labels):
        dc.write("%d %d\n" % (idx,item))

print newEval.getF1("HW2_dev.gold_standards", "dev_docCluster.txt")
