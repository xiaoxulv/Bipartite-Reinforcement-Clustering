__author__ = 'Ariel'

import numpy as np
import kmeans



def BiClustering(data, k1, k2):

    word2Dcluster, doc2D = kmeans.Kmean(data, k1)
    base = "docCluster"
    s = [base+str(x) for x in xrange(20)]

    iter = 0
    while iter < 20:

        _, word2W = kmeans.Kmean(word2Dcluster.transpose(), k2)
        doc2Wcluster = np.zeros((data.shape[0], k2))
        for i in word2W.keys():
            doc2Wcluster[:,i] = np.mean(data.transpose()[word2W[i]], axis = 0)

        _, doc2D = kmeans.Kmean(doc2Wcluster, k1)
        word2Dcluster = np.zeros((k1, data.shape[1]))
        for i in doc2D.keys():
            word2Dcluster[i] = np.mean(data[doc2D[i]], axis = 0)

        # for debug here
        assign = np.array([-1]*data.shape[0])
        for key, value in doc2D.iteritems():
            for v in value:
                assign[v] = key
        with open(s[iter], 'w') as dc:
            for idx, item in enumerate(assign):
                dc.write("%d %d\n" % (idx,item))

        iter += 1

    return (doc2D, word2W)



