__author__ = 'Ariel'

import numpy as np
import kmeans
import kmeans_w



def BiClustering(data, k1, k2):

    doc2Wcluster, word2W = kmeans_w.Kmean(data.transpose(), k2)
    # base = "docCluster"
    # s = [base+str(x) for x in xrange(20)]

    iter = 0
    while iter < 20:

        _, doc2D = kmeans_w.Kmean(doc2Wcluster, k1)
        word2Dcluster = np.zeros((len(doc2D), data.shape[1]))
        idx = 0
        for i in doc2D.keys():
            word2Dcluster[idx] = np.mean(data[doc2D[i]], axis = 0)
            idx += 1


        _, word2W = kmeans_w.Kmean(word2Dcluster.transpose(), k2)
        doc2Wcluster = np.zeros((data.shape[0], len(word2W)))
        idx = 0
        for i in word2W.keys():
            doc2Wcluster[:,idx] = np.mean(data.transpose()[word2W[i]], axis = 0)
            idx += 1


        # # for debug here
        # assign = np.array([-1]*data.shape[0])
        # for key, value in doc2D.iteritems():
        #     for v in value:
        #         assign[v] = key
        # with open(s[iter], 'w') as dc:
        #     for idx, item in enumerate(assign):
        #         dc.write("%d %d\n" % (idx,item))

        iter += 1
    cosD = kmeans_w.sumOfCos(data, word2Dcluster, doc2D)
    cosW = kmeans_w.sumOfCos(data.transpose(), doc2Wcluster.transpose(), word2W)

    return (doc2D, word2W, cosD, cosW)



