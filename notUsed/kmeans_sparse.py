__author__ = 'Ariel'

import random
import numpy as np
from scipy.sparse import vstack
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm

def Kmeans(data, k):
    # randomly initialize the centroids
    newCenters = data[random.sample(xrange(data.shape[0]), k)]
    centers = data[random.sample(xrange(data.shape[0]), k)]
    while not ifConverged(centers, newCenters):
        print "!!"
        centers = newCenters
        clusters = getCluster(data, centers)
        newCenters = getNewCenter(centers, clusters)
        print "!"

    return (newCenters, clusters)

def getCluster(data, centers):
    clusters = {}
    for x in xrange(data.shape[0]):
        pair = [(i, getCosine(centers[i], data[x])) for i in xrange(centers.shape[0])]
        cur = min(pair, key = lambda t:t[1])[0]
        try:
            clusters[cur] = vstack([data[x], clusters[cur]])
        except:
            clusters[cur] = data[x]
#    # empty clusters
#    if len(clusters) != len(centers):
#        empty = [x for x in centers if x not in clusters.keys()]
#        for x in empty:
#            clusters[clusters.keys()[0]] = clusters[clusters.keys()[0]][:-1]
#            clusters[x] = clusters,keys()[0][-1]
    return clusters


def getNewCenter(centers, clusters):
    newCenters = csr_matrix((0, clusters.values()[0].shape[1]))
    for i in clusters.keys():
        newCenters = vstack([csr_matrix(clusters[i].mean(axis = 0)), newCenters])
    return newCenters

def ifConverged(centers, newCenters):
    return set(centers) == set(newCenters)


def getCosine(x1, x2):
    return 1 - x1.dot(x2.transpose())[0,0]/(norm(x1)*norm(x2))



