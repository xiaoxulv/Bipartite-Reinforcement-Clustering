__author__ = 'Ariel'

import random
import numpy as np
from scipy.spatial import distance

def Kmean(data, k):
    centers = data[random.sample(xrange(data.shape[0]), k)]
    newCenters = data[random.sample(xrange(data.shape[0]), k)]

    while not ifConverged(centers, newCenters):
        print "!!"
        centers = newCenters
        clusters = getCluster(data, centers)
        newCenters = getNewCenter(centers, clusters)
        print "!"
    return (centers, clusters)

def getCluster(data, centers):
    clusters = {}
    for x in xrange(data.shape[0]):
        pair = [(i, distance.cosine(centers[i], data[x])) for i in xrange(centers.shape[0])]
        cur = min(pair, key = lambda t:t[1])[0]
        try:
            clusters[cur] = np.vstack([data[x], clusters[cur]])
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
    newCenters = np.zeros((0, clusters.values()[0].shape[1]))
    for i in clusters.keys():
        newCenters = np.vstack([np.mean(clusters[i], axis = 0),newCenters])
    return newCenters

def ifConverged(centers, newCenters):
    return set([tuple(i) for i in centers]) == set([tuple(i) for i in newCenters])