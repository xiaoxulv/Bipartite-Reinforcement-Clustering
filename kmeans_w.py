__author__ = 'Ariel'

import random
import numpy as np
from scipy.spatial import distance


def Kmean(data, k):
    "K-means clustering on data matrix, row vectors based"
    # Initialization
    seed = random.sample(xrange(data.shape[0]), k)
    centers = data[seed]
    newAssign = [0]*data.shape[0]
    assign = [1]*data.shape[0]
    # Iteration
    iter = 0
    while newAssign != assign and iter < 30:
        assign = newAssign
        clusters, newAssign = getCluster(data, centers)
        weights = getWeight(data, centers, clusters)
        centers = getCenter(data, clusters, weights)
        iter += 1
        #print iter

    return (np.array(centers), clusters)


def getCluster(data, centers):
    clusters = {}
    assign = np.argmax(1-distance.cdist(data, centers, 'cosine'), axis = 1)

    for (x, y) in zip(xrange(assign.shape[0]), assign):
        try:
            clusters[y].append(x)
        except:
            clusters[y] = [x]

    return (clusters, assign.tolist())

def getWeight(data, centers, clusters):
    weights = {}

    for i in clusters.keys():
        temp = 1-distance.cdist(data[clusters[i]], centers[i].reshape(1, centers.shape[1]),'cosine')
        temp = temp/np.linalg.norm(temp)
        weights[i] = temp

    return weights

def getCenter(data, clusters, weights):
    newCenters = np.zeros((len(clusters), data.shape[1]))
    idx = 0
    for i in clusters.keys():
        newCenters[idx] = np.mean(data[clusters[i]]*weights[i], axis = 0)
        idx += 1

    return newCenters

def sumOfCos(data, centers, clusters):
    res = 0.
    idx = 0
    for i in clusters.keys():
        res += np.sum(1-distance.cdist(data[clusters[i]], centers[idx].reshape(1,centers.shape[1]), 'cosine'))
        idx += 1
    return res