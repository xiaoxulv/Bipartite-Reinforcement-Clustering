__author__ = 'Ariel'
import kmeans_sparse
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import distance
from scipy.sparse import dok_matrix
import timeit


doc_idx = 0
row = []
col = []
weight = []
with open("HW2_dev.docVectors",'r') as dv:
    for line in dv:
        line = line.rstrip()#remove the trailing newline
        pairs = line.split(" ")
        for pair in pairs:
            temp = pair.split(":")
            col.append(temp[0])
            weight.append(temp[1])
        for x in [doc_idx]*len(pairs):
            row.append(x)
        #print len(pairs)
        doc_idx += 1
print doc_idx

row = np.array(row).astype(np.int)
col = np.array(col).astype(np.int)
weight = np.array(weight).astype(np.int)
M = csr_matrix((weight, (row, col)))
# M.shape indicates the number of docs and unique words
print M.shape
# M nnz indicates the total words in the collection
print M.nnz

#M = dok_matrix(M)
#m = M[0:100,0:100]
timeit.timeit(kmeans_sparse.Kmeans(M,20),number=1)
