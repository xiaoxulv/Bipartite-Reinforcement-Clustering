__author__ = 'Ariel'

import numpy as np
import time
import math
import biClustering
import biClustering_wf
import newEval
import kmeans_w


start = time.time()

doc_idx = 0
row = []
col = []
weight = []
with open("HW2_test.docVectors",'r') as dv:
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
#print doc_idx
row = np.array(row).astype(np.int)
col = np.array(col).astype(np.int)
weight = np.array(weight).astype(np.int)
M = np.zeros((doc_idx, max(col)+1))

for (x,y,z) in zip(row, col, weight):
    M[x, y] = z

#print(" %s seconds " % (time.time()-start))

# normalize tf here
M = (M.transpose()/np.linalg.norm(M, axis=1)).transpose()
# read df here
idf = []
with open("HW2_test.df") as df:
    for line in df:
        line = line.strip()
        idf.append(int(line.split(":")[1]))
for i, x in enumerate(idf):
   idf[i] = math.log(M.shape[0]/x+1)
# multiply by idf
for x in xrange(M.shape[1]):
    M[:,x] = M[:,x] * idf[x]



# only 1 time
doc2D, word2W, cosD, cosW = biClustering.BiClustering(M, 200, 800)
assign = np.array([-1]*M.shape[0])
for key, value in doc2D.iteritems():
    for v in value:
        assign[v] = key

with open("test_docCluster.txt", 'w') as dc:
    for idx, item in enumerate(assign):
        dc.write("%d %d\n" % (idx,item))
# f1 = newEval.getF1("HW2_dev.gold_standards", "dev_docCluster.txt")
# print f1
# print cosD
# print cosW

# repeat for 10 times
# iter = 0
# f1s = []
# cosDs =[]
# cosWs = []
#
# while iter < 10:
#     doc2D, word2W, cosD, cosW = biClustering_wf.BiClustering(M, 200, 800)
#
#     #print(" %s seconds " % (time.time()-start))
#     #print len(doc2D)
#
#     assign = np.array([-1]*M.shape[0])
#     for key, value in doc2D.iteritems():
#         for v in value:
#             assign[v] = key
#
#     with open("_docCluster.txt", 'w') as dc:
#         for idx, item in enumerate(assign):
#             dc.write("%d %d\n" % (idx,item))
#     f1 = newEval.getF1("HW2_dev.gold_standards", "dev_docCluster.txt")
#     f1s.append(f1)
#
#     cosDs.append(cosD)
#     cosWs.append(cosW)
#     iter += 1
#
# for x in f1s:
#     print x
# print "!!!"
# for x in cosDs:
#     print x
# print "!!!"
# for x in cosWs:
#     print x
# print "!!!"
# print "The mean of F1 is %f" % np.mean(f1s)
# print "The variance of F1 is %f" % np.var(f1s)
# print "The max of F1 is %f" % np.max(f1s)
# print "The F1 of best sum of cosine similarity is %f" % f1s[np.argmax(cosDs)]
# print "The mean of cosW is %f" % np.mean(cosWs)
# print "The variance of cosW is %f" % np.var(cosWs)
# print "The max of cosW is %f" % np.max(cosWs)



# # get word clusters and write to file using word dictionary
# assign = np.array([-1]*M.shape[1])
# for key, value in word2W.iteritems():
#     for v in value:
#         assign[v] = key
#
# wordDict = {}
# with open("HW2_dev.dict",'r') as wd:
#     for line in wd:
#         line = line.strip()
#         wordDict[int(line.split(" ")[1])] = line.split(" ")[0]
#
#
# with open("dev_wordCluster.txt", 'w') as dc:
#     for idx, item in enumerate(assign):
#         dc.write("%s %d\n" % (wordDict[idx],item))


# #f1 store for each iteration through the Bipartite Clustering
# res = []
# base = "docCluster"
# s = [base+str(x) for x in xrange(20)]
# for x in xrange(20):
#     temp = newEval.getF1("HW2_dev.gold_standards",s[x])
#     print temp
#     res.append(temp)
# res = np.array(res)
# print ("F1 mean is %f" % np.mean(res))


