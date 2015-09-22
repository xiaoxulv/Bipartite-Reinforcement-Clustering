__author__ = 'Ariel'

import numpy as np
import time
import math
import kmeans
import biClustering
import newEval

start = time.time()

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
with open("HW2_dev.df") as df:
    for line in df:
        line = line.strip()
        idf.append(int(line.split(":")[1]))
for i, x in enumerate(idf):
   idf[i] = math.log(M.shape[0]/x+1)

for x in xrange(M.shape[1]):
    M[:,x] = M[:,x] * idf[x]



#kmeans.Kmean(M,50)
#kmeans.Kmean(M.transpose(),200)
doc2D, word2W = biClustering.BiClustering(M, 150, 800)
print(" %s seconds " % (time.time()-start))
print len(doc2D)

assign = np.array([-1]*M.shape[0])
for key, value in doc2D.iteritems():
    for v in value:
        assign[v] = key

with open("dev_docCluster.txt", 'w') as dc:
    for idx, item in enumerate(assign):
        dc.write("%d %d\n" % (idx,item))

print newEval.getF1("HW2_dev.gold_standards", "dev_docCluster.txt")

##########
assign = np.array([-1]*M.shape[1])
for key, value in word2W.iteritems():
    for v in value:
        assign[v] = key

wordDict = {}
with open("HW2_dev.dict",'r') as wd:
    for line in wd:
        line = line.strip()
        wordDict[int(line.split(" ")[1])] = line.split(" ")[0]




with open("dev_wordCluster.txt", 'w') as dc:
    for idx, item in enumerate(assign):
        dc.write("%s %d\n" % (wordDict[idx],item))


#f1 store for each iteration
res = []
base = "docCluster"
s = [base+str(x) for x in xrange(20)]
for x in xrange(20):
    temp = newEval.getF1("HW2_dev.gold_standards",s[x])
    print temp
    res.append(temp)
res = np.array(res)
print ("F1 mean is %f" % np.mean(res))


