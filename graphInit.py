import pandas as pd
import numpy as np

class NodeAdjacencyList:
    def __init__(self, l):
        self.l = l
    def append(self, destination, cost):
        if self.has(destination): return
        self.l.append((destination, cost))
    def has(self, destination):
        for pair in self.l:
            if pair[0] == destination: return True
        return False
    @property
    def all(self):
        return self.l

nodes = pd.read_csv('./reports/phaseTwoPartitioning/dataset/pa/csv/nodes.csv').iloc[:, 1:3].values
realNodeIds = pd.read_csv('./reports/phaseTwoPartitioning/dataset/pa/csv/nodes_in_slovakia.csv').iloc[:, 1:].values
allEdges = pd.read_csv('./reports/phaseTwoPartitioning/dataset/sr/csv/edges.csv').iloc[:, [0,1,2,4]]
allEdges["oneway"] = allEdges["oneway"].replace("yes",True)
allEdges["oneway"] = allEdges["oneway"].replace("None",False)
allEdges["oneway"] = allEdges["oneway"].replace("no",False)
allEdges = allEdges.values

adjacencyArray = np.array([NodeAdjacencyList([]) for _ in range(nodes.shape[0])])

for sourceNodeIndex in range(nodes.shape[0]):
    realId = realNodeIds[sourceNodeIndex]
    destinations = allEdges[np.where(allEdges[:,0] == realId)]
    nodeAdjacencyList = adjacencyArray[sourceNodeIndex]
    for destination in destinations:
        destinationNodeIndex = np.where(realNodeIds == destination[1])[0]
        if destinationNodeIndex.shape[0] == 0: continue
        print("after")
        destinationNodeIndex = destinationNodeIndex[0]
        cost = destination[2]
        bidirectional = destination[3]
        nodeAdjacencyList.append(destinationNodeIndex, cost)
        if bidirectional:
            adjacencyArray[destinationNodeIndex].append(sourceNodeIndex, cost)