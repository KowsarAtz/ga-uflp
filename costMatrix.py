import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import sample

SAMPLES_NO = 200

inputFile = open('./reports/phaseTwoPartitioning/dataset/pa/csv/Dmatrix.txt', 'r')
rowsCount = int(inputFile.readline())
columnsCount = int(inputFile.readline())
costMatrix = np.array([int(i) for i in inputFile.readlines()]).reshape(rowsCount, columnsCount)
nodes = pd.read_csv('./reports/phaseTwoPartitioning/dataset/pa/csv/nodes.csv').iloc[:, 1:3].values

samples = nodes[sample(range(nodes.shape[0]), SAMPLES_NO)]

plt.scatter(nodes[:,0], nodes[:,1], marker='.', linewidth=1)
plt.scatter(samples[:,0], samples[:,1], marker='.', linewidth=1, color='red')
plt.show()