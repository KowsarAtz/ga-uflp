import numpy as np
import pandas as pd

inputFile = open('./reports/phaseTwoPartitioning/dataset/pa/csv/Dmatrix.txt', 'r')
rowsCount = int(inputFile.readline())
columnsCount = int(inputFile.readline())
costMatrix = np.array([int(i) for i in inputFile.readlines()]).reshape(rowsCount, columnsCount)
nodes = pd.read_csv('./reports/phaseTwoPartitioning/dataset/pa/csv/nodes.csv').iloc[:, 1:3].values
