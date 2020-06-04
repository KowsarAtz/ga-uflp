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

sampleIndices = sample(range(nodes.shape[0]), SAMPLES_NO)
facilityToCustomerCost = costMatrix[sampleIndices]
facilityToCustomerCost = facilityToCustomerCost[:, [i for i in range(costMatrix.shape[1]) if i not in sampleIndices]]

facilityNodes = nodes[sampleIndices]
customerNodes = nodes[[i for i in range(costMatrix.shape[1]) if i not in sampleIndices]]

from UFLPGeneticProblem import UFLPGeneticProblem

ga = UFLPGeneticProblem(
    np.zeros((SAMPLES_NO, ), np.float64), #potentialSitesFixedCosts
    facilityToCustomerCost,
    mutationRate = 0.01,
    crossoverMaskRate = 0.4,
    eliteFraction = 1/3,
    populationSize = 150,
    cacheParam = 50,
    maxRank = 2.5,
    minRank = 0.712,
    maxGenerations = 2000,
    nRepeat = None,
    printProgress = True
)

ga.run()
bestIndividual = ga.population[0]
bestPlan = ga.bestIndividualPlan()

establishedFacilities = np.unique(ga.bestPlan)
for facility in establishedFacilities:
    coveredNodes = customerNodes[np.where(np.array(ga.bestPlan) == facility)]
    plt.scatter(nodes[:,0], nodes[:,1], marker='.', linewidth=1)
    plt.scatter(coveredNodes[:,0], coveredNodes[:,1], marker='.', linewidth=1, color='red')
    plt.scatter(facilityNodes[facility,0], facilityNodes[facility,1], marker='x', color='green')
    plt.show()
    
import seaborn as sns