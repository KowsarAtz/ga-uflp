import numpy as np
from gaUtils import *
from settings import *

# Optimals
f = open(ORLIB_PATH+DATASET_FILE+'.txt.opt', 'r')
optimals = f.readline().split()
optimalCost = float(optimals[-1])
optimals = [int(string) for string in optimals[:-1]]

# Input Data
f = open(ORLIB_PATH+DATASET_FILE+'.txt', 'r')
(totalPotentialSites,totalCustomers) = [int(string) for string in f.readline().split()]
potentialSitesFixedCosts = np.empty((totalPotentialSites,))
facilityToCustomerUnitCost = np.empty((totalPotentialSites, totalCustomers))
facilityToCustomerCost = np.empty((totalPotentialSites, totalCustomers))

for i in range(totalPotentialSites):
    potentialSitesFixedCosts[i] = np.float64(f.readline().split()[1])

for j in range(totalCustomers):
    demand = np.float64(f.readline())
    lineItems = f.readline().split()
    for i in range(totalPotentialSites):
        facilityToCustomerUnitCost[i,j] = np.float64(lineItems[i%COST_VALUES_PER_LINE])/demand
        facilityToCustomerCost[i,j] = np.float64(lineItems[i%COST_VALUES_PER_LINE])
        if i%COST_VALUES_PER_LINE == COST_VALUES_PER_LINE - 1:
            lineItems = f.readline().split()

# Population Random Initialization
population = np.empty((POPULATION_SIZE, totalPotentialSites), np.bool)
for i in range(POPULATION_SIZE):
    for j in range(totalPotentialSites):
        if np.random.uniform() > 0.5:
            population[i,j] = True
        else:
            population[i,j] = False

# GA Main Loop
fitness = np.empty((population.shape[0], ))
rank = np.empty((population.shape[0], ), dtype=np.int16)
bestIndividual = []
bestPlanSoFar = []
for generation in range(MAX_GENERATIONS):
    updateFitness(population, fitness, facilityToCustomerCost, potentialSitesFixedCosts)
    (population, fitness) = sortAll(population, fitness)
    bestIndividual += [fitness[0]]
    bestPlanSoFar = bestIndividualPlan(population, 0, facilityToCustomerCost)
    offsprings = replaceWeaks(population, POPULATION_SIZE - ELITE_SIZE)

def compareBestFoundPlanToOptimalPlan(optimal, bestFound):
    compare = []
    for i in range(len(optimal)):
        if optimal[i] == bestFound[i]: 
            compare += [True]
        else: 
            compare += [False]
    return np.array(compare)

compareToOptimal = compareBestFoundPlanToOptimalPlan(optimals, bestPlanSoFar)


