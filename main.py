import numpy as np
from gaUtils import *
from settings import *
from timeit import default_timer

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

# Start Timing
startTimeit = default_timer()

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
bestIndividual = None
bestIndividualRepeatedTime = 0
bestPlanSoFar = []

for generation in range(MAX_GENERATIONS):
    updateFitness(population, fitness, facilityToCustomerCost, potentialSitesFixedCosts)
    (population, fitness) = sortAll(population, fitness)
    if fitness[0] != bestIndividual:
        bestIndividualRepeatedTime = 0
    bestIndividual = fitness[0]
    bestIndividualRepeatedTime += 1
    bestPlanSoFar = bestIndividualPlan(population, 0, facilityToCustomerCost)
    offsprings = replaceWeaks(population, POPULATION_SIZE - ELITE_SIZE)

# End Timing
endTimeit = default_timer()

def compareBestFoundPlanToOptimalPlan(optimal, bestFound):
    compare = []
    for i in range(len(optimal)):
        if optimal[i] == bestFound[i]: 
            compare += [True]
        else: 
            compare += [False]
    return np.array(compare)

compareToOptimal = compareBestFoundPlanToOptimalPlan(optimals, bestPlanSoFar)

print('dataset name:',DATASET_FILE)
print('total generations of', MAX_GENERATIONS)
print('best individual fitness',bestIndividual,\
      'repeated for last',bestIndividualRepeatedTime,'times')
if False not in compareToOptimal:
    print('REACHED OPTIMAL OF', optimalCost)
else:
    print('DID NOT REACHED OPTIMAL OF', optimalCost)
print('total elapsed time:', endTimeit - startTimeit)
assignedFacilitiesString = ''
for f in bestPlanSoFar:
    assignedFacilitiesString += str(f) + ' '
print('assigned facilities:')
print(assignedFacilitiesString)