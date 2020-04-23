import numpy as np
from gaUtils import *
from settings import *
from timeit import default_timer
from math import ceil

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
score = np.empty((population.shape[0], ))
rank = np.empty((population.shape[0], ))
bestIndividual = None
bestIndividualRepeatedTime = 0
bestPlanSoFar = []
nRepeat = ceil(10 * (totalCustomers * totalPotentialSites) ** 0.5)
generation = 1

while True:
    print('\rgeneration number %d               ' % generation, end='')
    updateScore(population, ELITE_SIZE, score, facilityToCustomerCost, potentialSitesFixedCosts)
    (population, score) = sortAll(population, score)
    if score[0] != bestIndividual:
        bestIndividualRepeatedTime = 0
        bestPlanSoFar = bestIndividualPlan(population, 0, facilityToCustomerCost)
        bestIndividual = score[0]
    bestIndividualRepeatedTime += 1
    if bestIndividualRepeatedTime > nRepeat or generation >= MAX_GENERATIONS: break
    updateRank(score, rank)
    punishDuplicates(population, rank)
    punishElites(rank, ELITE_SIZE)
    replaceWeaks(population, POPULATION_SIZE - ELITE_SIZE, rank, MUTATION_RATE, CROSSOVER_RATE)
    generation += 1

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

print('\rdataset name:',DATASET_FILE)
print('total generations of', generation)
print('best individual score',bestIndividual,\
      'repeated for last',bestIndividualRepeatedTime,'times')
if False not in compareToOptimal:
    print('REACHED OPTIMAL OF', optimalCost)
else:
    print('DID NOT REACHED OPTIMAL OF', optimalCost, "|",\
          (bestIndividual - optimalCost) * 100 / optimalCost,"% ERROR")
print('total elapsed time:', endTimeit - startTimeit)
assignedFacilitiesString = ''
for f in bestPlanSoFar:
    assignedFacilitiesString += str(f) + ' '
print('assigned facilities:')
print(assignedFacilitiesString)