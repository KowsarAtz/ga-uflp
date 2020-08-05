from UFLPGeneticProblem import UFLPGeneticProblem as GA
from UFLPGAProblem import UFLPGAProblem as GA_T
from processORLIB import getCostMatrices, getOptimals, compareResults
import sys
import numpy as np

MAX_GENERATIONS = {
    'A': 20,
    'B': 200,
    'C': 1000,
    'D': 4000
}

args = sys.argv[:]
outputFileName = args[1]
f = open(outputFileName, 'w')
ITERATIONS = int(args[2])

datasets = []

# A: 16 * 50 (n. o. potential facility locations * n. o. customers)
# datasets += [('cap/40/cap4%d' % (i+1), MAX_GENERATIONS['A']) for i in range(4)]
# datasets += [('cap/50/cap51', MAX_GENERATIONS['A'])]
# datasets += [('cap/60/cap6%d' % (i+1), MAX_GENERATIONS['A']) for i in range(4)]
# datasets += [('uncap/70/cap7%d' % (i+1), MAX_GENERATIONS['A']) for i in range(4)]

# B: 25 * 50
# datasets += [('cap/80/cap8%d' % (i+1), MAX_GENERATIONS['B']) for i in range(4)]
# datasets += [('cap/90/cap9%d' % (i+1), MAX_GENERATIONS['B']) for i in range(4)]
# datasets += [('uncap/100/cap10%d' % (i+1), MAX_GENERATIONS['B']) for i in range(4)]

# C: 50 * 50
# datasets += [('cap/110/cap11%d' % (i+1), MAX_GENERATIONS['C']) for i in range(4)]
# datasets += [('cap/120/cap12%d' % (i+1), MAX_GENERATIONS['C']) for i in range(4)]
datasets += [('uncap/130/cap13%d' % (i+1), MAX_GENERATIONS['C']) for i in range(4)]

# D: 100 * 1000
# datasets += [('uncap/a-c/cap%s' % s, MAX_GENERATIONS['D']) for s in ['a', 'b', 'c']]

OPTIMAL = 'optimal'
BELOWP2 = 'below 0.2'
BELOW1 = 'below 1'
ABOVE1 = 'above 1'
REACHED = 'reached scores'
FAILED = 'failed scores'
ERRORS = 'error percs'
TIMES = 'elapsed times'
FIRSTREACHES = 'first reached generations'
MAXGENERATION = 'max generation'
BESTREPEATED = 'best individual repeated times'
OPTIMALPLAN = 'optimal plan'
OPTIMALCOST = 'optimal cost'
lastFaild = 'NULL'

reached = {}
for dataset in datasets:
    reached[dataset[0]] = {OPTIMAL: 0, BELOWP2: 0, BELOW1: 0, ABOVE1: 0,
                        BESTREPEATED: [], FIRSTREACHES: [], MAXGENERATION: dataset[1],
                        REACHED: [], FAILED: [], ERRORS: [], TIMES: []}
    (optimals, optimalCost) = getOptimals(orlibPath = './reports/ORLIB/ORLIB-', orlibDataset = dataset[0])
    reached[dataset[0]][OPTIMALCOST] = optimalCost
    reached[dataset[0]][OPTIMALPLAN] = optimals

totalReached = 0
totalFailed = 0
total = ITERATIONS * len(datasets)

for i in range(ITERATIONS):
    print("\niteration:", i+1, file=f)

    for dataset in datasets:   
        mxGen = dataset[1]
        dataset = dataset[0]

        (costMatrix, costVector) = getCostMatrices(orlibPath = './reports/ORLIB/ORLIB-', orlibDataset = dataset)
        individualLength = costVector.shape[0]
        individual = np.zeros((individualLength,), dtype=np.bool)
        for i in reached[dataset][OPTIMALPLAN]:
            individual[i] = True
        
        problem = GA(
            potentialSitesFixedCosts = costVector,
            facilityToCustomerCost = costMatrix,
            maxGenerations = mxGen,
            printProgress = True,
            problemTitle = dataset
            # stoppingIndividual = individual
        )
        # problem = GA_T(
        #     potentialSitesFixedCosts = costVector,
        #     facilityToCustomerCost = costMatrix,
        #     maxGenerations = mxGen
        #     # printProgress = True,
        #     # problemTitle = dataset
        #     # stoppingIndividual = individual
        # )
        problem.run()
        
        reachedOptimal = compareResults(
            orlibDatasetName = dataset,
            totalGeneration = problem.generation,
            bestFoundCost = problem.bestIndividualScore, 
            bestPlan = problem.bestPlan, 
            optimalCost = reached[dataset][OPTIMALCOST], 
            optimals = reached[dataset][OPTIMALPLAN],
            mainLoopElapsedTime = problem.mainLoopElapsedTime, 
            bestIndividualRepeatedTime = 0,
            bestFoundElapsedTime = 0,
            # bestIndividualRepeatedTime = problem.bestIndividualRepeatedTime, 
            # bestFoundElapsedTime = problem.bestFoundElapsedTime,
            fout = f
        )

        error = (problem.bestIndividualScore - reached[dataset][OPTIMALCOST]) * 100 / reached[dataset][OPTIMALCOST]
        reached[dataset][TIMES] += [problem.mainLoopElapsedTime]
        reached[dataset][BESTREPEATED] += [0]
        # reached[dataset][BESTREPEATED] += [problem.bestIndividualRepeatedTime]
        reached[dataset][FIRSTREACHES] += [mxGen - 0]
        # reached[dataset][FIRSTREACHES] += [mxGen - problem.bestIndividualRepeatedTime + 1]
        
        if reachedOptimal:
            totalReached += 1
            reached[dataset][OPTIMAL] += 1
            reached[dataset][REACHED] += [[]]
            # reached[dataset][REACHED] += [problem.score]
        else:
            totalFailed += 1
            lastFaild = dataset
            reached[dataset][FAILED] += [[]]
            # reached[dataset][FAILED] += [problem.score]
            reached[dataset][ERRORS] += [error]
            if error < 0.2:
                reached[dataset][BELOWP2] += 1
            elif error < 1:
                reached[dataset][BELOW1] += 1
            else:
                reached[dataset][ABOVE1] += 1
        
        print('\nTotal Reached %d/%d  Failed %d (last failed %s)\n' % (totalReached, total, totalFailed, lastFaild))

print('\n\nSUMMARY', file=f)

for key in reached:
    print('\ndataset:', key, file=f)
    data = reached[key]
    print(OPTIMAL, data[OPTIMAL], file=f)
    print(BELOWP2, data[BELOWP2], file=f)
    print(BELOW1, data[BELOW1], file=f)
    print(ABOVE1, data[ABOVE1], file=f)
    print(ERRORS, data[ERRORS], file=f)
    print(MAXGENERATION, data[MAXGENERATION], file=f)
    print('average elapsed time', sum(data[TIMES])/len(data[TIMES]), file=f)
    print('average first reached', sum(data[FIRSTREACHES])/len(data[FIRSTREACHES]), file=f)
    print('average best repeated time', sum(data[BESTREPEATED])/len(data[BESTREPEATED]), file=f)

