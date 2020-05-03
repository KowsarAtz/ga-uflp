from UFLPGeneticProblem import UFLPGeneticProblem as GA
import sys

args = sys.argv[:]
outputFileName = args[1]
f = open(outputFileName, 'w')
ITERATIONS = int(args[2])
CACHE = int(args[3])

datasets = []
datasets += ['70/cap71','70/cap72','70/cap73','70/cap74']
# datasets += ['100/cap101', '100/cap102', '100/cap103', '100/cap104']
# datasets += ['130/cap131', '130/cap132', '130/cap133', '130/cap134']
# datasets += ['a-c/capa', 'a-c/capb', 'a-c/capc']

OPTIMAL = 'optimal'
BELOWP2 = 'below 0.2'
BELOW1 = 'below 1'
ABOVE1 = 'above 1'
REACHED = 'reached scores'
FAILED = 'failed scores'
ERRORS = 'error percs'
TIMES = 'elapsed times'
lastFaild = 'NULL'

reached = {}
for dataset in datasets:
    reached[dataset] = {OPTIMAL: 0, BELOWP2: 0, BELOW1: 0, ABOVE1: 0,
                        REACHED: [], FAILED: [], ERRORS: [], TIMES: []}

totalReached = 0
totalFailed = 0
total = ITERATIONS * len(datasets)

for i in range(ITERATIONS):
    print("\niteration:", i+1, file=f)

    for dataset in datasets:    
        problem = GA(
            orlibPath = './reports/ORLIB/ORLIB-uncap/', 
            orlibDataset = dataset,
            outputFile = f,
            maxGenerations = 2000,
            nRepeatParams = (2,0.5), 
            mutationRate = 0.005,
            crossoverMaskRate = 0.3,
            eliteFraction = 1/3,
            printSummary = True,
            populationSize = 150,
            cacheParam = CACHE
        )
        
        problem.run()
        
        error = problem.errorPercentage
        reached[dataset][TIMES] += [problem.mainLoopElapsedTime]
        if error == 0:
            totalReached += 1
            reached[dataset][OPTIMAL] += 1
            reached[dataset][REACHED] += [problem.score]
        else:
            totalFailed += 1
            lastFaild = dataset
            reached[dataset][FAILED] += [problem.score]
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
    print('average elapsed time', sum(data[TIMES])/len(data[TIMES]), file=f)
    