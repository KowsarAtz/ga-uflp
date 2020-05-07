from UFLPGeneticProblem import UFLPGeneticProblem as GA
import sys

args = sys.argv[:]
outputFileName = args[1]
f = open(outputFileName, 'w')
ITERATIONS = int(args[2])
CACHE = int(args[3])

datasets = []

datasets += ['cap/40/cap4%d' % (i+1) for i in range(4)]
datasets += ['cap/50/cap51']
datasets += ['cap/60/cap6%d' % (i+1) for i in range(4)]
datasets += ['cap/80/cap8%d' % (i+1) for i in range(4)]
datasets += ['cap/90/cap9%d' % (i+1) for i in range(4)]
datasets += ['cap/110/cap11%d' % (i+1) for i in range(4)]
datasets += ['cap/120/cap12%d' % (i+1) for i in range(4)]

datasets += ['uncap/70/cap7%d' % (i+1) for i in range(4)]
datasets += ['uncap/100/cap10%d' % (i+1) for i in range(4)]
datasets += ['uncap/130/cap13%d' % (i+1) for i in range(4)]
datasets += ['uncap/a-c/cap%s' % s for s in ['a', 'b', 'c']]

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
            orlibPath = './reports/ORLIB/ORLIB-', 
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
