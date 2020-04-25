from UFLPGeneticProblem import UFLPGeneticProblem

datasets = []
# datasets += ['70/cap71','70/cap72','70/cap73','70/cap74']
# datasets += ['100/cap101', '100/cap102', '100/cap103', '100/cap104']
# datasets += ['130/cap131', '130/cap132', '130/cap133', '130/cap134']
# datasets += ['a-c/capa', 'a-c/capb', 'a-c/capc']
datasets += ['a-c/capa']


OPTIMAL = 'optimal'
BELOWp2 = 'below 0.2'
BELOW1 = 'below 1'
ABOVE1 = 'above 1'
REACHED = 'reached scores'
FAILED = 'failed scores'
lastFaild = ''
ITERATIONS = 5

reached = {}
for dataset in datasets:
    reached[dataset] = {OPTIMAL: 0, BELOWp2: 0, BELOW1: 0, ABOVE1: 0,
                        REACHED: [], FAILED: []}

# problems = []
failedScores = []
reachedScores = []
totalReached = 0
totalFailed = 0
total = ITERATIONS * len(datasets)

for i in range(ITERATIONS):
    for dataset in datasets:    
        problem = UFLPGeneticProblem(orlibPath = '/tmp/ORLIB/ORLIB-uncap/', 
                                     orlibDataset = dataset,
                                     maxGenerations = 2000,
                                     nRepeatParams = (2,0.5), 
                                     mutationRate = 0.005,
                                     crossoverMaskRate = 0.3,
                                     eliteFraction = 2/3,
                                     printSummary = True,
                                     populationSize = 150
                                     )
        
        problem.run()
        
        error = problem.errorPercentage
        if error == 0:
            totalReached += 1
            reached[dataset][OPTIMAL] += 1
            reached[dataset][REACHED] += [problem.score]
        else:
            totalFailed += 1
            lastFaild = dataset
            reached[dataset][FAILED] += [problem.score]
            if error < 0.2:
                reached[dataset][BELOWp2] += 1
            elif error < 1:
                reached[dataset][BELOW1] += 1
            else:
                reached[dataset][ABOVE1] += 1
        
        # problems += [(dataset, i, problem)]
        # print('\rTotal Reached %d/%d  Failed %d (last failed %s)                     ' \
        #       % (totalReached, total, totalFailed, lastFaild), end = '')

    