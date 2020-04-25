from UFLPGeneticProblem import UFLPGeneticProblem

ITERATIONS = 10

datasets = []
# datasets += ['70/cap71','70/cap72','70/cap73','70/cap74']
datasets += ['100/cap101', '100/cap102', '100/cap103', '100/cap104']
# datasets += ['130/cap131', '130/cap132', '130/cap133', '130/cap134']
# datasets += ['a-c/capa', 'a-c/capb', 'a-c/capc']

problems = []
for dataset in datasets:
    for i in range(ITERATIONS):
        problem = UFLPGeneticProblem(orlibPath = '/tmp/ORLIB/ORLIB-uncap/', 
                                     orlibDataset = dataset,
                                     maxGenerations = 2000,
                                     nRepeatParams = (2,0.5), 
                                     mutationRate = 0.005,
                                     crossoverMaskRate = 0.3,
                                     eliteFraction = 2/3
                                     )
        
        problem.run()
        problems += [(dataset, i, problem)]
        
reached = {}
for dataset in datasets:
    reached[dataset] = 0
    
for tupl in problems:
    prob = tupl[2]
    if False not in prob.compareToOptimal:
        reached[tupl[0]] += 1