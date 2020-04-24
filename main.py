from UFLPGeneticProblem import UFLPGeneticProblem

problem = UFLPGeneticProblem(orlibPath = '/tmp/ORLIB/ORLIB-uncap/', 
                             orlibDataset = 'a-c/capb',
                             maxGenerations = 2000,
                             nRepeatParams = (2,0.5), 
                             mutationRate = 0.005,
                             crossoverRate = 0.3,
                             eliteFraction = 2/3
                             )

problem.run()