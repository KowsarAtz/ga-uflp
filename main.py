from UFLPGeneticProblem import UFLPGeneticProblem

problem = UFLPGeneticProblem(orlibPath = '/tmp/ORLIB/ORLIB-uncap/', 
                             orlibDataset = '100/cap104',
                             maxGenerations = 2000,
                             nRepeatParams = (2,0.5)
                             )

problem.run()