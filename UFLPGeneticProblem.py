import numpy as np
from math import ceil
from timeit import default_timer
from pylru import lrucache
from sys import stdout

class UFLPGeneticProblem:
    MAX_FLOAT = np.finfo(np.float64).max
    def __init__(
        self, 
        orlibPath,
        orlibDataset,
        outputFile = stdout,
        orlibCostValuePerLine = 7,
        # maxFacilities = None,
        populationSize = 150,
        eliteFraction = 2/3,
        maxRank = 2.5,
        minRank = 0.712,
        maxGenerations = 4000,
        mutationRate = 0.05,
        crossoverMaskRate = 0.3,
        nRepeatParams = None,
        cacheParam = 5,
        printSummary = True,
    ):

        self.orlibDataset = orlibDataset
        self.fout = outputFile
        self.printSummary = printSummary

        # GA Parameters
        self.populationSize = populationSize
        self.eliteSize = ceil(eliteFraction * self.populationSize)
        self.totalOffsprings = self.populationSize - self.eliteSize
        self.maxGenerations = maxGenerations
        self.mutationRate = mutationRate
        self.crossoverMaskRate = crossoverMaskRate
        # self.maxFacilities = maxFacilities
        
        # Cache
        self.cacheSize = cacheParam * self.eliteSize
        self.cache = lrucache(self.cacheSize)
        
        # Optimals
        f = open(orlibPath+orlibDataset+'.txt.opt', 'r')
        self.optimals = f.readline().split()
        self.optimalCost = float(self.optimals[-1])
        self.optimals = [int(string) for string in self.optimals[:-1]]
        
        # Input Data
        f = open(orlibPath+orlibDataset+'.txt', 'r')
        (self.totalPotentialSites, self.totalCustomers) = [int(string) for string in f.readline().split()]
        self.potentialSitesFixedCosts = np.empty((self.totalPotentialSites,))
        self.facilityToCustomerCost = np.empty((self.totalPotentialSites, self.totalCustomers))
        
        for i in range(self.totalPotentialSites):
            self.potentialSitesFixedCosts[i] = np.float64(f.readline().split()[1])
        
        for j in range(self.totalCustomers):
            self.demand = np.float64(f.readline())
            lineItems = f.readline().split()
            for i in range(self.totalPotentialSites):
                self.facilityToCustomerCost[i,j] = np.float64(lineItems[i%orlibCostValuePerLine])
                if i%orlibCostValuePerLine == orlibCostValuePerLine - 1:
                    lineItems = f.readline().split()

        # Rank Paramters
        self.maxRank = maxRank
        self.rankStep = (maxRank - minRank) / (self.populationSize - 1)

        # Population Random Initialization
        self.population = np.random.choice(a=[True, False], size=(self.populationSize, self.totalPotentialSites), p=[0.5, 0.5])
        self.offsprings = np.empty((self.totalOffsprings, self.totalPotentialSites))
        
        # GA Main Loop
        self.score = np.empty((self.populationSize, ))
        self.offspringsScore = np.empty((self.totalOffsprings, ))
        self.rank = np.ones((self.populationSize, ))
        self.fromPrevGeneration = np.zeros((self.populationSize, ), dtype=np.bool)
        self.bestIndividual = None
        self.bestIndividualRepeatedTime = 0
        self.bestPlanSoFar = []
        self.duplicateIndices = []
        self.nRepeat = None
        if nRepeatParams != None:
            self.nRepeat = ceil(self.nRepeatParams[0] * (self.totalCustomers * self.totalPotentialSites) ** self.nRepeatParams[1])
        self.generation = 1
        self.compareToOptimal = None
        self.errorPercentage = None
        self.mainLoopElapsedTime = None
        
        # PreScore Calculations
        for individualIndex in range(self.populationSize):
            self.score[individualIndex] = self.calculateScore(individualIndex)
                    
    def calculateScore(self, individualIndex=None, individual=None, cached=True):
        if individualIndex != None:
            individual = self.population[individualIndex, :]
        cacheKey = individual.tobytes()
        if cacheKey in self.cache:
            return self.cache.peek(cacheKey)
        openFacilites = np.where(individual == True)[0]
        if openFacilites.shape[0] == 0: 
            return UFLPGeneticProblem.MAX_FLOAT
        score = np.sum(np.amin(self.facilityToCustomerCost[openFacilites, :], axis=0))
        score += self.potentialSitesFixedCosts.dot(individual)
        if cached: self.cache[cacheKey] = score
        return score
    
    def sortAll(self):
        sortArgs = self.score.argsort()
        self.population = self.population[sortArgs]
        self.score = self.score[sortArgs]
        self.fromPrevGeneration = self.fromPrevGeneration[sortArgs]

    def calculateOffspringsScore(self):
        for individual in range(self.totalOffsprings):
            self.offspringsScore[individual] = self.calculateScore(individual=self.offsprings[individual])
    
    def sortOffsprings(self):
        sortArgs = self.offspringsScore.argsort()
        self.offsprings = self.offsprings[sortArgs]
        self.offspringsScore = self.offspringsScore[sortArgs]

    def uniformCrossoverOffspring(self, indexA, indexB):
        crossoverMask = np.random.choice(a=[True, False], size=(self.totalPotentialSites,), p=[self.crossoverMaskRate, 1-self.crossoverMaskRate])
        crossoverMaskComplement = np.invert(crossoverMask)
        parentA = self.population[indexA,:]
        parentB = self.population[indexB,:]
        return (
            crossoverMask * parentA + crossoverMaskComplement * parentB,
            crossoverMask * parentB + crossoverMaskComplement * parentA
        )
    
    def mutateOffsprings(self):      
        mutationRate = self.mutationRate
        mask =  np.random.choice(a=[True, False], size=(self.totalOffsprings, self.totalPotentialSites), p=[mutationRate, 1-mutationRate])
        self.offsprings = self.offsprings != mask
        
        
    def rouletteWheelParentSelection(self):
        rankSum = np.sum(self.rank)
        rand = np.random.uniform(low=0, high=rankSum)
        partialSum = 0
        for individualIndex in range(self.populationSize):
            partialSum += self.rank[individualIndex]
            if partialSum > rand:
                return individualIndex
    
    def replaceWeaks(self):
        # Selection and Crossover
        individual = 0
        while individual < self.totalOffsprings:
            parentIndexA = self.rouletteWheelParentSelection()
            parentIndexB = self.rouletteWheelParentSelection()
            while parentIndexA == parentIndexB : parentIndexB = self.rouletteWheelParentSelection() 
            offspringA, offspringB = self.uniformCrossoverOffspring(parentIndexA, parentIndexB)
            self.offsprings[individual, :] = offspringA
            self.offsprings[(individual + 1) % self.totalOffsprings, :] = offspringB
            individual += 2

        # Mutation
        self.mutateOffsprings()
        self.calculateOffspringsScore()
        self.sortOffsprings()


        # Replacement
        offspringsIndex = 0
        for dupIndex in self.duplicateIndices:
            self.population[dupIndex, :] = self.offsprings[offspringsIndex, :]
            self.score[dupIndex] = self.offspringsScore[offspringsIndex]
            self.fromPrevGeneration[dupIndex] = False
            offspringsIndex += 1

        populationIndex = self.populationSize - 1
        while  offspringsIndex < self.totalOffsprings:
            currentScore = self.score[populationIndex]
            newScore = self.offspringsScore[offspringsIndex]
            if newScore > currentScore:
                break
            self.population[populationIndex, :] = self.offsprings[offspringsIndex, :]
            self.score[populationIndex] = newScore
            self.fromPrevGeneration[populationIndex] = False
            populationIndex -= 1
            offspringsIndex += 1
    
    def bestIndividualPlan(self, individualIndex):
        openFacilites = np.where(self.population[individualIndex, :] == True)[0]
        plan = []
        for customerIndex in range(self.totalCustomers):
            openFacilityCosts = self.facilityToCustomerCost[openFacilites, customerIndex]
            chosenFacilityIndex = np.where(openFacilityCosts == np.min(openFacilityCosts))[0][0]
            plan += [openFacilites[chosenFacilityIndex]]
        return plan
                
    def punishElites(self):
        averageRank = np.average(self.rank)
        for individualIndex in range(self.populationSize):
            if self.fromPrevGeneration[individualIndex]:
                if self.rank[individualIndex] > averageRank:
                    self.rank[individualIndex] -= averageRank
                else:
                    self.rank[individualIndex] = 0
        
    def identicalIndividuals(self, indexA, indexB):
        return False not in (self.population[indexA, :] == self.population[indexB, :])
        
    def updateRank(self):
        self.duplicateIndices = []
        currentRank = self.maxRank
        self.rank[0] = currentRank
        for individualIndex in range(1,self.populationSize):
            currentRank -= self.rankStep
            if self.identicalIndividuals(individualIndex, individualIndex - 1):
                self.rank[individualIndex] = 0
                self.duplicateIndices = [individualIndex] + self.duplicateIndices
            else:
                self.rank[individualIndex] = currentRank    
    
    def markElites(self):
        self.fromPrevGeneration = np.ones((self.populationSize, ), dtype=np.bool)
        
    
    def compareBestFoundPlanToOptimalPlan(self):
        compare = []
        for i in range(len(self.optimals)):
            if self.optimals[i] == self.bestPlanSoFar[i]: 
                compare += [True]
            else: 
                compare += [False]
        return np.array(compare)
    
    def finsih(self):
        if self.nRepeat == None:
            return self.generation >= self.maxGenerations
        return self.bestIndividualRepeatedTime > self.nRepeat or\
             self.generation >= self.maxGenerations

    def run(self):
        
        # Start Timing
        startTimeit = default_timer()
        
        while True:
            if self.printSummary:
                print('\r' + self.orlibDataset, 'generation number %d' % self.generation, end='', file=stdout)
            self.sortAll()
            if self.score[0] != self.bestIndividual:
                self.bestIndividualRepeatedTime = 0
                self.bestIndividual = self.score[0]
            self.bestIndividualRepeatedTime += 1
            if self.finsih(): 
                self.bestPlanSoFar = self.bestIndividualPlan(0)
                break
            self.updateRank()
            self.punishElites()
            self.markElites()
            self.replaceWeaks()
            self.generation += 1
        
        # End Timing
        endTimeit = default_timer()
        self.mainLoopElapsedTime = endTimeit - startTimeit
        
        self.compareToOptimal = self.compareBestFoundPlanToOptimalPlan()
        if False in self.compareToOptimal:    
            self.errorPercentage = (self.bestIndividual - self.optimalCost) * 100 / self.optimalCost
        else:
            self.errorPercentage = 0
        
        if self.printSummary:
            print('\rdataset name:',self.orlibDataset, file=self.fout)
            print('total generations of', self.generation, file=self.fout)
            print('best individual score', self.bestIndividual,\
                  'repeated for last', self.bestIndividualRepeatedTime,'times', file=self.fout)
            if False not in self.compareToOptimal:
                print('REACHED OPTIMAL OF', self.optimalCost, file=self.fout)
            else:
                print('DID NOT REACHED OPTIMAL OF', self.optimalCost, "|",\
                      self.errorPercentage,"% ERROR", file=self.fout)
            print('total elapsed time:', self.mainLoopElapsedTime, file=self.fout)
            assignedFacilitiesString = ''
            for f in self.bestPlanSoFar:
                assignedFacilitiesString += str(f) + ' '
            print('assigned facilities:', file=self.fout)
            print(assignedFacilitiesString, file=self.fout)