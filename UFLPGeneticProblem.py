import numpy as np
from math import ceil
from timeit import default_timer
from pylru import lrucache

class UFLPGeneticProblem:
    def __init__(self, orlibPath, orlibDataset, orlibCostValuePerLine = 7, populationSize = 150, eliteFraction = 2/3, maxGenerations = 4000, mutationRate = 0.05, crossoverMaskRate = 0.3, nRepeatParams = (10,0.5), cacheParam = 5, printSummary = True):
        self.orlibDataset = orlibDataset
        self.printSummary = printSummary
        # GA Parameters
        self.populationSize = populationSize
        self.eilteSize = ceil(eliteFraction * self.populationSize)
        self.totalOffsprings = self.populationSize - self.eilteSize
        self.maxGenerations = maxGenerations
        self.mutationRate = mutationRate
        self.crossoverMaskRate = crossoverMaskRate
        self.nRepeatParams = nRepeatParams
        
        # Cache
        self.cacheSize = cacheParam * self.eilteSize
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
        self.facilityToCustomerUnitCost = np.empty((self.totalPotentialSites, self.totalCustomers))
        self.facilityToCustomerCost = np.empty((self.totalPotentialSites, self.totalCustomers))
        
        for i in range(self.totalPotentialSites):
            self.potentialSitesFixedCosts[i] = np.float64(f.readline().split()[1])
        
        for j in range(self.totalCustomers):
            self.demand = np.float64(f.readline())
            lineItems = f.readline().split()
            for i in range(self.totalPotentialSites):
                self.facilityToCustomerUnitCost[i,j] = np.float64(lineItems[i%orlibCostValuePerLine])/self.demand
                self.facilityToCustomerCost[i,j] = np.float64(lineItems[i%orlibCostValuePerLine])
                if i%orlibCostValuePerLine == orlibCostValuePerLine - 1:
                    lineItems = f.readline().split()
                    
        # Mutation Paramters
        self.mutationDistributionMean = (self.populationSize - self.eilteSize) * self.totalPotentialSites * self.mutationRate
        self.mutationDistributionVariance = (self.populationSize - self.eilteSize) * self.totalPotentialSites * self.mutationRate * (1 - self.mutationRate)
                    
        # Population Random Initialization
        self.population = np.empty((self.populationSize, self.totalPotentialSites), np.bool)
        for i in range(self.populationSize):
            for j in range(self.totalPotentialSites):
                if np.random.uniform() > 0.5:
                    self.population[i,j] = True
                else:
                    self.population[i,j] = False
        
        # GA Main Loop
        self.score = np.empty((self.populationSize, ))
        self.rank = np.ones((self.populationSize, ))
        self.fromPrevGeneration = np.zeros((self.populationSize, ), dtype=np.bool)
        self.bestIndividual = None
        self.bestIndividualRepeatedTime = 0
        self.bestPlanSoFar = []
        self.nRepeat = ceil(self.nRepeatParams[0] * (self.totalCustomers * self.totalPotentialSites) ** self.nRepeatParams[1])
        self.generation = 1
        self.compareToOptimal = None
        self.errorPercentage = None
        
        # PreScore Calculations
        for individualIndex in range(self.populationSize):
            self.score[individualIndex] = self.calculateScore(individualIndex)
                    
    def calculateScore(self, individualIndex):
        individual = self.population[individualIndex, :]
        cacheKey = individual.tobytes()
        if cacheKey in self.cache:
            return self.cache.peek(cacheKey)
        openFacilites = np.where(individual == True)[0]
        score = 0
        if openFacilites.shape[0] == 0: return np.finfo(np.float64).max
        for customerIndex in range(self.totalCustomers):
            openFacilityCosts = self.facilityToCustomerCost[openFacilites, customerIndex]
            score += np.min(openFacilityCosts)
        score += self.potentialSitesFixedCosts.dot(individual)
        self.cache[cacheKey] = score
        return score
    
    def sortAll(self):
        sortArgs = self.score.argsort()
        self.population = self.population[sortArgs]
        self.score = self.score[sortArgs]
        self.fromPrevGeneration = self.fromPrevGeneration[sortArgs]
        
    def uniformCrossoverOffspring(self, indexA, indexB):
        crossoverMask = np.random.choice(a=[True, False], size=(self.totalPotentialSites,), p=[self.crossoverMaskRate, 1-self.crossoverMaskRate])
        crossoverMaskComplement = np.invert(crossoverMask)
        parentA = self.population[indexA,:]
        parentB = self.population[indexB,:]
        # offspringA = crossoverMask * parentA + crossoverMaskComplement * parentB
        # offspringB = crossoverMask * parentB + crossoverMaskComplement * parentA
        # return (offspringA, offspringB)
        return crossoverMask * parentA + crossoverMaskComplement * parentB
    
    def mutateOffsprings(self):
        totalMutations = np.random.normal(loc=self.mutationDistributionMean, scale=self.mutationDistributionVariance)
        totalMutations = np.round(totalMutations)
        while totalMutations >= 1:
            individual = np.random.randint(self.eilteSize, self.populationSize)
            individualIndex = np.random.randint(0, self.totalPotentialSites)
            if self.population[individual, individualIndex] == True:
                self.population[individual, individualIndex] = False
            else:
                self.population[individual, individualIndex] = True
            totalMutations -= 1        
    
    def rouletteWheelParentSelection(self):
        rankSum = np.sum(self.rank)
        rand = np.random.uniform(low=0, high=rankSum)
        partialSum = 0
        for individualIndex in range(self.populationSize):
            partialSum += self.rank[individualIndex]
            if partialSum > rand:
                return individualIndex
    
    def replaceWeaks(self):
        # Selection
        parentIndexes = [self.rouletteWheelParentSelection()]
        while len(parentIndexes) < self.totalOffsprings:
            parentIndex = self.rouletteWheelParentSelection()
            if parentIndex != parentIndexes[-1]:
                parentIndexes += [parentIndex]
        while parentIndexes[-1] == parentIndexes[0] or parentIndexes[-1] == parentIndexes[-2]:
            parentIndexes[-1] = self.rouletteWheelParentSelection()
        # Crossover
        i = 0
        for individual in range(self.eilteSize, self.populationSize):
            parentIndexA = parentIndexes[i % self.totalOffsprings]
            parentIndexB = parentIndexes[(i+1) % self.totalOffsprings]
            offspring = self.uniformCrossoverOffspring(parentIndexA, parentIndexB)
            self.population[individual, :] = np.transpose(offspring)
            i += 1
        # Mutation
        self.mutateOffsprings()
        # Update Scores
        for individual in range(self.eilteSize, self.populationSize):
            self.score[individual] = self.calculateScore(individual)
    
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
        lastRank = self.populationSize
        self.rank[0] = lastRank
        for individualIndex in range(1,self.populationSize):
            if self.identicalIndividuals(individualIndex, individualIndex - 1):
                self.rank[individualIndex] = 0
            else:
                lastRank -= 1
                self.rank[individualIndex] = lastRank
        self.rank -= (lastRank - 1)
        
    
    def markElites(self):
        ones = np.ones((self.eilteSize, ), dtype=np.bool)
        zeros = np.ones((self.populationSize - self.eilteSize, ), dtype=np.bool)
        self.fromPrevGeneration = np.concatenate((ones, zeros))
        
    
    def compareBestFoundPlanToOptimalPlan(self):
        compare = []
        for i in range(len(self.optimals)):
            if self.optimals[i] == self.bestPlanSoFar[i]: 
                compare += [True]
            else: 
                compare += [False]
        return np.array(compare)
    
    def run(self):
        
        # Start Timing
        startTimeit = default_timer()
        
        while True:
            if self.printSummary:
                print('\rgeneration number %d               ' % self.generation, end='')
            self.sortAll()
            if self.score[0] != self.bestIndividual:
                self.bestIndividualRepeatedTime = 0
                self.bestPlanSoFar = self.bestIndividualPlan(0)
                self.bestIndividual = self.score[0]
            self.bestIndividualRepeatedTime += 1
            if self.bestIndividualRepeatedTime > self.nRepeat or self.generation >= self.maxGenerations: break
            self.updateRank()
            self.punishElites()
            self.replaceWeaks()
            self.markElites()
            self.generation += 1
        
        # End Timing
        endTimeit = default_timer()
        
        self.compareToOptimal = self.compareBestFoundPlanToOptimalPlan()
        if False in self.compareToOptimal:    
            self.errorPercentage = self.bestIndividual - self.optimalCost * 100 / self.optimalCost
        else:
            self.errorPercentage = 0
        
        if self.printSummary:
            print('\rdataset name:',self.orlibDataset)
            print('total generations of', self.generation)
            print('best individual score', self.bestIndividual,\
                  'repeated for last', self.bestIndividualRepeatedTime,'times')
            if False not in self.compareToOptimal:
                print('REACHED OPTIMAL OF', self.optimalCost)
            else:
                print('DID NOT REACHED OPTIMAL OF', self.optimalCost, "|",\
                      self.errorPercentage,"% ERROR")
            print('total elapsed time:', endTimeit - startTimeit)
            assignedFacilitiesString = ''
            for f in self.bestPlanSoFar:
                assignedFacilitiesString += str(f) + ' '
            print('assigned facilities:')
            print(assignedFacilitiesString)