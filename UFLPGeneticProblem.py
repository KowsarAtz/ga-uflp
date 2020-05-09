import numpy as np
from math import ceil
from timeit import default_timer
from pylru import lrucache
from sys import stdout

class UFLPGeneticProblem:
    MAX_FLOAT = np.finfo(np.float64).max
    def __init__(
        self, 
        potentialSitesFixedCosts,
        facilityToCustomerCost,
        # maxFacilities = None,
        mutationRate = 0.01,
        crossoverMaskRate = 0.4,
        eliteFraction = 1/3,
        populationSize = 150,
        cacheParam = 50,
        maxRank = 2.5,
        minRank = 0.712,
        maxGenerations = 4000,
        nRepeatParams = None
    ):

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
        
        # Input Data
        self.potentialSitesFixedCosts = potentialSitesFixedCosts
        self.facilityToCustomerCost = facilityToCustomerCost
        self.totalPotentialSites = self.facilityToCustomerCost.shape[0]
        self.totalCustomers = self.facilityToCustomerCost.shape[1]

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
        self.duplicateIndices = []
        self.nRepeat = None
        if nRepeatParams != None:
            self.nRepeat = ceil(self.nRepeatParams[0] * (self.totalCustomers * self.totalPotentialSites) ** self.nRepeatParams[1])
        self.generation = 1
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
    
    def finsih(self):
        if self.nRepeat == None:
            return self.generation >= self.maxGenerations
        return self.bestIndividualRepeatedTime > self.nRepeat or\
             self.generation >= self.maxGenerations

    def run(self):
        
        # Start Timing
        startTimeit = default_timer()

        self.sortAll()
        while not self.finish():
            self.updateRank()
            self.punishElites()
            self.markElites()
            self.replaceWeaks()
            self.sortAll()
            if self.score[0] != self.bestIndividual:
                self.bestIndividualRepeatedTime = 0
                self.bestIndividual = self.score[0]
            self.bestIndividualRepeatedTime += 1
            self.generation += 1
        self.bestPlan = self.bestIndividualPlan(0)

        # End Timing
        endTimeit = default_timer()
        self.mainLoopElapsedTime = endTimeit - startTimeit