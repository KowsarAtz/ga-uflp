import numpy as np
from math import ceil
from random import sample, randint
from timeit import default_timer
from pylru import lrucache
from sys import stdout

class UFLPGeneticProblem:
    MAX_FLOAT = np.finfo(np.float64).max
    def __init__(
        self, 
        potentialSitesFixedCosts,
        facilityToCustomerCost,
        mutationRate = 0.01,
        crossoverMaskRate = 0.4,
        eliteFraction = 1/3,
        populationSize = 150,
        cacheParam = 50,
        maxRank = 2.5,
        minRank = 0.712,
        maxGenerations = None,
        nRepeat = None,
        maxFacilities = None,
        printProgress = False,
        problemTitle = 'noTitle'
    ):

        if maxGenerations == None and nRepeat == None:
            raise Exception("at least one of the termination paramters (maxGenerations/nRepeat) must be defined") 

        self.printProgress = printProgress
        self.problemTitle = problemTitle

        # GA Parameters
        self.populationSize = populationSize
        self.eliteSize = ceil(eliteFraction * self.populationSize)
        self.totalOffsprings = self.populationSize - self.eliteSize
        self.maxGenerations = maxGenerations
        self.mutationRate = mutationRate
        self.maxFacilities = maxFacilities
        self.crossoverMaskRate = crossoverMaskRate
        
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
        if maxFacilities == None:
            self.population = np.random.choice(a=[True, False], size=(self.populationSize, self.totalPotentialSites), p=[0.5, 0.5])
        else:
            self.population = np.zeros((self.populationSize, self.totalPotentialSites), np.bool)
            for i in range(self.populationSize):
                self.population[i, sample(range(self.totalPotentialSites), maxFacilities)] = True
        self.offsprings = np.empty((self.totalOffsprings, self.totalPotentialSites))
        
        # GA Main Loop
        self.score = np.empty((self.populationSize, ))
        self.offspringsScore = np.empty((self.totalOffsprings, ))
        self.rank = np.ones((self.populationSize, ))
        self.fromPrevGeneration = np.zeros((self.populationSize, ), dtype=np.bool)
        self.bestIndividualScore = UFLPGeneticProblem.MAX_FLOAT
        self.bestIndividualRepeatedTime = 0
        self.duplicateIndices = np.zeros((self.populationSize, ), np.bool)
        self.nRepeat = nRepeat
        self.generation = 1
        self.mainLoopElapsedTime = None
        self.bestFoundElapsedTime = 0
        
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

    def balancedCrossoverOffspring(self, indexA=None, indexB=None, parentA=None, parentB=None):
        if indexA != None:
            parentA = self.population[indexA,:]
            parentB = self.population[indexB,:]
        
        offspringA = parentA * parentB
        offspringB = offspringA.copy()
        
        diff = self.maxFacilities - np.sum(offspringA)
        if diff <= 0: return offspringA, offspringB
        
        otherCandidates = parentA != parentB
        otherCandidateIndices = list(np.where(otherCandidates == True)[0])
        if len(otherCandidateIndices) == 0: return offspringA, offspringB
        
        newFacilitiesCount = randint(0, diff)
        chosenIndices = sample(otherCandidateIndices, min(newFacilitiesCount, len(otherCandidateIndices)))
        offspringA[chosenIndices] = True
        
        newFacilitiesCount = randint(0, diff)   
        chosenIndices = sample(otherCandidateIndices, min(newFacilitiesCount, len(otherCandidateIndices)))
        offspringB[chosenIndices] = True     
        
        return offspringA, offspringB
    
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
            offspringA, offspringB = self.crossover(parentIndexA, parentIndexB)
            self.offsprings[individual, :] = offspringA
            self.offsprings[(individual + 1) % self.totalOffsprings, :] = offspringB
            individual += 2
        
        # Mutation
        self.mutateOffsprings()
        self.calculateOffspringsScore()
        self.sortOffsprings()

        # Replacement
        dupIndices = np.where(self.duplicateIndices == True)
        dupIndicesCount = len(dupIndices[0])
        self.population[dupIndices, :] = self.offsprings[:dupIndicesCount, :]
        self.score[dupIndices] = self.offspringsScore[:dupIndicesCount]
        self.fromPrevGeneration[dupIndices] = False

        offspringsIndex = dupIndicesCount
        populationIndex = self.populationSize - 1
        while offspringsIndex < self.totalOffsprings:
            if self.duplicateIndices[populationIndex]:
                populationIndex -= 1
                continue
            currentScore = self.score[populationIndex]
            newScore = self.offspringsScore[offspringsIndex]
            if newScore > currentScore:
                break
            self.population[populationIndex, :] = self.offsprings[offspringsIndex, :]
            self.score[populationIndex] = newScore
            self.fromPrevGeneration[populationIndex] = False
            populationIndex -= 1
            offspringsIndex += 1
    
    def bestIndividualPlan(self, individualIndex=0):
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
        self.duplicateIndices = np.zeros((self.populationSize, ), np.bool)
        currentRank = self.maxRank
        self.rank[0] = currentRank
        for individualIndex in range(1,self.populationSize):
            currentRank -= self.rankStep
            if self.identicalIndividuals(individualIndex, individualIndex - 1):
                self.rank[individualIndex] = 0
                self.duplicateIndices[individualIndex] = True
            else:
                self.rank[individualIndex] = currentRank    
    
    def markElites(self):
        self.fromPrevGeneration = np.ones((self.populationSize, ), dtype=np.bool)
    
    def finish(self):
        if self.maxGenerations != None and self.generation >= self.maxGenerations:
            return True
        if self.nRepeat != None and self.bestIndividualRepeatedTime >= self.nRepeat:
            return True
        return False

    def run(self):
        if self.maxFacilities == None:
            self.crossover = self.uniformCrossoverOffspring
        else:
            self.crossover = self.balancedCrossoverOffspring
        
        # Start Timing
        startTimeit = default_timer()

        self.sortAll()
        while not self.finish():
            if self.printProgress:
                print('\r' + self.problemTitle, 'generation number %d' % self.generation, end='', file=stdout)
            self.updateRank()
            self.punishElites()
            self.markElites()
            self.replaceWeaks()
            self.sortAll()
            if self.score[0] != self.bestIndividualScore:
                self.bestFoundElapsedTime = default_timer() - startTimeit
                self.bestIndividualRepeatedTime = 0
                self.bestIndividualScore = self.score[0]
            self.bestIndividualRepeatedTime += 1
            self.generation += 1
        self.bestPlan = self.bestIndividualPlan(0)

        if self.printProgress:
            print('\r' + self.problemTitle, 'generation number %d' % self.generation, end='', file=stdout)

        # End Timing
        endTimeit = default_timer()
        self.mainLoopElapsedTime = endTimeit - startTimeit