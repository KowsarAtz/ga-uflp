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
        geneMutationRate = 0.01,
        chromosomeMutationRate = 0.4,
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
        self.totalOffspring = self.populationSize - self.eliteSize
        self.maxGenerations = maxGenerations
        self.geneMutationRate = geneMutationRate
        self.chromosomeMutationRate = chromosomeMutationRate
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
        self.offspring = np.empty((self.totalOffspring, self.totalPotentialSites))
        
        # GA Main Loop
        self.score = np.empty((self.populationSize, ))
        self.offspringScore = np.empty((self.totalOffspring, ))
        self.fitness = np.ones((self.populationSize, ))
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
        openFacilities = np.where(individual == True)[0]
        if openFacilities.shape[0] == 0: 
            return UFLPGeneticProblem.MAX_FLOAT
        score = np.sum(np.amin(self.facilityToCustomerCost[openFacilities, :], axis=0))
        score += self.potentialSitesFixedCosts.dot(individual)
        if cached: self.cache[cacheKey] = score
        return score
    
    def sortAll(self):
        sortArgs = self.score.argsort()
        self.population = self.population[sortArgs]
        self.score = self.score[sortArgs]
        self.fromPrevGeneration = self.fromPrevGeneration[sortArgs]

    def calculateOffspringScore(self):
        for individual in range(self.totalOffspring):
            self.offspringScore[individual] = self.calculateScore(individual=self.offspring[individual])
    
    def sortOffspring(self):
        sortArgs = self.offspringScore.argsort()
        self.offspring = self.offspring[sortArgs]
        self.offspringScore = self.offspringScore[sortArgs]

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
    
    def defaultMutateOffspring(self):      
        r = self.geneMutationRate
        mask =  np.random.choice(a=[True, False], size=(self.totalOffspring, self.totalPotentialSites), p=[r, 1-r])
        self.offspring = self.offspring != mask
        
    def balancedMutateOffspring(self):
        for i in range(self.totalOffspring):
            if np.random.uniform() > self.chromosomeMutationRate:
                continue
            
            a = sum(self.offspring[i])

            falseIndices = np.where(self.offspring[i]==False)[0]
            chosenFalseIndex = sample(list(falseIndices), 1)
            self.offspring[i,chosenFalseIndex] = True
            
            trueIndices = np.where(self.offspring[i]==True)[0]
            chosenTrueIndex = sample(list(trueIndices), 1)
            self.offspring[i,chosenTrueIndex] = False
            
            b = sum(self.offspring[i])
            if b != a:
                print('hereeee3', sum(self.offspring[i]))


    def rouletteWheelParentSelection(self):
        fitnessSum = np.sum(self.fitness)
        rand = np.random.uniform(low=0, high=fitnessSum)
        partialSum = 0
        for individualIndex in range(self.populationSize):
            partialSum += self.fitness[individualIndex]
            if partialSum > rand:
                return individualIndex
    
    def produceOffspring(self):
        # Selection and Crossover
        individual = 0
        while individual < self.totalOffspring:
            parentIndexA = self.rouletteWheelParentSelection()
            parentIndexB = self.rouletteWheelParentSelection()
            while parentIndexA == parentIndexB : parentIndexB = self.rouletteWheelParentSelection() 
            offspringA, offspringB = self.crossover(parentIndexA, parentIndexB)
            self.offspring[individual, :] = offspringA
            self.offspring[(individual + 1) % self.totalOffspring, :] = offspringB
            individual += 2
        
        # Mutation
        self.mutateOffspring()
        self.calculateOffspringScore()
        self.sortOffspring()

    def updatePopulation(self):
        dupIndices = np.where(self.duplicateIndices == True)
        dupIndicesCount = len(dupIndices[0])
        self.population[dupIndices, :] = self.offspring[:dupIndicesCount, :]
        self.score[dupIndices] = self.offspringScore[:dupIndicesCount]
        self.fromPrevGeneration[dupIndices] = False

        offspringIndex = dupIndicesCount
        populationIndex = self.populationSize - 1
        while offspringIndex < self.totalOffspring:
            if self.duplicateIndices[populationIndex]:
                populationIndex -= 1
                continue
            currentScore = self.score[populationIndex]
            newScore = self.offspringScore[offspringIndex]
            if newScore > currentScore:
                break
            self.population[populationIndex, :] = self.offspring[offspringIndex, :]
            self.score[populationIndex] = newScore
            self.fromPrevGeneration[populationIndex] = False
            populationIndex -= 1
            offspringIndex += 1
    
    def bestIndividualPlan(self, individualIndex=0):
        openFacilities = np.where(self.population[individualIndex, :] == True)[0]
        plan = []
        for customerIndex in range(self.totalCustomers):
            openFacilityCosts = self.facilityToCustomerCost[openFacilities, customerIndex]
            chosenFacilityIndex = np.where(openFacilityCosts == np.min(openFacilityCosts))[0][0]
            plan += [openFacilities[chosenFacilityIndex]]
        return plan
                
    def punishElites(self):
        averageFitness = np.average(self.fitness)
        for individualIndex in range(self.populationSize):
            if self.fromPrevGeneration[individualIndex]:
                if self.fitness[individualIndex] > averageFitness:
                    self.fitness[individualIndex] -= averageFitness
                else:
                    self.fitness[individualIndex] = 0
        
    def identicalIndividuals(self, indexA, indexB):
        return False not in (self.population[indexA, :] == self.population[indexB, :])
        
    def updateFitness(self):
        self.duplicateIndices = np.zeros((self.populationSize, ), np.bool)
        currentRank = self.maxRank
        self.fitness[0] = currentRank
        for individualIndex in range(1,self.populationSize):
            currentRank -= self.rankStep
            if self.identicalIndividuals(individualIndex, individualIndex - 1):
                self.fitness[individualIndex] = 0
                self.duplicateIndices[individualIndex] = True
            else:
                self.fitness[individualIndex] = currentRank    
    
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
            self.mutateOffspring = self.defaultMutateOffspring
        else:
            self.crossover = self.balancedCrossoverOffspring
            self.mutateOffspring = self.balancedMutateOffspring

        # Start Timing
        startTimeit = default_timer()

        self.sortAll()
        while not self.finish():
            if self.printProgress:
                print('\r' + self.problemTitle, 'generation number %d' % self.generation, end='', file=stdout)
            self.updateFitness()
            self.punishElites()
            self.markElites()
            self.produceOffspring()
            self.updatePopulation()
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