import numpy as np
from math import ceil
from timeit import default_timer
from pylru import lrucache
from sys import stdout
from random import sample, randint

T_SELECTION = 'tournament'
R_SELECTION = 'rank-based-roulette-wheel'
MAX_FLOAT = np.finfo(np.float64).max

class UFLPGAProblem:
    def __init__(
        self, 
        potentialSitesFixedCosts,
        facilityToCustomerCost,
        mutationRate = 0.005,
        crossoverRate = 0.75,
        crossoverMaskRate = 0.4,
        populationSize = 150,
        tournamentSize = 3,
        eliteFraction = 1/3,
        maxRank = 2.5,
        minRank = 0.712,
        cacheParam = 10,
        maxGenerations = None,
        nRepeat = None,
        maxFacilities = None,
        printProgress = False,
        selectionMethod = T_SELECTION, # OR R_SELECTION
        problemTitle = 'noTitle'
    ):
        if maxGenerations == None and nRepeat == None:
            raise Exception("at least one of the termination paramters (maxGenerations/nRepeat) must be defined") 

        self.printProgress = printProgress
        self.problemTitle = problemTitle
        self.selectionMethod = selectionMethod

        # GA Parameters
        self.populationSize = populationSize
        self.crossoverRate = crossoverRate
        self.crossoverMaskRate = crossoverMaskRate
        self.maxGenerations = maxGenerations
        self.nRepeat = nRepeat
        self.mutationRate = mutationRate
        self.maxFacilities = maxFacilities
        self.tournamentSize = tournamentSize

        # Cache
        self.cache = lrucache(cacheParam*self.populationSize)
        
        # Input Data
        self.potentialSitesFixedCosts = potentialSitesFixedCosts
        self.facilityToCustomerCost = facilityToCustomerCost
        self.totalPotentialSites = self.facilityToCustomerCost.shape[0]
        self.totalCustomers = self.facilityToCustomerCost.shape[1]

        # Population Random Initialization
        if maxFacilities == None:
            self.population = np.random.choice(a=[True, False], size=(self.populationSize, self.totalPotentialSites), p=[0.5, 0.5])
        else:
            self.population = np.zeros((self.populationSize, self.totalPotentialSites), np.bool)
            for i in range(self.populationSize):
                self.population[i, sample(range(self.totalPotentialSites), maxFacilities)] = True
        
        # Tournament/Rank-BasedRW Selection
        self.intermediatePopulation = np.empty_like(self.population)
        self.bestIndividual = np.empty_like(self.population[0])
        self.bestIndividualScore = MAX_FLOAT

        # Rank-BasedRW Selection
        if self.selectionMethod == R_SELECTION:
            self.fitness = np.empty((self.populationSize, ), np.float64)

        # GA Main Loop
        self.generation = 1
        self.bestIndividualRepeatedTime = 1
        self.startTimeit = None
        self.bestFoundElapsedTime = None

        # Timing
        self.mainLoopElapsedTime = None
        self.selectionTime = 0
        self.recombinationTime = 0
        self.mutationTime = 0
                    
    def calculateScore(self, individualIndex=None, individual=None, save=True):
        if individualIndex != None:
            individual = self.population[individualIndex, :]
        cacheKey = individual.tobytes()
        if cacheKey in self.cache:
            return self.cache.peek(cacheKey)
        openFacilities = np.where(individual == True)[0]
        if openFacilities.shape[0] == 0: 
            return MAX_FLOAT
        score = np.sum(np.amin(self.facilityToCustomerCost[openFacilities, :], axis=0))
        score += self.potentialSitesFixedCosts.dot(individual)
        if save: self.cache[cacheKey] = score
        return score

    def uniformCrossoverOffspring(self, indexA=None, indexB=None, parentA=None, parentB=None):
        crossoverMask = np.random.choice(a=[True, False], size=(self.totalPotentialSites,), p=[self.crossoverMaskRate, 1-self.crossoverMaskRate])
        crossoverMaskComplement = np.invert(crossoverMask)
        if indexA != None:
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

    def mutateOffspring(self):      
        mutationRate = self.mutationRate
        mask =  np.random.choice(a=[True, False], size=(self.populationSize, self.totalPotentialSites), p=[mutationRate, 1-mutationRate])
        self.population = self.population != mask

    def finish(self):
        scores = [self.calculateScore(individualIndex=i) for i in range(self.populationSize)]
        argBestScore = np.argmin(scores)
        bestScore = scores[argBestScore]
        if bestScore < self.bestIndividualScore:
            self.bestIndividualScore = bestScore
            self.bestIndividual[:] = self.population[argBestScore, :]
            self.bestIndividualRepeatedTime = 1
            self.bestFoundElapsedTime = default_timer() - self.startTimeit
        else:
            self.bestIndividualRepeatedTime += 1
        if self.printProgress:
                print('\r' + self.problemTitle, 'generation number %d' % self.generation, end='', file=stdout)
        
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

        if self.selectionMethod == T_SELECTION:
            self.selection = self.tournamentSelection
        elif self.selectionMethod == R_SELECTION:
            self.selection = self.rankBasedRWSelection

        # Start Timing
        self.startTimeit = default_timer()

        while not self.finish():
            # Selection
            tempTime = default_timer()
            self.selection()
            self.selectionTime += default_timer() - tempTime
            
            # Recombination
            tempTime = default_timer()
            self.recombination()
            self.recombinationTime += default_timer() - tempTime
            
            # Mutation
            if self.maxFacilities == None:
                tempTime = default_timer()
                self.mutateOffspring()
                self.mutationTime += default_timer() - tempTime

            self.generation += 1

        # End Timing
        endTimeit = default_timer()
        self.mainLoopElapsedTime = endTimeit - self.startTimeit

    def tournamentSelection(self):
        for i in range(self.populationSize):
            tournamentIndices = sample(range(self.populationSize), self.tournamentSize)
            scores = [self.calculateScore(individualIndex=i) for i in tournamentIndices]
            self.intermediatePopulation[i] = self.population[tournamentIndices[np.argmin(scores)]]

    def rankBasedRWSelection(self):
        sortArgs = self.score.argsort()
        self.population = self.population[sortArgs]

    def recombination(self):
        i = 0
        while i < self.intermediatePopulation.shape[0] - 1:
            if np.random.uniform() < self.crossoverRate:
                self.population[i], self.population[i+1] = self.crossover(
                    parentA=self.intermediatePopulation[i],
                    parentB=self.intermediatePopulation[i+1]
                )
            else:
                self.population[i:i+2] = self.intermediatePopulation[i:i+2]
            i += 2
        if i == self.intermediatePopulation.shape[0] - 1:
            self.population[i:] = self.intermediatePopulation[i:]
        

    def bestIndividualPlan(self, individual):
        openFacilities = np.where(individual == True)[0]
        plan = []
        for customerIndex in range(self.totalCustomers):
            openFacilityCosts = self.facilityToCustomerCost[openFacilities, customerIndex]
            chosenFacilityIndex = np.where(openFacilityCosts == np.min(openFacilityCosts))[0][0]
            plan += [openFacilities[chosenFacilityIndex]]
        return plan

    @property
    def bestPlan(self):
        return self.bestIndividualPlan(self.bestIndividual)

    @property
    def score(self):
        return np.array([self.calculateScore(i) for i in range(self.populationSize)])