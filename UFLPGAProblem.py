import numpy as np
from math import ceil
from pylru import lrucache
from random import sample

class UFLPGAProblem:
    MAX_FLOAT = np.finfo(np.float64).max
    def __init__(
        self, 
        potentialSitesFixedCosts,
        facilityToCustomerCost,
        mutationRate = 0.005,
        crossoverRate = 0.75,
        crossoverMaskRate = 0.4,
        populationSize = 100,
        tournamentSize = 3,
        cacheParam = 10,
        maxGenerations = None,
        maxFacilities = None,
    ):
        # GA Parameters
        self.populationSize = populationSize
        self.totalCrossoverOffspring = ceil(crossoverRate*populationSize)
        self.crossoverMaskRate = crossoverMaskRate
        self.maxGenerations = maxGenerations
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
        
        # Tournament Selection
        self.intermediatePopulation = np.empty_like(self.population)
        self.bestIndividual = np.empty_like(self.population[0])
        self.bestIndividualScore = UFLPGAProblem.MAX_FLOAT

        # GA Main Loop
        self.generation = 1
                    
    def calculateScore(self, individualIndex=None, individual=None, save=True):
        if individualIndex != None:
            individual = self.population[individualIndex, :]
        cacheKey = individual.tobytes()
        if cacheKey in self.cache:
            return self.cache.peek(cacheKey)
        openFacilites = np.where(individual == True)[0]
        if openFacilites.shape[0] == 0: 
            return UFLPGAProblem.MAX_FLOAT
        score = np.sum(np.amin(self.facilityToCustomerCost[openFacilites, :], axis=0))
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
        if self.generation > self.maxGenerations:
            return True
        return False

    def run(self):
        while not self.finish():
            self.selection()
            self.reproduction()
            self.mutateOffspring()
            self.generation += 1
    
    def selection(self):
        for i in range(self.populationSize):
            tournamentIndices = sample(range(self.populationSize), self.tournamentSize)
            scores = [self.calculateScore(individualIndex=i) for i in tournamentIndices]
            self.intermediatePopulation[i] = self.population[tournamentIndices[np.argmin(scores)]]

    def reproduction(self):
        i = 0
        while i < self.totalCrossoverOffspring - 1:
            self.population[i], self.population[i+1] = self.uniformCrossoverOffspring(
                parentA=self.intermediatePopulation[i],
                parentB=self.intermediatePopulation[i+1]
            )
            i += 2
        self.population[i:] = self.intermediatePopulation[i:]

    def bestIndividualPlan(self, individual):
        openFacilites = np.where(individual == True)[0]
        plan = []
        for customerIndex in range(self.totalCustomers):
            openFacilityCosts = self.facilityToCustomerCost[openFacilites, customerIndex]
            chosenFacilityIndex = np.where(openFacilityCosts == np.min(openFacilityCosts))[0][0]
            plan += [openFacilites[chosenFacilityIndex]]
        return plan

    @property
    def bestPlan(self):
        return self.bestIndividualPlan(self.bestIndividual)
