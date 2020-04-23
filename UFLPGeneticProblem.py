import numpy as np
from math import ceil
from timeit import default_timer

class UFLPGeneticProblem:
    def __init__(self, orlibPath, orlibDataset, orlibCostValuePerLine = 7, populationSize = 150, eliteFraction = 2/3, maxGenerations = 4000, mutationRate = 0.05, crossoverRate = 0.3, nRepeatParams = (10,0.5)):
        self.orlibDataset = orlibDataset
        # GA Parameters
        self.populationSize = populationSize
        self.eilteSize = ceil(eliteFraction * self.populationSize)
        self.totalOffsprings = self.populationSize - self.eilteSize
        self.maxGenerations = maxGenerations
        self.mutationRate = mutationRate
        self.crossoverRate = crossoverRate
        self.nRepeatParams = nRepeatParams
        
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
        self.rank = np.empty((self.populationSize, ))
        self.bestIndividual = None
        self.bestIndividualRepeatedTime = 0
        self.bestPlanSoFar = []
        self.nRepeat = ceil(self.nRepeatParams[0] * (self.totalCustomers * self.totalPotentialSites) ** self.nRepeatParams[1])
        self.generation = 1
        
        # PreScore Calculations
        for individualIndex in range(self.populationSize):
            self.score[individualIndex] = self.calculateScore(individualIndex)
                    
    def calculateScore(self, individualIndex):
        openFacilites = np.where(self.population[individualIndex, :] == True)[0]
        score = 0
        if openFacilites.shape[0] == 0: return np.finfo(np.float64).max
        for customerIndex in range(self.totalCustomers):
            openFacilityCosts = self.facilityToCustomerCost[openFacilites, customerIndex]
            score += np.min(openFacilityCosts)
        for openFacilityIndex in openFacilites:
            score += self.potentialSitesFixedCosts[openFacilityIndex]
        return score
    
    def sortAll(self):
        sortArgs = self.score.argsort()
        self.population = self.population[sortArgs]
        self.score = self.score[sortArgs]
        
    def uniformCrossoverOffspring(self, indexA, indexB):
        offspring = np.empty((self.totalPotentialSites, ))
        for i in range(self.totalPotentialSites):
            if np.random.uniform() < self.crossoverRate: 
                offspring[i] = self.population[indexA,i]
            else: 
                offspring[i] = self.population[indexB,i]
        return offspring
    
    def mutateOffspring(self, offspring):
        for i in range(self.totalPotentialSites):
            if np.random.uniform() < self.mutationRate:
                if offspring[i] == True:
                    offspring[i] = False
                else:
                    offspring[i] = True
    
    def rouletteWheelParentSelection(self):
        rankSum = np.sum(self.rank)
        rand = np.random.uniform(low=0, high=rankSum)
        randOpposite = (rand + rankSum//2) % rankSum
        partialSum = 0
        parentA, parentB = None, None
        for individualIndex in range(self.populationSize):
            partialSum += self.rank[individualIndex]
            if partialSum > rand:
                parentA = individualIndex
                break
        partialSum = 0
        for individualIndex in range(self.populationSize):
            partialSum += self.rank[individualIndex]
            if partialSum > randOpposite:
                parentB = individualIndex
                return (parentA, parentB)
    
    def replaceWeaks(self):
        for i in range(self.totalOffsprings):
            parentAIndex, parentBIndex = self.rouletteWheelParentSelection()
            offspring = self.uniformCrossoverOffspring(parentAIndex, parentBIndex)
            self.mutateOffspring(offspring)
            self.population[self.totalOffsprings+i, :] = np.transpose(offspring)
            self.score[self.totalOffsprings+i] = self.calculateScore(self.totalOffsprings+i)
    
    def bestIndividualPlan(self, individualIndex):
        openFacilites = np.where(self.population[individualIndex, :] == True)[0]
        plan = []
        for customerIndex in range(self.totalCustomers):
            openFacilityCosts = self.facilityToCustomerCost[openFacilites, customerIndex]
            chosenFacilityIndex = np.where(openFacilityCosts == np.min(openFacilityCosts))[0][0]
            plan += [openFacilites[chosenFacilityIndex]]
        return plan
    
    def punishDuplicates(self):
        _, index = np.unique(self.population, return_index=True, axis=0)
        for individualIndex in range(self.populationSize):
            if individualIndex not in index:
                self.rank[individualIndex] = 0
                
    def punishElites(self):
        averageRank = np.average(self.rank)
        for individualIndex in range(self.eilteSize):
            if self.rank[individualIndex] > averageRank:
                self.rank[individualIndex] -= averageRank
            else:
                self.rank[individualIndex] = 0
                
    def updateRank(self):
        self.rank[0] = self.populationSize
        for individualIndex in range(1,self.populationSize):
            if self.score[individualIndex] == self.score[individualIndex-1]:
                self.rank[individualIndex] = self.rank[individualIndex-1]
            else:
                self.rank[individualIndex] = self.rank[individualIndex-1]-1
        self.rank -= (np.min(self.rank) - 1)
    
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
            print('\rgeneration number %d               ' % self.generation, end='')
            self.sortAll()
            if self.score[0] != self.bestIndividual:
                self.bestIndividualRepeatedTime = 0
                self.bestPlanSoFar = self.bestIndividualPlan(0)
                self.bestIndividual = self.score[0]
            self.bestIndividualRepeatedTime += 1
            if self.bestIndividualRepeatedTime > self.nRepeat or self.generation >= self.maxGenerations: break
            self.updateRank()
            self.punishDuplicates()
            self.punishElites()
            self.replaceWeaks()
            self.generation += 1
        
        # End Timing
        endTimeit = default_timer()
        
        compareToOptimal = self.compareBestFoundPlanToOptimalPlan()
        
        print('\rdataset name:',self.orlibDataset)
        print('total generations of', self.generation)
        print('best individual score', self.bestIndividual,\
              'repeated for last', self.bestIndividualRepeatedTime,'times')
        if False not in compareToOptimal:
            print('REACHED OPTIMAL OF', self.optimalCost)
        else:
            print('DID NOT REACHED OPTIMAL OF', self.optimalCost, "|",\
                  (self.bestIndividual - self.optimalCost) * 100 / self.optimalCost,"% ERROR")
        print('total elapsed time:', endTimeit - startTimeit)
        assignedFacilitiesString = ''
        for f in self.bestPlanSoFar:
            assignedFacilitiesString += str(f) + ' '
        print('assigned facilities:')
        print(assignedFacilitiesString)