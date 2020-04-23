import numpy as np
from random import shuffle

def calculateScore(population, individualIndex, facilityToCustomerCost, potentialSitesFixedCosts):
    openFacilites = np.where(population[individualIndex, :] == True)[0]
    score = 0
    for customerIndex in range(facilityToCustomerCost.shape[1]):
        openFacilityCosts = facilityToCustomerCost[openFacilites, customerIndex]
        score += np.min(openFacilityCosts)
    for openFacilityIndex in openFacilites:
        score += potentialSitesFixedCosts[openFacilityIndex]
    return score
        
def updateScore(population, elites, score, facilityToCustomerCost, potentialSitesFixedCosts):
    for individualIndex in range(population.shape[0]):
        score[individualIndex] = calculateScore(population, individualIndex, facilityToCustomerCost, potentialSitesFixedCosts)
        
def sortAll(population, score):
    sortArgs = score.argsort()
    population = population[sortArgs]
    score = score[sortArgs]
    return (population, score)
    
def uniformCrossoverOffspring(indexA, indexB, population, maskProbability):
    offspring = np.empty((population.shape[1], ))
    for i in range(population.shape[1]):
        if np.random.uniform() < maskProbability: 
            offspring[i] = population[indexA,i]
        else: 
            offspring[i] = population[indexB,i]
    return offspring
    
def mutateOffspring(offspring, mutationProbability):
    for i in range(offspring.shape[0]):
        if np.random.uniform() < mutationProbability:
            if offspring[i] == True:
                offspring[i] = False
            else:
                offspring[i] = True

def rouletteWheelParentSelection(rank):
    rankSum = np.sum(rank)
    rand = np.random.uniform(low=0, high=rankSum)
    randOpposite = (rand + rankSum//2) % rankSum
    partialSum = 0
    parentA, parentB = None, None
    for individualIndex in range(rank.shape[0]):
        partialSum += rank[individualIndex]
        if partialSum > rand:
            parentA = individualIndex
            break
    partialSum = 0
    for individualIndex in range(rank.shape[0]):
        partialSum += rank[individualIndex]
        if partialSum > randOpposite:
            parentB = individualIndex
            return (parentA, parentB)

def replaceWeaks(population,totalOffsprings, rank, mutationRate, crossoverRate):
    for i in range(totalOffsprings):
        parentAIndex, parentBIndex = rouletteWheelParentSelection(rank)
        offspring = uniformCrossoverOffspring(parentAIndex, parentBIndex, population, crossoverRate)
        mutateOffspring(offspring, mutationRate)
        population[totalOffsprings+i, :] = np.transpose(offspring)

def bestIndividualPlan(population, individualIndex, facilityToCustomerCost):
    openFacilites = np.where(population[individualIndex, :] == True)[0]
    plan = []
    for customerIndex in range(facilityToCustomerCost.shape[1]):
        openFacilityCosts = facilityToCustomerCost[openFacilites, customerIndex]
        chosenFacilityIndex = np.where(openFacilityCosts == np.min(openFacilityCosts))[0][0]
        plan += [openFacilites[chosenFacilityIndex]]
    return plan

def punishDuplicates(population, rank):
    _, index = np.unique(population, return_index=True, axis=0)
    for individualIndex in range(population.shape[0]):
        if individualIndex not in index:
            rank[individualIndex] = 0
            
def punishElites(rank, elites):
    averageRank = np.average(rank)
    for individualIndex in range(elites):
        if rank[individualIndex] > averageRank:
            rank[individualIndex] -= averageRank
        else:
            rank[individualIndex] = 0
            
def updateRank(score, rank):
    rank[0] = rank.shape[0]
    for individualIndex in range(1,rank.shape[0]):
        if score[individualIndex] == score[individualIndex-1]:
            rank[individualIndex] = rank[individualIndex-1]
        else:
            rank[individualIndex] = rank[individualIndex-1]-1
    rank -= (np.min(rank) - 1)