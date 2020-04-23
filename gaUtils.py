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

def replaceWeaks(population,totalOffsprings):
    parentIndexes = list(range(totalOffsprings))
    shuffle(parentIndexes)
    for i in range(totalOffsprings):
        parentAIndex = parentIndexes[i%totalOffsprings]
        parentBIndex = parentIndexes[(i+1)%totalOffsprings]
        offspring = uniformCrossoverOffspring(parentAIndex, parentBIndex, population, 0.3)
        mutateOffspring(offspring, 0.05)
        population[totalOffsprings+i, :] = np.transpose(offspring)

def bestIndividualPlan(population, individualIndex, facilityToCustomerCost):
    openFacilites = np.where(population[individualIndex, :] == True)[0]
    plan = []
    for customerIndex in range(facilityToCustomerCost.shape[1]):
        openFacilityCosts = facilityToCustomerCost[openFacilites, customerIndex]
        chosenFacilityIndex = np.where(openFacilityCosts == np.min(openFacilityCosts))[0][0]
        plan += [openFacilites[chosenFacilityIndex]]
    return plan

def punishDuplicates(population, score):
    _, index = np.unique(population, return_index=True, axis=0)
    for individualIndex in range(population.shape[0]):
        if individualIndex not in index:
            score[individualIndex] = np.finfo(np.float64).max