import numpy as np
from random import shuffle

def calculateFitness(population, individualIndex, facilityToCustomerCost, potentialSitesFixedCosts):
    openFacilites = np.where(population[individualIndex, :] == True)[0]
    fitness = 0
    for customerIndex in range(facilityToCustomerCost.shape[1]):
        openFacilityCosts = facilityToCustomerCost[openFacilites, customerIndex]
        fitness += np.min(openFacilityCosts)
    for openFacilityIndex in openFacilites:
        fitness += potentialSitesFixedCosts[openFacilityIndex]
    return fitness
        
def updateFitness(population, fitness, facilityToCustomerCost, potentialSitesFixedCosts):
    for individualIndex in range(population.shape[0]):
        fitness[individualIndex] = calculateFitness(population, individualIndex, facilityToCustomerCost, potentialSitesFixedCosts)
        
def sortAll(population, fitness):
    sortArgs = fitness.argsort()
    population = population[sortArgs]
    fitness = fitness[sortArgs]
    return (population, fitness)
    
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