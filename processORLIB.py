import numpy as np

def getCostMatrices(orlibPath, orlibDataset, orlibCostValuePerLine = 7):
    f = open(orlibPath+orlibDataset+'.txt', 'r')
    (totalPotentialSites, totalCustomers) = [int(string) for string in f.readline().split()]
    potentialSitesFixedCosts = np.empty((totalPotentialSites,))
    facilityToCustomerCost = np.empty((totalPotentialSites, totalCustomers))
    
    for i in range(totalPotentialSites):
        potentialSitesFixedCosts[i] = np.float64(f.readline().split()[1])
    
    for j in range(totalCustomers):
        demand = np.float64(f.readline())
        lineItems = f.readline().split()
        for i in range(totalPotentialSites):
            facilityToCustomerCost[i,j] = np.float64(lineItems[i%orlibCostValuePerLine])
            if i%orlibCostValuePerLine == orlibCostValuePerLine - 1:
                lineItems = f.readline().split()

    return (facilityToCustomerCost, potentialSitesFixedCosts)

def getOptimals(orlibPath, orlibDataset):
    f = open(orlibPath+orlibDataset+'.txt.opt', 'r')
    optimals = f.readline().split()
    optimalCost = float(optimals[-1])
    optimals = [int(string) for string in optimals[:-1]]
    return (optimals, optimalCost)

def compareResults(orlibDatasetName, totalGeneration, bestFoundCost, bestPlan, optimalCost, optimals, mainLoopElapsedTime, bestIndividualRepeatedTime, fout):
    reached = False
    print('\rdataset name:',orlibDatasetName, file=fout)
    print('total generations of', totalGeneration, file=fout)
    print('best individual score', bestFoundCost,\
            'repeated for last', bestIndividualRepeatedTime,'times', file=fout)
    if sorted(bestPlan) == sorted(optimals):
        reached = True
        print('REACHED OPTIMAL OF', bestFoundCost, file=fout)
    else:
        errorPercentage = (bestFoundCost - optimalCost) * 100 / optimalCost
        print('DID NOT REACHED OPTIMAL OF', optimalCost, "|", errorPercentage,"% ERROR", file=fout)
    print('total elapsed time:', mainLoopElapsedTime, file=fout)
    assignedFacilitiesString = ''
    for f in bestPlan:
        assignedFacilitiesString += str(f) + ' '
    print('assigned facilities:', file=fout)
    print(assignedFacilitiesString, file=fout)
    return reached