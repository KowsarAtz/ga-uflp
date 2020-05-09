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