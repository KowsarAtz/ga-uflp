# ga-uflp

This is an Implementation of _**Genetic Algorithm  for Uncapacitated Facility Location Problem**_. The implementation is an adaptation of what is proposed [Here](https://www.researchgate.net/publication/2488533_Solving_The_Simple_Plant_Location_Problem_By_Genetic_Algorithm).
For more information about the details of the implementation and [ORLIB](http://people.brunel.ac.uk/~mastjjb/jeb/orlib/uncapinfo.html) benchmark results visit this [Report](https://www.dropbox.com/s/gjujbn3a8hxv9i8/main.pdf?dl=0).

## **Getting Started**

* Download the module
```
curl -O --location https://raw.githubusercontent.com/KowsarAtz/ga-uflp/master/UFLPGeneticProblem.py
```

* In a python script, import UFLPGeneticProblem class from the downloaded module
```
from UFLPGeneticProblem import UFLPGeneticProblem
```

* Instantiate the imported class with desired parameters

```
ga = UFLPGeneticProblem(
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
    printProgress = True
)
```

* Finally run the instantiated model and get the best individual (first individual in a sorted population) as follows:

```
ga.run()
bestIndividual = ga.population[0]
```

* Also you can get the plan for the best individual using ***bestIndividualPlan*** method which returns a list of assigned facility indices to customers. *(e.g. if the returned plan is [0 1 2 1 2] it means the first customer is connected to facility 0, second and forth customers are connected to facility 1 and finally third and fifth customers are connected to facility with index 2)*

```
bestPlan = ga.bestIndividualPlan()
```

* Total Elapsed time of the execution time is also available:
```
totalTime = ga.mainLoopElapsedTime
```


## **Parameters**
A brief explanation about problem-specific parameters of the class. ***(For GA-specific parameters and a more detailed explanation, visit this [Report](https://www.dropbox.com/s/gjujbn3a8hxv9i8/main.pdf?dl=0))*** :

### ***facilityToCustomerCost***
A numpy 2 dimensional array where ***facilityToCustomerCost[i,j]*** denotes the cost of the established link between ***facility i*** and ***customer j*** 

### ***potentialSitesFixedCosts***
A numpy vector where ***potentialSitesFixedCosts[i]*** denotes the ***facility i*** fixed cost (cost of establishment).

### ***maxGenerations***
An upper bound on the total number of generations.
    
### ***nRepeat***
Beside ***maxGenerations***, nRepeat is another termination parameter. The execution of the algorithm is terminated when the best individual of the population is not changed in ***nRepeat*** number of generations.