import re
import sys
from statistics import stdev
avg = lambda x: sum(x)/len(x)

MAX_GENERATIONS = {
    'A': (200, '16 * 50'),
    'B': (400, '25 * 50'),
    'C': (2000, '50 * 50'), 
    'D': (4000, '100 * 1000'),
}

datasets = []

# A: 16 * 50 (n. o. potential facility locations * n. o. customers)
datasets += [('cap/40/cap4%d' % (i+1), MAX_GENERATIONS['A'][0], MAX_GENERATIONS['A'][1]) for i in range(4)]
datasets += [('cap/50/cap51', MAX_GENERATIONS['A'][0], MAX_GENERATIONS['A'][1])]
datasets += [('cap/60/cap6%d' % (i+1), MAX_GENERATIONS['A'][0], MAX_GENERATIONS['A'][1]) for i in range(4)]
datasets += [('uncap/70/cap7%d' % (i+1), MAX_GENERATIONS['A'][0], MAX_GENERATIONS['A'][1]) for i in range(4)]

# B: 25 * 50
datasets += [('cap/80/cap8%d' % (i+1), MAX_GENERATIONS['B'][0], MAX_GENERATIONS['B'][1]) for i in range(4)]
datasets += [('cap/90/cap9%d' % (i+1), MAX_GENERATIONS['B'][0], MAX_GENERATIONS['B'][1]) for i in range(4)]
datasets += [('uncap/100/cap10%d' % (i+1), MAX_GENERATIONS['B'][0], MAX_GENERATIONS['B'][1]) for i in range(4)]

# C: 50 * 50
datasets += [('cap/110/cap11%d' % (i+1), MAX_GENERATIONS['C'][0], MAX_GENERATIONS['C'][1]) for i in range(4)]
datasets += [('cap/120/cap12%d' % (i+1), MAX_GENERATIONS['C'][0], MAX_GENERATIONS['C'][1]) for i in range(4)]
datasets += [('uncap/130/cap13%d' % (i+1), MAX_GENERATIONS['C'][0], MAX_GENERATIONS['C'][1]) for i in range(4)]

# D: 100 * 1000
datasets += [('uncap/a-c/cap%s' % s, MAX_GENERATIONS['D'][0], MAX_GENERATIONS['D'][1]) for s in ['a', 'b', 'c']]

fileName = sys.argv[1]
test_str = open(fileName,'r').read()

print('dataset,dimension,avg. generation,avg. time,max generation, max time')

for dataset in datasets:
    regex = r"%s[\s\S]*?total generations of (\d+)[\s\S]*?best found elapsed time: (\d*.\d*)$" % dataset[0]
    matches = re.finditer(regex, test_str, re.MULTILINE)
    reached = []
    times = []
    for matchNum, match in enumerate(matches, start=1):
        reached += [int(match.group(1))]
        times += [float(match.group(2))]
    print('{dataset},{dimension},{avgReached},{avgTime},{maxReached},{maxTime}'.format(
        dataset = dataset[0],
        dimension = dataset[2],
        avgReached = round(avg(reached), 2),
        maxReached = max(reached),
        avgTime = round(avg(times), 3),
        maxTime = round(max(times), 3)
    ))

