#GA_rastringin runs here

#imports
import numpy as np
from functions import genesis
from functions import ras_calc
from functions import fitness
from functions import mate
from agent import agent

#Global Variables
g = 0         #instantaenous generation count
target_ras = 0  #global minima for convergence criteria
max_gen = 50  #generation for exit criteria
size = 100    #population size
mut_prob = 0.1  #overall mutation probability
fittest = []

#generation 0
population = genesis(size)
population = ras_calc(population)


while g <= max_gen: #exit criteria
    #evaluation
    parents = fitness(population)
    fittest.append(parents[0])
    ras_fittest = parents[0].ras

    # convergence criteria
    if ras_fittest <= target_ras:
        break

    #generation update (crossover,mutation,etc)
    g += 1
    children = mate(parents,mut_prob,g,max_gen)
    if children == None:
        break
    children = ras_calc(children)
    parents.extend(children) #
    population = parents

#final stats:
print(f'Defined Parameters:')
print(f'    Target Objective Value: {target_ras}')
print(f'    Maximum Generation Count: {max_gen}')
print(f'    Population Size: {size}')
print(f'    Mutation Probability:  {mut_prob*100} %')
print('')

if g<max_gen:
    print(f'Premature search stop at generation {g}')
else:
    print(f'Search completed for defined generations')

minimization = fittest[0].ras - fittest[-1].ras
percent_improvement = (minimization/fittest[0].ras)*100
print(f'GA minimized: {minimization}')
print(f'Percent Improvement: {percent_improvement} %')
print(f'Best objective value: {fittest[-1].ras}')
print(f'With x = {fittest[-1].x} and y = {fittest[-1].y}')