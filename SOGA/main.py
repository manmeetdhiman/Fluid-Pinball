#GA runs here
#imports
from functions import genesis
from functions import cost_calc
from functions import fitness
from functions import mate
from functions import response
from functions import rastrigin
from functions import fitness_ras
from functions import mate_ras
import random


#Global Variables

#GA Parameters
gen = 0
target_cost = 0
max_gen = 100
size = 100
mut_prob = 0.1
mut_type = 'gene' #mutation type - all,gene,motor
fittest = []
search_limit = 5000

#Simulation Parameters
dt = 5e-4
tsteps = 250

#Generation 0
population = genesis(size)
population = response(population,dt,tsteps)
population = cost_calc(gen,population)
#population = rastrigin(population)

#GA loop
while gen != max_gen:
    parents = fitness(population)
    #parents = fitness_ras(population)

    fittest.append(parents[0])

    for item in range(3):
        random.shuffle(parents)

    cost_fittest = parents[0].j_fluc
    #cost_fittest = parents[0].ras


    if cost_fittest <= target_cost:
        break
    gen += 1
    children = mate(parents, mut_prob, mut_type, gen, max_gen, search_limit)
    #children = mate_ras(parents, mut_prob, mut_type, gen, max_gen, search_limit)

    if children == None:
        break
    children = response(children,dt,tsteps)
    children = cost_calc(gen,children)
    #children = rastrigin(children)

    parents.extend(children)
    population = parents

#final stats:
print(f'Defined Parameters:')
print(f'    Target Objective Value: {target_cost}')
print(f'    Maximum Generation Count: {max_gen}')
print(f'    Population Size: {size}')
print(f'    Mutation Probability:  {mut_prob*100} %')
print('')

if gen < max_gen:
    print(f'Premature search stop at generation {gen}')
else:
    print(f'Search completed for defined generations')

minimization = fittest[0].j_fluc - fittest[-1].j_fluc
#minimization = fittest[0].ras - fittest[-1].ras

percent_improvement = (minimization/fittest[0].j_fluc)*100
#percent_improvement = (minimization/fittest[0].ras)*100

print(f'GA minimized: {minimization}')
print(f'Percent Improvement: {percent_improvement} %')

print(f'Best objective value: {fittest[-1].j_fluc}')
#print(f'Best objective value: {fittest[-1].ras}')

print(f'With genes: {fittest[-1].genes}')
