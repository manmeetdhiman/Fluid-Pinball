#GA runs here
#imports
from functions import genesis
from functions import cost_calc
from functions import fitness
from functions import mate
from functions import response


#Global Variables

#GA Parameters
gen = 0
target_cost = 1
max_gen = 10
size = 10
mut_prob = 0.1
fittest = []
search_limit = 50

#Simulation Parameters
dt = 5e-4
tsteps = 250

#Generation 0
population = genesis(size)
population = response(population,dt,tsteps)
population = cost_calc(gen,population)

#GA loop
while gen != max_gen:
    parents = fitness(population)
    fittest.append(parents[0])
    cost_fittest = parents[0].j_fluc
    if cost_fittest <= target_cost:
        break
    gen += 1
    children = mate(parents,mut_prob,gen,max_gen,search_limit)
    if children == None:
        break
    children = response(children,dt,tsteps)
    children = cost_calc(gen,children)
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
percent_improvement = (minimization/fittest[0].j_fluc)*100
print(f'GA minimized: {minimization}')
print(f'Percent Improvement: {percent_improvement} %')
print(f'Best objective value: {fittest[-1].j_fluc}')
print(f'With genes: {fittest[-1].genes}')
