#GA runs here
from functions import genesis
from functions import genes
from functions import cost_calc
from functions import fitness
from functions import mate
from functions import response


#Global Variables

#GA Parameters
gen = 1
target_cost = 1
max_gen = 1
size = 1
mut_prob = 1

#Simulation Parameters
dt = 5e-4
tsteps = 250

#Generation 0
population = genesis(size)
population = genes(population)
population = response(population,dt,tsteps)
population = cost_calc(gen,population)

#GA loop
while gen != max_gen:
    parents = fitness(population)
    gen += 1
    children = mate(parents,mut_prob)
    children = response(children,dt,tsteps)
    children = cost_calc(gen,children)
    population = parents.append(children)
    cost_fittest = parents[0].j_fluc
    if cost_fittest <= target_cost:
        break

