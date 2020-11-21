#All functions defined here

#imports
from individual import individual
import numpy as np
import math
from PI_motor import PI_motor


#Global variables
possible_genes = {
    'amplitude': range(0,10000), #RPM
    'frequency': range(0,10), #rad/s
    'phase': range(0,10), #rad
    'offset': range(0,10) #RPM
}
gene_keys = ['amplitude','frequency','phase','offset']

#first generation
def genesis(size):
    population = [individual() for i in range(size)]
    population = genes(population)
    return population


#Gene population
def genes(population):
    for individual in population:
        for motor in range(3):
            for key in gene_keys:
                individual.genes[key].append(np.random.uniform(possible_genes[key]))
    return population


#cost assignment function
def cost_calc(gen, population):
    cost = CFD(gen, population)
    for i in range(len(population)):
        population[i].j_fluc = cost[i]
    return population

#Fitness
def fitness(population):
    population.sort(key=lambda individual: individual.j_fluc)
    parents = population[slice(len(population)/2)]
    return parents

#Major Mating
def mate(parents,mut_prob):
    children = []
    while len(children) != len(parents):
        mates = selection(parents)
        child = cross_mut(mates,mut_prob)
        children.append(child)
    return children

#Pool Selection
def selection(parents):
    max_cost = max(individual.j_fluc for individual in parents)
    min_cost = min(individual.j_fluc for individual in parents)
    cost_range = range(min_cost,max_cost)
    mates = []
    while len(mates) !=2 and (mates[0] == mates[2]):
        candidate = parents[np.random.randint(0,len(parents))]
        cost_check = np.random.uniform(cost_range)
        if cost_check > candidate.j_fluc:
            mates.append(candidate)
    return mates

#Crossover and Mutation
def cross_mut(mates,mut_prob):
    child = individual() #new child
    mates.sort(key=lambda individual: individual.j_fluc) #mates sorted in ascending order
    cost_ratio = mates[0].j_fluc/mates[1].j_fluc #for crossover bias

    if cost_ratio < 0.5:
        bias = 1-cost_ratio
    if cost_ratio >= 0.5:
        bias = cost_ratio
    #arithematic crossover
        for motor in range(3):
            for key in gene_keys:
                child.genes[key][motor] = bias*mates[0].genes[key][motor] + (1-bias)*mates[1].genes[key][motor]

    #Mutation
    mut_check = np.random.uniform(0,1)
    if mut_check < mut_prob:
        motor = np.random.randint(0,2)
        for key in gene_keys:
            child.genes[key][motor] = np.random.uniform(possible_genes[key])

    return child

#response
def response(population,dt,tsteps):
    tf  = dt*tsteps
    t_sim = np.linspace(0,tf,tsteps)

    #calculating desired motor speed, in front top bottom order
    for individual in population:
        for motor in range(3):
            individual.revolutions[motor] = []      #resetting revolutions vector from previous generation
            for t in t_sim:
                A = individual.genes['amplitude'][motor]
                f = individual.genes['frequency'][motor]
                ph = individual.genes['phase'][motor]
                offset = individual.genes['offset'][motor]

                w_desired = A*math.sin(f*t + ph) + offset
                individual.revolutions[motor].append(w_desired)

            #updating revolutions to realistic motor response
            individual.revolutions[motor] = PI_motor(individual.revolutions[motor],dt,tsteps)

    return population













