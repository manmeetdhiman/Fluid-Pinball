#All functions defined here

#imports
from individual import individual
import numpy as np
import math
from PI_motor_GA import PI_motor


#Global variables
ub_offset = 0.0000000000000001

#gene limits:
u_bound= {
    'amplitude': 10000,
    'frequency': 10,
    'phase': 10,
    'offset': 10
}
#dictionary
possible_genes = {
    'amplitude': range(0,u_bound['amplitude']), #RPM
    'frequency': range(0,u_bound['frequency']), #rad/s
    'phase': range(0,u_bound['phase']), #rad
    'offset': range(0,u_bound['offset']) #RPM
}
gene_keys = ['amplitude','frequency','phase','offset']

#first generation
def genesis(size):
    population = [individual() for _ in range(size)]
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
    population.sort(key=lambda individual: individual.j_fluc,reverse = False)
    parents = population[slice(int(len(population)/2))]
    return parents

#Major Mating
def mate(parents,mut_prob,g,G,limit):
    children = []
    while len(children) != len(parents):
        mates = selection(parents,limit)
        if mates == None:
            return None
        child = cross_mut(mates,mut_prob,g,G)
        children.append(child)
    return children

#Pool Selection
def selection(parents,limit):
    max_cost = max(individual.j_fluc for individual in parents)
    min_cost = min(individual.j_fluc for individual in parents)
    mates = []

    counter = 0
    while len(mates) !=2:
        candidate = parents[np.random.randint(0,len(parents))]
        cost_check = np.random.uniform(min_cost,max_cost)
        if cost_check > candidate.j_fluc:
            mates.append(candidate)
        counter +=1
        if counter == limit:
            return None

    counter = 0
    while (mates[0] == mates[2]):
        candidate = parents[np.random.randint(0, len(parents))]
        cost_check = np.random.uniform(min_cost, max_cost)
        if cost_check > candidate.j_fluc:
            mates[1] = candidate
        counter +=1
        if counter == limit:
            return None
    return mates

#Crossover and Mutation
def cross_mut(mates,mut_prob,g,G):
    child = individual() #new child
    bias = np.random.uniform(0,1)
    #arithematic crossover
    for motor in range(3):
        for key in gene_keys:
            child.genes[key][motor] = bias*mates[0].genes[key][motor] + (1-bias)*mates[1].genes[key][motor]

    #Mutation
    mut_check = np.random.uniform(0,1)
    if mut_check < mut_prob:
        motor = np.random.randint(0,2)
        r = np.random.uniform(0,1)
        tau = 0
        while tau == 0:
            tau = np.random.randint(-1,1)
        for key in gene_keys:
            child.genes[key][motor] = child.genes[key][motor] + tau*(u_bound[key] - 0)*(1 - r**(g/G))
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













