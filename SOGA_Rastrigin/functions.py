#All functions defined here

#imports
import numpy as np
from agent import agent
import math
import operator



#Global variables

#possible genes
bound = 1
ub_offset = 0.0000000000000001
min_xy = -bound
max_xy = bound+ub_offset

#first generation
def genesis(size):
    population = [agent() for _ in range(size)]
    population = genes(population)
    return population

#Gene population
def genes(population):
    for agent in population:
        agent.x = np.random.uniform(min_xy,max_xy)
        agent.y = np.random.uniform(min_xy,max_xy)
    return population

#rastringin function calculation
def ras_calc(population):
    pi = math.pi
    for agent in population:
        x = agent.x
        y = agent.y
        agent.ras = 20 + x**2 + y**2  - 10*(math.cos(2*pi*x) + math.cos(2*pi*y))
    return population


#fitness
def fitness(population):
    population.sort(key=operator.attrgetter('ras'),reverse = False)
    parents = population[slice(int(len(population)/2))]
    return parents

#Major Mating
def mate(parents,mut_prob,g,G):
    children = []
    while len(children) != len(parents):
        mates = selection(parents)
        if mates == None:
            return None
        child = cross_mut(mates,mut_prob,g,G)
        children.append(child)
    return children

#pool/roullete wheel selection
def selection(parents):
    max_ras = max(agent.ras for agent in parents)
    min_ras = min(agent.ras for agent in parents)
    mates = []

    counter = 0
    while len(mates) !=2:
        candidate = parents[np.random.randint(0,len(parents))]
        ras_check = np.random.uniform(min_ras,max_ras)
        if ras_check > candidate.ras:
            mates.append(candidate)
        counter += 1
        if counter == 50:
            return None


    counter = 0
    while (mates[0] == mates[1]) and counter <=5:
        candidate = parents[np.random.randint(0, len(parents))]
        ras_check = np.random.uniform(min_ras, max_ras)
        if ras_check > candidate.ras:
            mates[1] = candidate
        counter += 1
    return mates

#Crossover and Mutation
#modifications from pinball GA
    #changing gene bias in arithematic crossover to a random value between (0,1) per Cedric's paper
    #implementing mutation equation that Cedric's paper defined to encourage convergence at higher generation counts
    #equation was slightly modified to remove the tau parameter,

def cross_mut(mates,mut_prob,g,G):
    child = agent() #new child
    bias =  np.random.uniform(0,1)

    #arithematic crossover
    child.x = bias * mates[0].x + (1-bias) * mates[1].x
    child.y = bias * mates[0].y + (1-bias) * mates[1].y

    #Mutation
    mut_check = np.random.uniform(0,1)
    if mut_check < mut_prob:
        param = np.random.randint(0,1)

        #mutation equation from Bingham et al
        #paramters
        r = np.random.uniform(0,1)  #generational bias parameter
        tau = 0 #mutation direction parameter
        while tau == 0:
            tau = np.random.randint(-1,1)

        #picking value paramter to mutate
        if param == 0:
            child.x = child.x + tau*(max_xy - min_xy)*(1 - r**(g/G))
        else:
            child.y = child.y + tau*(max_xy - min_xy)*(1 - r**(g/G))

    return child

