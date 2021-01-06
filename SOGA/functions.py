#All functions defined here

#imports
from individual import individual
import numpy as np
import math
from PI_motor_GA import PI_motor
import random
from matplotlib import pyplot as plt


#Global variables
offset = 0.0000000000000001 #for random.uniform()

#motor parameters
omega_lim = 1048 #rad/s
phase_lim = math.pi #rad
freq_u_lim = 52.9044 #rad/s or 8.42 Hz
freq_l_lim = 27.2690 #rad/s or 4.34 Hz

#rastrigin parameters
bound_ras = 5.12
mockfreq_u = 5.12
mockfreq_l = mockfreq_u / 2



#gene limits:
u_bound= {
    'amplitude': bound_ras + offset,
    'frequency': mockfreq_u + offset,
    'phase': bound_ras+ offset,
    'offset': bound_ras+ offset
}
l_bound= {
    'amplitude': -bound_ras ,
    'frequency': mockfreq_l,
    'phase': -bound_ras,
    'offset': -bound_ras
}
#dictionary
possible_genes = {
    'amplitude': [l_bound['amplitude'],u_bound['amplitude']], #RPM
    'frequency': [l_bound['frequency'],u_bound['frequency']], #rad/s
    'phase': [l_bound['phase'],u_bound['phase']], #rad
    'offset': [l_bound['offset'],u_bound['offset']] #RPM
}
gene_keys = ['amplitude','frequency','phase','offset']

#first generation
def genesis(size):
    population = [individual() for _ in range(size)]
    population = genes(population = population)
    return population

#Gene population
def genes(population):
    for individual in population:
        for motor in range(3):
            for key in gene_keys:
                if key == 'frequency':
                    bin_value = random.uniform(possible_genes[key][0],possible_genes[key][1])
                    freq_gene = random.choice([0,bin_value])
                    individual.genes[key].append(freq_gene)
                    continue
                individual.genes[key].append(random.uniform(possible_genes[key][0],possible_genes[key][1]))
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
def mate(parents,mut_prob,mut_type,g,G,limit,n_genes = 1):
    children = []
    mate_pool = selection(parents = parents)
    while len(children) != len(parents):
        child = cross_mut(mate_pool = mate_pool,mut_prob = mut_prob,g = g,G = G,limit = limit,n_genes = n_genes)
        if child == None:
            return None
        children.append(child)
    return children

#Pool Selection
def selection(parents):
    mate_pool = []
    shuffled = [None]*len(parents)
    for i in range(len(parents)):
        shuffled[i-1] = parents[i]

    for i in range(len(parents)):
        if parents[i].j_fluc < shuffled[i].j_fluc:
            mate_pool.append(parents[i])
        elif parents[i].j_fluc == shuffled[i].j_fluc:
            choices  = [parents[i],shuffled[i]]
            mate_pool.append(random.choice(choices))
        else:
            mate_pool.append(shuffled[i])
    return mate_pool

#Crossover and Mutation
def cross_mut(mate_pool,mut_prob,mut_type,g,G,limit,n_genes = 1):
    child = individual()
    p0 = 0
    p1 = 0
    counter = 0
    while p0 == p1:
        p0 = random.choice(mate_pool)
        p1 = random.choice(mate_pool)

   #loop protection
    if counter > limit:
        return None

    #arithmetic crossover
    for motor in range(3):
        for key in gene_keys:
            bias = random.uniform(0, 1 + offset)
            gene = bias * p0.genes[key][motor] + (1-bias) * p1.genes[key][motor]
            if gene > u_bound[key]:
                gene = u_bound[key]
            if gene < l_bound[key]:
                gene = l_bound[key]
            if key == 'frequency':
                if gene > u_bound[key]:
                    gene = u_bound[key]
                if gene < l_bound[key]:
                    gene = 0
            child.genes[key].append(gene)
    #mutation
    mut_check = random.uniform(0,1)
    r = random.uniform(0,1+offset)
    tau = random.choice([-1,1])

    if mut_check < mut_prob:
        if mut_type == 'all':
            for motor in range(3):
                for key in gene_keys:
                    mut_gene = child.genes[key][motor] + tau*(u_bound[key] - l_bound[key])*(1-r**(g/G))
                    if mut_gene > u_bound[key]:
                        mut_gene = u_bound[key]
                    if mut_gene < l_bound[key]:
                        mut_gene = l_bound[key]
                    if key == 'frequency':
                        mut_gene = child.genes[key][motor] + tau * (u_bound[key] - 0) * (1 - r ** (g / G))
                        if mut_gene > u_bound[key]:
                            mut_gene = u_bound[key]
                        if mut_gene < l_bound[key]:
                            mut_gene = 0
                    child.genes[key][motor] = mut_gene

        if mut_type == 'motor':
            motor = random.choice(range(3))
            for key in gene_keys:
                mut_gene = child.genes[key][motor] + tau * (u_bound[key] - l_bound[key]) * (1 - r ** (g / G))
                if mut_gene > u_bound[key]:
                    mut_gene = u_bound[key]
                if mut_gene < l_bound[key]:
                    mut_gene = l_bound[key]
                if key == 'frequency':
                    mut_gene = child.genes[key][motor] + tau * (u_bound[key] - 0) * (1 - r ** (g / G))
                    if mut_gene > u_bound[key]:
                        mut_gene = u_bound[key]
                    if mut_gene < l_bound[key]:
                        mut_gene = 0
                child.genes[key][motor] = mut_gene

        if mut_type == 'gene':
            motor = random.choice(range(3))
            key = random.choice(gene_keys)
            mut_gene = child.genes[key][motor] + tau * (u_bound[key] - l_bound[key]) * (1 - r ** (g / G))
            if mut_gene > u_bound[key]:
                mut_gene = u_bound[key]
            if mut_gene < l_bound[key]:
                mut_gene = l_bound[key]
            if key == 'frequency':
                mut_gene = child.genes[key][motor] + tau * (u_bound[key] - 0) * (1 - r ** (g / G))
                if mut_gene > u_bound[key]:
                    mut_gene = u_bound[key]
                if mut_gene < l_bound[key]:
                    mut_gene = 0
            child.genes[key][motor] = mut_gene

        if mut_type == 'n_genes':
            shuffled_motors = list(range(3))
            shuffled_keys = gene_keys.copy()
            for _ in range(3):
                random.shuffle(shuffled_motors)
                random.shuffle(shuffled_keys)
            mut_counter = 0
            while mut_counter <= n_genes:
                for motor in shuffled_motors:
                    for key in shuffled_keys:
                        mut_counter += 1
                        mut_gene = child.genes[key][motor] + tau*(u_bound[key] - l_bound[key])*(1-r**(g/G))
                        if mut_gene > u_bound[key]:
                            mut_gene = u_bound[key]
                        if mut_gene < l_bound[key]:
                            mut_gene = l_bound[key]
                        if key == 'frequency':
                            mut_gene = child.genes[key][motor] + tau * (u_bound[key] - 0) * (1 - r ** (g / G))
                            if mut_gene > u_bound[key]:
                                mut_gene = u_bound[key]
                            if mut_gene < l_bound[key]:
                                mut_gene = 0
                        child.genes[key][motor] = mut_gene
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
            individual.revolutions[motor] = PI_motor(w_des = individual.revolutions[motor],dt = dt,tsteps = tsteps)
    return population

#Rastrigin Functions, cross_mut is common for both
#cost
def rastrigin(population):
    A = 10
    n = 12
    for agent in population:
        ras = A*n
        for motor in range(3):
            for key in gene_keys:
                x = agent.genes[key][motor]
                ras = ras + (x**2 - A*math.cos(2*math.pi*x))
        agent.ras = ras
    return population

#fitness
def fitness_ras(population):
    population.sort(key=lambda agent: agent.ras,reverse = False)
    parents = population[slice(int(len(population)/2))]
    return parents

#mating
def mate_ras(parents,mut_prob,mut_type,g,G,limit,n_genes = 1):
    children = []
    mate_pool = selection_ras(parents = parents)
    while len(children) != len(parents):
        child = cross_mut(mate_pool = mate_pool,mut_prob = mut_prob,mut_type = mut_type,g = g,G = G,limit = limit,n_genes = n_genes)
        if child == None:
            return None
        children.append(child)
    return children

#selection
def selection_ras(parents):
    mate_pool = []
    shuffled = [None]*len(parents)
    for i in range(len(parents)):
        shuffled[i-1] = parents[i]

    for i in range(len(parents)):
        if parents[i].ras < shuffled[i].ras:
            mate_pool.append(parents[i])
        elif parents[i].ras == shuffled[i].ras:
            choices = [parents[i],shuffled[i]]
            mate_pool.append(random.choice(choices))
        else:
            mate_pool.append(shuffled[i])
    return mate_pool













