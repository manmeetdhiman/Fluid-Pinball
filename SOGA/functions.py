#All functions defined here

from individual import individual
import numpy as np
import math
from PI_motor_GA import PI_motor
import random
from matplotlib import pyplot as plt
from matplotlib import figure as fig
import yaml
from pprint import pprint
#import CFD.cfd

GA_type = 'Rastrigin' #choose Rastrigin or Pinball
offset = 0.0000000000000001 #for random.uniform()

##motor parameters
omega_lim = 1048 #rad/s
phase_lim = math.pi #rad
freq_u_lim = 52.9044 #rad/s or 8.42 Hz
freq_l_lim = 27.2690 #rad/s or 4.34 Hz

#rastrigin parameters
bound_ras = 5.12
mockfreq_u = 5.12
mockfreq_l = mockfreq_u / 2

#gene limits:
if GA_type == 'Rastrigin':
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
elif GA_type == 'Pinball':
    u_bound = {
        'amplitude': omega_lim + offset,
        'frequency': freq_u_lim + offset,
        'phase': phase_lim + offset,
        'offset': omega_lim + offset
    }
    l_bound = {
        'amplitude': -omega_lim,
        'frequency': freq_l_lim,
        'phase': -phase_lim,
        'offset': -omega_lim
    }

possible_genes = {
    'amplitude': [l_bound['amplitude'],u_bound['amplitude']], #rad/s
    'frequency': [l_bound['frequency'],u_bound['frequency']], #rad/s
    'phase': [l_bound['phase'],u_bound['phase']], #rad
    'offset': [l_bound['offset'],u_bound['offset']] #rad/s
}
gene_keys = ['amplitude','frequency','phase','offset']


#first generation
def genesis(size):
    population = [individual() for _ in range(size)]
    population = genes(population = population)
    return population

#Fluid Pinball GA
def GA(starting_gen,ga_type, target_cost, max_gen, size, mut_prob, mut_type, search_limit, dt, tsteps,n_genes,abs_counter):
    fittest = []
    cost_fittest_s = []
    gen_s = []


    if ga_type == 'Pinball':
        print(f'Starting from generation {starting_gen}')
        if starting_gen == 0:
            population = genesis(size = size)
            population,abs_counter = tag(group = population,abs_counter = abs_counter)
            population = response(population = population, dt = dt, tsteps = tsteps)
            population = cost_calc(gen = starting_gen, population = population)
        else:
            population = load_population_from_file(starting_gen)

        for gen in range(starting_gen+1,max_gen+1):
            gen_s.append(gen)
            parents = fitness(population = population)
            fittest.append(parents[0])
            cost_fittest = parents[0].j_fluc
            cost_fittest_s.append(parents[0].j_fluc)
            if cost_fittest <= target_cost:
                print(f'Fittest cost {cost_fittest} less than equal target cost {target_cost}.')
                break
            for item in range(3):
                random.shuffle(parents)

            children = mate(parents=parents, mut_prob=mut_prob, mut_type=mut_type, g=gen, G=max_gen, limit=search_limit,n_genes = n_genes)
            children,abs_counter = tag(group = children, abs_counter = abs_counter)
            if children == None:
                break
            children = response(population=children, dt=dt, tsteps=tsteps)
            children = cost_calc(gen=gen, population=children)

            parents.extend(children)
            population = parents
            output_population_to_file(population = population, gen = gen)

        # return parameters
        minimization = fittest[0].j_fluc - fittest[-1].j_fluc
        percent_improvement = (minimization / fittest[0].j_fluc) * 100
        bov = fittest[-1].j_fluc

    if ga_type == 'Rastrigin':
        population = genesis(size=size)
        population, abs_counter = tag(group=population, abs_counter=abs_counter)
        population = rastrigin(population = population)

        for gen in range(starting_gen+1,max_gen+1):
            gen_s.append(gen)
            parents = fitness_ras(population = population)
            fittest.append(parents[0])
            cost_fittest = parents[0].ras
            cost_fittest_s.append(cost_fittest)
            if cost_fittest <= target_cost:
                break
            for item in range(3):
                random.shuffle(parents)

            children = mate_ras(parents = parents, mut_prob = mut_prob, mut_type = mut_type, g = gen, G = max_gen, limit = search_limit,n_genes = n_genes)
            children, abs_counter = tag(group=children, abs_counter=abs_counter)
            if children == None:
                break

            children = rastrigin(children)
            parents.extend(children)
            population = parents

        #parameters
        minimization = fittest[0].ras - fittest[-1].ras
        percent_improvement = abs((minimization/fittest[0].ras)*100)
        bov = fittest[-1].ras

    return gen_s,fittest,cost_fittest_s, minimization, percent_improvement, bov

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

#cost assignment function, CFD Called here
def cost_calc(gen, population):
    cost = CFD.cfd.run_CFD(gen, population, submit_slurm=True, GA_to_CFD=True, compute_motor_rotation=False)
    for i in range(len(population)):
        population[i].j_fluc = cost[i]['Jtotal']
    return population

#Fitness
def fitness(population):
    population.sort(key=lambda individual: individual.j_fluc,reverse = False)
    parents = population[slice(int(len(population)/2))]
    return parents

#Major Mating
def mate(parents,mut_prob,mut_type,g,G,limit,n_genes):
    children = []
    mate_pool = selection(parents = parents)
    while len(children) != len(parents):
        child = cross_mut(mate_pool = mate_pool,mut_prob = mut_prob,mut_type = mut_type,g = g,G = G,limit = limit,n_genes = n_genes)
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
def cross_mut(mate_pool,mut_prob,mut_type,g,G,limit,n_genes):
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
            gene = sat_lim(gene = gene,key = key)
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
                    if key == 'frequency':
                        mut_gene = child.genes[key][motor] + tau * (u_bound[key] - 0) * (1 - r ** (g / G))
                    mut_gene = sat_lim(gene = mut_gene,key = key)
                    child.genes[key][motor] = mut_gene

        if mut_type == 'motor':
            motor = random.choice(range(3))
            for key in gene_keys:
                mut_gene = child.genes[key][motor] + tau * (u_bound[key] - l_bound[key]) * (1 - r ** (g / G))
                if key == 'frequency':
                    mut_gene = child.genes[key][motor] + tau * (u_bound[key] - 0) * (1 - r ** (g / G))
                mut_gene = sat_lim(gene=mut_gene, key=key)
                child.genes[key][motor] = mut_gene

        if mut_type == 'gene':
            motor = random.choice(range(3))
            key = random.choice(gene_keys)
            mut_gene = child.genes[key][motor] + tau * (u_bound[key] - l_bound[key]) * (1 - r ** (g / G))
            if key == 'frequency':
                mut_gene = child.genes[key][motor] + tau * (u_bound[key] - 0) * (1 - r ** (g / G))
            mut_gene = sat_lim(gene=mut_gene, key=key)
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
                        if key == 'frequency':
                            mut_gene = child.genes[key][motor] + tau * (u_bound[key] - 0) * (1 - r ** (g / G))
                        mut_gene = sat_lim(gene=mut_gene, key=key)
                        child.genes[key][motor] = mut_gene

        if mut_type == 'rand_genes':
            shuffled_motors = list(range(3))
            shuffled_keys = gene_keys.copy()
            for _ in range(3):
                random.shuffle(shuffled_motors)
                random.shuffle(shuffled_keys)

            mut_counter = 0
            rand_genes = random.choice(list(range(1,12)))
            while mut_counter <= rand_genes:
                for motor in shuffled_motors:
                    for key in shuffled_keys:
                        mut_counter += 1
                        mut_gene = child.genes[key][motor] + tau*(u_bound[key] - l_bound[key])*(1-r**(g/G))
                        if key == 'frequency':
                            mut_gene = child.genes[key][motor] + tau * (u_bound[key] - 0) * (1 - r ** (g / G))
                        mut_gene = sat_lim(gene=mut_gene, key=key)
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
def mate_ras(parents,mut_prob,mut_type,g,G,limit,n_genes):
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

def stats(target_cost,max_gen,size,mut_prob,gen,minimization,percent_improvement, bov, fittest, gen_s, cost_fittest_s):
    # final stats:
    print(f'Defined Parameters:')
    print(f'    Target Objective Value: {target_cost}')
    print(f'    Maximum Generation Count: {max_gen}')
    print(f'    Population Size: {size}')
    print(f'    Mutation Probability:  {mut_prob * 100} %')
    print('')

    if gen < max_gen:
        print(f'Premature search stop at generation {gen}')
    else:
        print(f'Search completed for defined generations')

    print(f'GA minimized: {minimization}')
    print(f'Percent Improvement: {percent_improvement} %')
    print(f'Best objective value: {bov}')
    pprint(fittest[-1].genes)

    # plotting
    fig.Figure()
    plt.scatter(gen_s, cost_fittest_s)
    plt.title('Best Rastrigin Evaluation From Each Generation')
    plt.xlabel('Generation')
    plt.ylabel('Rastrigin Evaluation')
    plt.xlim([0, max(gen_s) + 1])
    plt.ylim([0, max(cost_fittest_s) + 10])
    plt.text(0.5, 10, f"{percent_improvement} %")
    plt.show()

#saturation limiter for genes
def sat_lim(gene,key):
    if gene > u_bound[key]:
        gene = u_bound[key]
    if gene < l_bound[key]:
        gene = l_bound[key]
    if key == 'frequency':
        if gene > u_bound[key]:
            gene = u_bound[key]
        if gene < l_bound[key]:
            gene = 0
    return gene

#saving population data to a file
def output_population_to_file(population, gen):
    print(f'Outputting population {gen} to file')

    output = {}
    for n, ind in enumerate(population):
        output[n] = {}
        output[n]['genes'] = ind.genes
        output[n]['j_fluc'] = ind.j_fluc
        temp_rev_list = [sublist.tolist() for sublist in ind.revolutions]
        output[n]['revolutions'] = temp_rev_list

    with open(f'population-gen-{gen}.yaml', 'w') as f:
         yaml.dump(output, f)

#loading population data from file
def load_population_from_file(gen):
    with open(f'population-gen-{gen}.yaml', 'r') as f:
        reload = yaml.safe_load(f)
    reload_population = []
    for n in reload:
        ind = individual()
        ind.genes = reload[n]['genes']
        ind.j_fluc = reload[n]['j_fluc']
        ind.revolutions = reload[n]['revolutions']

        reload_population.append(ind)

    return reload_population


def tag(group,abs_counter):
    for individual in group:
        print(abs_counter)
        individual.id = abs_counter
        abs_counter += 1
    return group,abs_counter

def parse_j_fluc(raw_data,type):
    parsed_data = []
    return_data = []
    for gen in raw_data.keys():
        for ind in raw_data[gen].keys():
            parsed_data.append(raw_data[gen][ind]['j_fluc'])
        if type == 'best':
            return_data.append(min(parsed_data))
        if type == 'avg':
            return_data.append(stats.mean(parsed_data))
        if type == 'worst':
            return_data.append(max(parsed_data))
        if type == 'all':
            return_data.append(parsed_data)
    return return_data

def plotter(type,x,y,title,xlabel,ylabel,label,legend):
    if type == 'scatter':
        plt.plot(x,y,label = label)
    if type == 'line':
        plt.scatter(x,y,label = label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend == True:
        plt.legend([label])
    plt.show()







