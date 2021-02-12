#All functions defined here

from individual import individual
import numpy as np
import math
from PI_motor_GA import PI_motor
import random
from matplotlib import pyplot as plt
import yaml
from pprint import pprint
from individual import sim_type
#import CFD.cfd

GA_type = sim_type
offset = 0.0000000000000001 #for random.uniform()

##motor parameters
omega_lim = 1048 #rad/s
phase_lim = math.pi #rad
freq_u_lim = 52.9044 #rad/s or 8.42 Hz
freq_l_lim = 31.4159265 #rad/s

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
def GA(starting_gen, target_cost, max_gen, size, mut_prob, mut_type, search_limit, dt, tsteps,n_genes,abs_counter,gen_buffer_limit):

    fittest = []
    cost_fittest_s = []
    gen_s = []
    sat_counter = 0
    buffer_count = 0
    population_data = {}



    if GA_type == 'Pinball':
        print(f'Starting from generation {starting_gen}')
        if starting_gen == 0:
            population = genesis(size = size)
            population,abs_counter = tag(group = population,abs_counter = abs_counter)
            population = response(population = population, dt = dt, tsteps = tsteps)
            population = cost_calc(gen = starting_gen, population = population)
            output_population(population=population, gen=starting_gen)
            output_saturation_parameters(gen=starting_gen, buffer_count=buffer_count,
                                         sat_counter=sat_counter)
        else:
            population = load_population(starting_gen)
            buffer_count, sat_counter = load_saturation_parameters(starting_gen)
            cost_fittest_s = load_cost_fittest()

        for gen in range(starting_gen+1,max_gen+1):
            population_data[gen] = population
            gen_s.append(gen)
            parents = fitness(population = population)
            fittest.append(parents[0])
            cost_fittest = parents[0].j_total
            cost_fittest_s.append(cost_fittest)

            buffer_count += 1
            if buffer_count >= gen_buffer_limit:
                message = exit_check(cost_fittest_s = cost_fittest_s,target_cost = target_cost,
                                     sat_counter = sat_counter, buffer_count=buffer_count)
                if message == 'first_saturation':
                    print('first saturation')
                    mut_prob = 0.3
                    mut_type = 'motor'
                    sat_counter += 1
                    buffer_count = 0
                if message == 'second_saturation':
                    print('second saturation')
                    mut_prob = 1
                    mut_type = 'gene'
                    sat_counter += 1
                    buffer_count = 0
                if message == 'exit':
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
            output_population(population = population, gen = gen)
            output_saturation_parameters(gen=gen, buffer_count=buffer_count,
                                         sat_counter=sat_counter)
            output_cost_fittest(cost_fittest_s)

        # return parameters
        minimization = fittest[0].j_total - fittest[-1].j_total
        percent_improvement = (minimization / fittest[0].j_total) * 100
        bov = fittest[-1].j_total

    if GA_type == 'Rastrigin':
        population = genesis(size=size)
        population, abs_counter = tag(group=population, abs_counter=abs_counter)
        population = rastrigin(population = population)

        for gen in range(starting_gen+1,max_gen+1):
            population_data[gen] = population
            gen_s.append(gen)
            parents = fitness_ras(population = population)
            fittest.append(parents[0])
            cost_fittest = parents[0].ras
            cost_fittest_s.append(cost_fittest)

            buffer_count += 1
            if buffer_count >= gen_buffer_limit:
                message = exit_check(cost_fittest_s=cost_fittest_s, target_cost=target_cost,
                                     sat_counter=sat_counter, buffer_count=buffer_count)
                if message == 'first_saturation':
                    mut_prob = 0.3
                    mut_type = 'motor'
                    sat_counter += 1
                    buffer_count = 0
                if message == 'second_saturation':
                    mut_prob = 1
                    mut_type = 'gene'
                    sat_counter += 1
                    buffer_count = 0
                if message == 'exit':
                    print(f'Fittest cost {cost_fittest} less than equal target cost {target_cost}.')
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

    return gen_s,fittest,cost_fittest_s, minimization, percent_improvement, bov, population_data

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
    data = CFD.cfd.GA_run_CFD(gen, population)
    for i in range(len(population)):
        individual_id = population[i].id
        population[i].j_act = data[individual_id]['Jact']
        population[i].j_fluc = data[individual_id]['Jfluc']
        population[i].j_total = data[individual_id]['Jtotal']
        population[i].sensor = data[individual_id]['sensor_data']
    return population

#Fitness
def fitness(population):
    population.sort(key=lambda individual: individual.j_total,reverse = False)
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
        if parents[i].j_total < shuffled[i].j_total:
            mate_pool.append(parents[i])
        elif parents[i].j_total == shuffled[i].j_total:
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

def stats(target_cost,max_gen,size,mut_prob,gen,minimization,percent_improvement, bov, fittest, gen_s, cost_fittest_s,population_data):
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
    plt.figure(1)
    plt.scatter(gen_s, cost_fittest_s)
    plt.title('Best Rastrigin Evaluation From Each Generation')
    plt.xlabel('Generation')
    plt.ylabel('Rastrigin Evaluation')
    plt.xlim([0, max(gen_s) + 1])
    plt.ylim([0, max(cost_fittest_s) + 10])
    plt.text(0.5, 10, f"{percent_improvement} %")

    plt.figure(2)
    for gen in range(1, max(population_data.keys())):
        cost = []
        for ind in range(size):
            cost.append(population_data[gen][ind].ras)
        plt.scatter([gen] * size , cost, color='green')
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
def output_population(population, gen):
    print(f'Outputting population {gen} to file')

    output = {}
    for individual in population:
        output[individual.id] = {}
        output[individual.id]['genes'] = individual.genes
        output[individual.id]['j_fluc'] = individual.j_fluc
        output[individual.id]['j_act'] = individual.j_act
        output[individual.id]['j_total'] = individual.j_total
        output[individual.id]['sensor'] = individual.sensor
        temp_rev_list = [sublist.tolist() for sublist in individual.revolutions]
        output[individual.id]['revolutions'] = temp_rev_list

    with open(f'population-gen-{gen}.yaml', 'w') as f:
         yaml.dump(output, f)

#loading population data from file
def load_population(gen):
    with open(f'population-gen-{gen}.yaml', 'r') as f:
        reload = yaml.safe_load(f)
    reload_population = []
    for individual_id in reload:
        ind = individual()
        ind.id = individual_id
        ind.genes = reload[individual_id]['genes']
        ind.j_fluc = reload[individual_id]['j_fluc']
        ind.j_act = reload[individual_id]['j_act']
        ind.j_total = reload[individual_id]['j_total']
        ind.sensor = reload[individual_id]['sensor']
        ind.revolutions = reload[individual_id]['revolutions']
        # Needs to be np array not just regular list.
        ind.revolutions[0] = np.asarray(ind.revolutions[0])
        ind.revolutions[1] = np.asarray(ind.revolutions[1])
        ind.revolutions[2] = np.asarray(ind.revolutions[2])

        reload_population.append(ind)

    return reload_population


def output_saturation_parameters(gen, buffer_count, sat_counter):
    output = {}
    output['buffer_count'] = buffer_count
    output['sat_counter'] = sat_counter
    with open(f'saturation_parameters-{gen}.yaml', 'w') as f:
         yaml.dump(output, f)


def load_saturation_parameters(gen):
    with open(f'saturation_parameters-{gen}.yaml', 'r') as f:
        reload = yaml.safe_load(f)

    buffer_count = reload['buffer_count']
    sat_counter = reload['sat_counter']

    return buffer_count, sat_counter


def output_cost_fittest(cost_fittest):
    with open(f'cost_fittest.yaml', 'w') as f:
        yaml.dump(cost_fittest, f)


def load_cost_fittest():
    with open(f'cost_fittest.yaml', 'r') as f:
        cost_fittest = yaml.safe_load(f)

    return cost_fittest


def tag(group,abs_counter):
    for individual in group:
        individual.id = abs_counter
        abs_counter += 1
    return group,abs_counter

def parse_j(raw_data,type,key):
    return_data = []
    for gen in raw_data.keys():
        parsed_data = []
        for ind in raw_data[gen].keys():
            parsed_data.append(raw_data[gen][ind][key])
        if type == 'best':
            return_data.append([min(parsed_data)])
        if type == 'avg':
            return_data.append([stats.mean(parsed_data)])
        if type == 'worst':
            return_data.append([max(parsed_data)])
        if type == 'all':
            return_data.append(parsed_data)
    return return_data

def plotter(type,data,total_gen,title='',xlabel='',label='',individual_id = None, gen = None):
    if type == 'j_plot':
        cost_keys = ['j_act', 'j_fluc', 'j_total']
        js_all = {}
        js_best = {}
        j = {}
        for key in cost_keys:
            j_all = parse_j(raw_data=data, type='all', key=key)
            j_best = parse_j(raw_data=data, type='best', key=key)
            js_all[key] = j_all
            js_best[key] = j_best

        best_individuals = list(data[total_gen].keys())
        best_individual_index = js_all['j_total'][-1].index(min(js_all['j_total'][-1]))
        best_individual_ID = best_individuals[best_individual_index]
        best_j_total = round(js_best['j_total'][-1][0],3)
        j['js_all'] = js_all
        j['js_best'] = js_best

        for cost in j.keys():
            data = j[cost]
            n_plots = 3
            fig, axs = plt.subplots(n_plots, 1, sharex=True)
            i = 0
            for key in data.keys():
                for gen in range(0, total_gen+1):
                    axs[i].scatter([gen]*len(data[key][gen]), data[key][gen],color = 'green',label = label)
                    axs[i].set(ylabel = key)
                    plt.xticks(np.arange(0,total_gen+1,step = 1))

                i+=1
            plt.xlabel(xlabel)
            fig.suptitle(title + f' Best Individual is {best_individual_ID} with j_total of {best_j_total}')
            plt.show()
    if type == 'individual':
        sensor_data = []
        sensor_keys = []
        motor_keys = ['front','top','bottom']
        revolutions = data[gen][individual_id]['revolutions']
        for i in range(3):
            revolutions[i] = [rev * 9.5493 for rev in revolutions[i]]

        for key in data[gen][individual_id]['sensor'].keys():
            sensor_data.append(list(data[gen][individual_id]['sensor'][key].values()))
            sensor_keys.append(key)
        fig, axs = plt.subplots(4, 1,sharex=True)
        for i in range(3):
            axs[0].plot(list(range(len(revolutions[0]))),revolutions[i],label = motor_keys[i])
            axs[0].set(ylabel = 'Motor RPMs',ylim = ((-10000,10000)),xlim = ((0,2000)))
            axs[0].legend()
            axs[i+1].plot(list(data[gen][individual_id]['sensor']['top'].keys()),sensor_data[i])
            axs[i+1].set(ylabel = sensor_keys[i],ylim = ((0,5)),xlim = ((0,2000)))
        fig.suptitle(title + f', Individual {individual_id} in Generation {gen}')
        plt.xlabel(xlabel)
        plt.show()

def exit_check(cost_fittest_s,target_cost,sat_counter, buffer_count):
    p_diffs = []
    for i in range(1,buffer_count):
        p_diff = abs((cost_fittest_s[-1] - cost_fittest_s[-1-i])/((cost_fittest_s[-1] + cost_fittest_s[-1-i])/2))*100
        p_diffs.append(p_diff)
    if max(p_diffs) <= 0.5:
        if sat_counter == 0:
            return 'first_saturation'
        if sat_counter == 1:
            return 'second_saturation'
        if sat_counter == 2:
            return 'exit'
    if cost_fittest_s[-1] == target_cost:
        return 'exit'
    return 'no action'









