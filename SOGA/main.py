#GA runs here
#imports
from functions import GA
from functions import stats


#performance limits
max_ind = 90
size = 18

#GA Parameters
starting_gen = 0
ga_type = 'Rastrigin' #choose Rastrigin or Pinball
target_cost = 0
max_gen = int((max_ind - size)/(size/2))
mut_prob = 1
mut_type = 'all' #Mutation type - all,gene,motor,n_genes, rand_genes
search_limit = 50000000

if mut_type == 'n_genes':
    n_genes = 7
else:
    n_genes = 1



#Simulation Parameters
dt = 5e-4
tsteps = 100


#GA Loop
gen_s, fittest, cost_fittest_s, minimization, percent_improvement, bov = GA(starting_gen = starting_gen,ga_type = ga_type, target_cost = target_cost,
                                                                            max_gen = max_gen, size = size, mut_prob = mut_prob,
                                                                            mut_type = mut_type, search_limit = search_limit,
                                                                            dt = dt, tsteps = tsteps,n_genes = n_genes,abs_counter = 0)

#Printing and Plotting
stats(target_cost = target_cost,max_gen = max_gen,size = size,mut_prob=  mut_prob,gen = gen_s[-1],minimization = minimization,
      percent_improvement = percent_improvement, bov = bov, fittest = fittest, gen_s = gen_s, cost_fittest_s = cost_fittest_s)

