#GA runs here
#imports
from functions_R2 import GA
from functions_R2 import stats
import individual as config


#GA Loop
gen_s, fittest, cost_fittest_s, minimization, percent_improvement, bov, population_data = \
    GA(
    starting_gen = config.starting_gen,
    target_cost = config.target_cost,
    max_gen = config.max_gen,
    size = config.size,
    mut_prob = config.mut_prob,
    mut_type = config.mut_type,
    search_limit = config.search_limit,
    dt = config.dt,
    tsteps = config.tsteps,
    n_genes = config.n_genes,
    abs_counter = config.abs_counter
    )


#Printing and Plotting
stats(
    target_cost = config.target_cost,
    max_gen = config.max_gen,
    size = config.size,
    mut_prob=  config.mut_prob,
    gen = gen_s[-1],
    minimization = minimization,
    percent_improvement = percent_improvement,
    bov = bov, fittest = fittest,
    gen_s = gen_s,
    cost_fittest_s = cost_fittest_s,
    population_data = population_data
)

