#GA runs here
#imports
from functions import genesis
from functions import cost_calc
from functions import fitness
from functions import mate
from functions import response
from functions import rastrigin
from functions import fitness_ras
from functions import mate_ras
import random
from matplotlib import pyplot as plt
from matplotlib import figure as fig
import numpy as np
import statistics as stat
import xlsxwriter

#Generation Config  study inputs
run_type = 'Standard'  #choose Fixed Gen Config, Fixed Mut Prob, Both Fixed, or Standard
ind_limit = 150
gen_limit = 22
mut_steps = 20
mut_opt = 1
mut_type_study = 'n_genes_variable' #choose gene,motor,all, or n_genes_variable or n_genes_fixed
n_genes_study = 12
n_runs = 1068

#Fixed Generation Config Mutation study inputs
ind_op = 18
gen_op = 5
mut_op = 1
n_genes_op = 7
n_runs_mut = 1068


'''def GA(ind_limit,gen_limit,mut_steps,n_runs):
    config_perf = {
        'Config #' :[],
        'Gen_Limit' :[],
        'Ind/Gen' :[],
        'Mut_Prob' :[],

        'Fit_Ave_Fit' :[],
        'Ave_Ave_Fit' :[],
        'Weak_Ave_Fit' :[],
    }
    raw_data = {}

    #Test config modelling
    min_gen = 3
    gen_configs = list(range(min_gen,gen_limit+1))
    mut_configs = np.linspace(0,1,mut_steps+1)
    counter = 0
    for g_config in gen_configs:
         for m_config in mut_configs:
            counter += 1
            raw_data[f"Config # {counter}"] = []
            #config status
            print(f"Config # {counter}")

            config_perf['Config #'].append(counter)
            config_perf['Gen_Limit'].append(g_config)
            config_perf['Ind/Gen'].append(int(ind_limit/g_config))
            config_perf['Mut_Prob'].append(m_config)

            #intermediary Data Storage Lists
            percent_improvement = []
            ave_cost_fittest_g = []
            ave_cost_fittest_run = []
            stdv_g1 = []
            stdv_gEnd = []
            ch_stdv = []
            run = 0

            while run < n_runs:
                #iterator
                run += 1
                #performance limits
                max_ind = ind_limit
                #GA Parameters
                gen = 1
                target_cost = 0
                max_gen = g_config
                size = int(max_ind/max_gen)
                mut_prob = m_config
                mut_type = 'all' #mutation type - all,gene,motor
                n_genes = 6
                fittest = []
                search_limit = 5000

                #Simulation Parameters
                dt = 5e-4
                tsteps = 100

                #Generation 0
                population = genesis(size)
                #population = response(population,dt,tsteps)
                #population = cost_calc(gen,population)
                population = rastrigin(population)

                cost_g1_s = []
                for agent in population:
                    cost_g1_s.append(agent.ras)


                #storage for plotting
                cost_fittest_g = []
                gen_s = []

                #GA loop
                while gen <= max_gen:
                    gen_s.append(gen)
                    #parents = fitness(population)
                    parents = fitness_ras(population)


                    fittest.append(parents[0])

                    for item in range(3):
                        random.shuffle(parents)

                    #cost_fittest = parents[0].j_fluc
                    cost_fittest = parents[0].ras
                    #cost_fittest_g.append(parents[0].j_fluc)
                    cost_fittest_g.append(cost_fittest)


                    if cost_fittest <= target_cost:
                        break
                    gen += 1
                    #children = mate(parents, mut_prob, mut_type, gen, max_gen, search_limit,n_genes)
                    children = mate_ras(parents, mut_prob, mut_type, gen, max_gen, search_limit,n_genes)

                    if children == None:
                        break
                    #children = response(children,dt,tsteps)
                    #children = cost_calc(gen,children)
                    children = rastrigin(children)
                    parents.extend(children)
                    population = parents

                # final stats:
                "print(f'Defined Parameters:')
                print(f'    Target Objective Value: {target_cost}')
                print(f'    Maximum Generation Count: {max_gen}')
                print(f'    Population Size: {size}')
                print(f'    Mutation Probability:  {mut_prob*100} %')
                print('')

                if gen < max_gen:
                    print(f'Premature search stop at generation {gen}')
                else:
                    print(f'Search completed for defined generations')

                #minimization = fittest[0].j_fluc - fittest[-1].j_fluc
                minimization = fittest[0].ras - fittest[-1].ras

                #percent_improvement = (minimization/fittest[0].j_fluc)*100
                percent_improvement = abs((minimization/fittest[0].ras)*100)

                print(f'GA minimized: {minimization}')
                print(f'Percent Improvement: {percent_improvement} %')

                #print(f'Best objective value: {fittest[-1].j_fluc}')
                print(f'Best objective value: {fittest[-1].ras}')

                print(f'With genes: {fittest[-1].genes}')"

                cost_fittest_run = []
                raw_data[f"Config # {counter}"].append(fittest[-1].ras)
                cost_fittest_run.append(fittest[-1].ras)
                cost_gEnd_s = []
                for agent in population:
                    cost_gEnd_s.append(agent.ras)

                #Intermediary Performance Data
                percent_improvement.append(abs(((fittest[0].ras - fittest[-1].ras) / fittest[0].ras) * 100))
                ave_cost_fittest_g.append(stat.mean(cost_fittest_g))
                ave_cost_fittest_run.append(stat.mean(cost_fittest_run))
                stdv_g1.append(stat.pstdev(cost_g1_s))
                stdv_gEnd.append(stat.pstdev(cost_gEnd_s))
                ch_stdv.append(abs((stat.pstdev(cost_gEnd_s) - stat.pstdev(cost_g1_s))/stat.pstdev(cost_g1_s)))

            #Final Performance Data
            #config_perf['Min_%Imp'].append(min(percent_improvement))
            #config_perf['Ave_%Imp'].append(stat.mean(percent_improvement))
            #config_perf['Max_%Imp'].append(max(percent_improvement))

            config_perf['Fit_Ave_Fit'].append(min(ave_cost_fittest_run))
            config_perf['Ave_Ave_Fit'].append(stat.mean(ave_cost_fittest_run))
            config_perf['Weak_Ave_Fit'].append(max(ave_cost_fittest_run))

            #config_perf['Min_StDv_g1'].append(min(stdv_g1))
            #config_perf['Ave_StDv_g1'].append(stat.mean(stdv_g1))
            #config_perf['Max_StDv_g1'].append(max(stdv_g1))

            #config_perf['Min_StDv_gEnd'].append(min(stdv_gEnd))
            #config_perf['Ave_StDv_gEnd'].append(stat.mean(stdv_gEnd))
            #config_perf['Max_StDv_gEnd'].append(max(stdv_gEnd))

            #config_perf['Min_%Ch_StDv'].append(min(ch_stdv))
            #config_perf['Ave_%Ch_StDv'].append(stat.mean(ch_stdv))
            #config_perf['Max_%Ch_StDv'].append(max(ch_stdv))

    return config_perf,raw_data'''

def GA(run_type,ind_limit,gen_limit,mut_steps,mut_opt,mut_type_study, n_runs,n_genes_study):
    config_perf = {
        'Config #' :[],
        'Gen_Limit' :[],
        'Ind/Gen' :[],
        'N Genes': [],
        'Mut_Prob' :[],

        'Fit_Ave_Fit' :[],
        'Ave_Ave_Fit' :[],
        'Weak_Ave_Fit' :[],
    }
    raw_data = {}

    #Test config modelling
    min_gen = 3
    if run_type == 'Standard':
        gen_configs = list(range(min_gen,gen_limit+1))
        mut_configs = np.linspace(0, 1, mut_steps + 1)
    elif run_type == 'Fixed Gen Config':
        gen_configs = [gen_limit]
        mut_configs = np.linspace(0, 1, mut_steps + 1)
    elif run_type == 'Fixed Mut Prob':
        gen_configs = list(range(min_gen, gen_limit + 1))
        mut_configs = [mut_opt]
    elif run_type == 'Both Fixed':
        gen_configs = [gen_limit]
        mut_configs = [mut_opt]
    if mut_type_study == 'n_genes_variable':
        n_genes_configs = list(range(1, n_genes_study + 1))
        mut_type_study = 'n_genes'
    else:
        n_genes_configs = [n_genes_study]


    counter = 0
    for g_config in gen_configs:
         for m_config in mut_configs:
             for n_genes_config in n_genes_configs:
                counter += 1
                raw_data[f"Config # {counter}"] = []
                #config status
                print(f"Config # {counter}")

                config_perf['Config #'].append(counter)
                config_perf['Gen_Limit'].append(g_config)
                config_perf['Ind/Gen'].append(int(ind_limit/g_config))
                config_perf['N Genes'].append(n_genes_config)
                config_perf['Mut_Prob'].append(m_config)

                #intermediary Data Storage Lists
                percent_improvement = []
                ave_cost_fittest_g = []
                ave_cost_fittest_run = []
                stdv_g1 = []
                stdv_gEnd = []
                ch_stdv = []
                run = 0

                while run < n_runs:
                    #iterator
                    run += 1
                    #performance limits
                    max_ind = ind_limit
                    #GA Parameters
                    gen = 1
                    target_cost = 0
                    max_gen = g_config
                    size = int(max_ind/max_gen)
                    mut_prob = m_config
                    mut_type = mut_type_study #mutation type - all,gene,motor, n_genes
                    n_genes = n_genes_config
                    fittest = []
                    search_limit = 5000

                    #Simulation Parameters
                    dt = 5e-4
                    tsteps = 100

                    #Generation 0
                    population = genesis(size)
                    #population = response(population,dt,tsteps)
                    #population = cost_calc(gen,population)
                    population = rastrigin(population)

                    cost_g1_s = []
                    for agent in population:
                        cost_g1_s.append(agent.ras)


                    #storage for plotting
                    cost_fittest_g = []
                    gen_s = []

                    #GA loop
                    while gen <= max_gen:
                        gen_s.append(gen)
                        #parents = fitness(population)
                        parents = fitness_ras(population)


                        fittest.append(parents[0])

                        for item in range(3):
                            random.shuffle(parents)

                        #cost_fittest = parents[0].j_fluc
                        cost_fittest = parents[0].ras
                        #cost_fittest_g.append(parents[0].j_fluc)
                        cost_fittest_g.append(cost_fittest)


                        if cost_fittest <= target_cost:
                            break
                        gen += 1
                        #children = mate(parents, mut_prob, mut_type, gen, max_gen, search_limit,n_genes)
                        children = mate_ras(parents, mut_prob, mut_type, gen, max_gen, search_limit,n_genes)

                        if children == None:
                            break
                        #children = response(children,dt,tsteps)
                        #children = cost_calc(gen,children)
                        children = rastrigin(children)
                        parents.extend(children)
                        population = parents

                    # final stats:
                    '''print(f'Defined Parameters:')
                    print(f'    Target Objective Value: {target_cost}')
                    print(f'    Maximum Generation Count: {max_gen}')
                    print(f'    Population Size: {size}')
                    print(f'    Mutation Probability:  {mut_prob*100} %')
                    print('')
    
                    if gen < max_gen:
                        print(f'Premature search stop at generation {gen}')
                    else:
                        print(f'Search completed for defined generations')
    
                    #minimization = fittest[0].j_fluc - fittest[-1].j_fluc
                    minimization = fittest[0].ras - fittest[-1].ras
    
                    #percent_improvement = (minimization/fittest[0].j_fluc)*100
                    percent_improvement = abs((minimization/fittest[0].ras)*100)
    
                    print(f'GA minimized: {minimization}')
                    print(f'Percent Improvement: {percent_improvement} %')
    
                    #print(f'Best objective value: {fittest[-1].j_fluc}')
                    print(f'Best objective value: {fittest[-1].ras}')
    
                    print(f'With genes: {fittest[-1].genes}')'''

                    cost_fittest_run = []
                    raw_data[f"Config # {counter}"].append(fittest[-1].ras)
                    cost_fittest_run.append(fittest[-1].ras)
                    cost_gEnd_s = []
                    for agent in population:
                        cost_gEnd_s.append(agent.ras)

                    #Intermediary Performance Data
                    percent_improvement.append(abs(((fittest[0].ras - fittest[-1].ras) / fittest[0].ras) * 100))
                    ave_cost_fittest_g.append(stat.mean(cost_fittest_g))
                    ave_cost_fittest_run.append(stat.mean(cost_fittest_run))
                    stdv_g1.append(stat.pstdev(cost_g1_s))
                    stdv_gEnd.append(stat.pstdev(cost_gEnd_s))
                    ch_stdv.append(abs((stat.pstdev(cost_gEnd_s) - stat.pstdev(cost_g1_s))/stat.pstdev(cost_g1_s)))

                #Final Performance Data
                #config_perf['Min_%Imp'].append(min(percent_improvement))
                #config_perf['Ave_%Imp'].append(stat.mean(percent_improvement))
                #config_perf['Max_%Imp'].append(max(percent_improvement))

                config_perf['Fit_Ave_Fit'].append(min(ave_cost_fittest_run))
                config_perf['Ave_Ave_Fit'].append(stat.mean(ave_cost_fittest_run))
                config_perf['Weak_Ave_Fit'].append(max(ave_cost_fittest_run))

                #config_perf['Min_StDv_g1'].append(min(stdv_g1))
                #config_perf['Ave_StDv_g1'].append(stat.mean(stdv_g1))
                #config_perf['Max_StDv_g1'].append(max(stdv_g1))

                #config_perf['Min_StDv_gEnd'].append(min(stdv_gEnd))
                #config_perf['Ave_StDv_gEnd'].append(stat.mean(stdv_gEnd))
                #config_perf['Max_StDv_gEnd'].append(max(stdv_gEnd))

                #config_perf['Min_%Ch_StDv'].append(min(ch_stdv))
                #config_perf['Ave_%Ch_StDv'].append(stat.mean(ch_stdv))
                #config_perf['Max_%Ch_StDv'].append(max(ch_stdv))

    return config_perf,raw_data

def GA_FineMut(ind_op,gen_op,mut_prob_op,n_runs,n_genes_op):
    config_perf_FineMut = {
        'Config #' :[],
        'Mut_Prob' :[],

        'Fit_Ave_Fit' :[],
        'Ave_Ave_Fit' :[],
        'Weak_Ave_Fit' :[],
    }
    raw_data_FineMut = {}

    #Test config modelling
    mut_configs = np.linspace(mut_prob_op-0.05,mut_prob_op,11)
    counter = 0
    for m_config in mut_configs:
        counter += 1
        raw_data_FineMut[f"Config # {counter}"] = []
        #config status
        print(f"Config # {counter}")

        config_perf_FineMut['Config #'].append(counter)
        config_perf_FineMut['Mut_Prob'].append(m_config)

        #intermediary Data Storage Lists
        percent_improvement = []
        ave_cost_fittest_g = []
        ave_cost_fittest_run = []
        stdv_g1 = []
        stdv_gEnd = []
        ch_stdv = []

        run = 0
        while run < n_runs:
            #iterator
            run += 1
            #GA Parameters
            gen = 1
            target_cost = 0
            max_gen = gen_op
            size = ind_op
            mut_prob = m_config
            mut_type = 'n_gene' #mutation type - all,gene,motor
            n_genes = n_genes_op
            fittest = []
            search_limit = 5000

            #Simulation Parameters
            dt = 5e-4
            tsteps = 100

            #Generation 0
            population = genesis(size)
            #population = response(population,dt,tsteps)
            #population = cost_calc(gen,population)
            population = rastrigin(population)

            cost_g1_s = []
            for agent in population:
                cost_g1_s.append(agent.ras)


            #storage for plotting
            cost_fittest_g = []
            gen_s = []

            #GA loop
            while gen <= max_gen:
                gen_s.append(gen)
                #parents = fitness(population)
                parents = fitness_ras(population)


                fittest.append(parents[0])

                for item in range(3):
                    random.shuffle(parents)

                #cost_fittest = parents[0].j_fluc
                cost_fittest = parents[0].ras
                #cost_fittest_g.append(parents[0].j_fluc)
                cost_fittest_g.append(cost_fittest)


                if cost_fittest <= target_cost:
                    break
                gen += 1
                #children = mate(parents, mut_prob, mut_type, gen, max_gen, search_limit,n_genes)
                children = mate_ras(parents, mut_prob, mut_type, gen, max_gen, search_limit,n_genes)

                if children == None:
                    break
                #children = response(children,dt,tsteps)
                #children = cost_calc(gen,children)
                children = rastrigin(children)
                parents.extend(children)
                population = parents

            # final stats:
            '''print(f'Defined Parameters:')
            print(f'    Target Objective Value: {target_cost}')
            print(f'    Maximum Generation Count: {max_gen}')
            print(f'    Population Size: {size}')
            print(f'    Mutation Probability:  {mut_prob*100} %')
            print('')

            if gen < max_gen:
                print(f'Premature search stop at generation {gen}')
            else:
                print(f'Search completed for defined generations')

            #minimization = fittest[0].j_fluc - fittest[-1].j_fluc
            minimization = fittest[0].ras - fittest[-1].ras

            #percent_improvement = (minimization/fittest[0].j_fluc)*100
            percent_improvement = abs((minimization/fittest[0].ras)*100)

            print(f'GA minimized: {minimization}')
            print(f'Percent Improvement: {percent_improvement} %')

            #print(f'Best objective value: {fittest[-1].j_fluc}')
            print(f'Best objective value: {fittest[-1].ras}')

            print(f'With genes: {fittest[-1].genes}')'''

            cost_fittest_run = []
            raw_data_FineMut[f"Config # {counter}"].append(fittest[-1].ras)
            cost_fittest_run.append(fittest[-1].ras)
            cost_gEnd_s = []
            for agent in population:
                cost_gEnd_s.append(agent.ras)

            #Intermediary Performance Data
            percent_improvement.append(abs(((fittest[0].ras - fittest[-1].ras) / fittest[0].ras) * 100))
            ave_cost_fittest_g.append(stat.mean(cost_fittest_g))
            ave_cost_fittest_run.append(stat.mean(cost_fittest_run))
            stdv_g1.append(stat.pstdev(cost_g1_s))
            stdv_gEnd.append(stat.pstdev(cost_gEnd_s))
            ch_stdv.append(abs((stat.pstdev(cost_gEnd_s) - stat.pstdev(cost_g1_s))/stat.pstdev(cost_g1_s)))

        #Final Performance Data
        #config_perf['Min_%Imp'].append(min(percent_improvement))
        #config_perf['Ave_%Imp'].append(stat.mean(percent_improvement))
        #config_perf['Max_%Imp'].append(max(percent_improvement))

        config_perf_FineMut['Fit_Ave_Fit'].append(min(ave_cost_fittest_run))
        config_perf_FineMut['Ave_Ave_Fit'].append(stat.mean(ave_cost_fittest_run))
        config_perf_FineMut['Weak_Ave_Fit'].append(max(ave_cost_fittest_run))

        #config_perf['Min_StDv_g1'].append(min(stdv_g1))
        #config_perf['Ave_StDv_g1'].append(stat.mean(stdv_g1))
        #config_perf['Max_StDv_g1'].append(max(stdv_g1))

        #config_perf['Min_StDv_gEnd'].append(min(stdv_gEnd))
        #config_perf['Ave_StDv_gEnd'].append(stat.mean(stdv_gEnd))
        #config_perf['Max_StDv_gEnd'].append(max(stdv_gEnd))

        #config_perf['Min_%Ch_StDv'].append(min(ch_stdv))
        #config_perf['Ave_%Ch_StDv'].append(stat.mean(ch_stdv))
        #config_perf['Max_%Ch_StDv'].append(max(ch_stdv))

    return config_perf_FineMut,raw_data_FineMut

def data_out(study,raw_data,n_runs,ind_limit): #7 columns and 273 data rows and 1 title row
    study_titles = list(study.keys())
    raw_data_titles = list(raw_data.keys())
    outWorkbook = xlsxwriter.Workbook(f"{n_runs}Runs_{ind_limit}IndLimit.xlsx")
    outSheet1 = outWorkbook.add_worksheet('Study Data')
    outSheet2 = outWorkbook.add_worksheet('Raw Data')
    #writing out
    for std_col_id in range(len(study_titles)):
        for std_row_id in range(len(study['Config #']) + 1):
            if std_row_id  == 0:
                outSheet1.write(std_row_id, std_col_id, study_titles[std_col_id])
            else:
                outSheet1.write(std_row_id, std_col_id, study[study_titles[std_col_id]][std_row_id - 1])
    for raw_col_id in range(len(raw_data_titles)):
        for raw_row_id in range(len(raw_data['Config # 1']) + 1):
            if raw_row_id == 0:
                outSheet2.write(raw_row_id,raw_col_id,raw_data_titles[raw_col_id])
            else:
                outSheet2.write(raw_row_id, raw_col_id, raw_data[raw_data_titles[raw_col_id]][raw_row_id - 1])

    outWorkbook.close()

def plotting(study,raw_data):
    keys = ['Ave_']
    data_type = 'Ave_Fit'

    plt.figure(figsize=(8, 4))
    plots = []
    for key in keys:
        plots.append(plt.scatter(study['Config #'], study[key + data_type]))
        plt.title(f'Average of Fittest Individuals over {n_runs} runs of Each Configuration')
    plt.legend(plots, keys, loc='upper right')
    plt.xlabel('Configuration ID')
    plt.ylabel('Average Fittest Rastrigin Evaluation')

    # extracting best config parameters
    bci = study['Ave_' + data_type].index((min(study['Ave_' + data_type])))
    best_config_id = study['Config #'][bci]
    best_gen_limit = study['Gen_Limit'][bci]
    best_ind_gen = study['Ind/Gen'][bci]
    best_mut_prob = round(study['Mut_Prob'][bci], 2)
    best_fitness = round(study['Ave_Ave_Fit'][bci], 4)
    if 'N Genes' in study:
        best_n_genes = study['N Genes'][bci]
    else:
        best_n_genes = "N/A"

    # Final Parameter Printing
    best_config_data = f"Total # of Configs: {max(study['Config #'])}\nRuns Per Config: {n_runs}\nTotal Individuals: {best_gen_limit * best_ind_gen}\nTotal # of Runs: {max(study['Config #'])*n_runs}\n\nBest Config: {best_config_id}\nBest Gen Limit: {best_gen_limit}\nBest Ind Per Gen: {best_ind_gen}\nBest # Mut Genes: {best_n_genes}\nBest Mut Prob: {best_mut_prob}\nAvg of Best Fitness: {best_fitness}"
    plt.subplots_adjust(right=0.65)
    plt.text(max(study['Config #']) + 0.1*max(study['Config #']), min(study['Ave_' + data_type]), best_config_data, fontsize=12)
    plt.savefig(f"{n_runs}Runs_{ind_limit}IndLimit_Data.png")

    # histogram of fittest cost in all runs for best config
    plt.figure(figsize=(8, 4))
    plt.hist(raw_data[f'Config # {best_config_id}'], density=False, bins=n_runs)
    plt.title('Histogram of Fittest Cost From all Runs with Best Config ')
    plt.xlabel('Rastrigin Evaluation Value')
    plt.ylabel('Frequency')
    plt.savefig(f"{n_runs}Runs_{ind_limit}IndLimit_Histogram.png")
    plt.show()

def plotting_FineMut(study,raw_data):
    keys = ['Fit_', 'Ave_', 'Weak_']
    data_type = 'Ave_Fit'

    plt.figure(figsize=(8, 4))
    plots = []
    for key in keys:
        plots.append(plt.scatter(study['Config #'], study[key + data_type]))
        plt.title(f'Average of Fittest Individuals over {n_runs} runs of Each Configuration')
    plt.legend(plots, keys, loc='upper right')
    plt.xlabel('Configuration ID')
    plt.ylabel('Average Fittest Rastrigin Evaluation')

    # extracting best config parameters
    bci = study['Ave_' + data_type].index((min(study['Ave_' + data_type])))
    best_config_id = study['Config #'][bci]
    best_mut_prob = round(study['Mut_Prob'][bci], 2)
    best_fitness = round(study['Fit_Ave_Fit'][bci], 4)

    # Final Parameter Printing
    best_config_data = f"Total # of Configs: {max(study['Config #'])}\nRuns Per Config: {n_runs_mut}\nTotal # of Runs: {max(study['Config #'])*n_runs}\n\nBest Config: {best_config_id}\nBest Mut Prob: {best_mut_prob}\nBest Fitness: {best_fitness}"
    plt.subplots_adjust(right=0.65)
    plt.text(max(study['Config #']) + 1, max(study['Fit_' + data_type]) , best_config_data, fontsize=12)
    plt.savefig(f"{n_runs_mut}Runs_{ind_op}Ind_Data.png")

    # histogram of fittest cost in all runs for best config
    plt.figure(figsize=(8, 4))
    plt.hist(raw_data[f'Config # {best_config_id}'], density=False, bins=n_runs_mut)
    plt.title('Histogram of Fittest Cost From all Runs with Best Config ')
    plt.xlabel('Rastrigin Evaluation Value')
    plt.ylabel('Frequency')
    plt.savefig(f"{n_runs_mut}Runs_{ind_op}Ind_Histogram.png")
    plt.show()

#Generation Config and Coarse Mutation Config Test
#study, raw_data = GA(ind_limit, gen_limit, mut_steps, n_runs)
#data_out(study, raw_data, n_runs, ind_limit)
#plotting(study,raw_data)

#Generation config with N_genes mutation architecture
study, raw_data = GA(run_type,ind_limit, gen_limit, mut_steps, mut_opt, mut_type_study, n_runs,n_genes_study)
data_out(study, raw_data, n_runs, ind_limit)
plotting(study,raw_data)

#Fixed Generation Config and Fine Mutation Config Test
#study_mut, raw_data_mut = GA_FineMut(ind_op,gen_op,mut_op,n_runs_mut,n_genes_op)
#data_out(study_mut, raw_data_mut,n_runs_mut,ind_op)
#plotting_FineMut(study_mut,raw_data_mut)







