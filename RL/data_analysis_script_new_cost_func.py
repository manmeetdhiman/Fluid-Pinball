import pickle
import numpy as np
import matplotlib.pyplot as plt
import json

num_iterations=10
num_policies=5
CFD_timestep=5e-4
CFD_timestep_spacing=5
num_actions=15
dur_ramp=0.05
shedding_freq=8.42
free_stream_vel=1.5
sampling_periods=1.5
dur_action_one_add=0.50
Json_files = False

shedding_period=1/shedding_freq
CFD_timesteps_period=shedding_period/CFD_timestep
CFD_timesteps_action_one_add=CFD_timesteps_period*dur_action_one_add
CFD_timesteps_ramp=CFD_timesteps_period*dur_ramp

master_data=[]
actor_losses=[]
critic_losses=[]

for policy in range(num_policies):
    for iteration in range(num_iterations):
        iteration_ID=policy*num_iterations+iteration+1
        filename='../../Production Runs/Production Run 2/pickle_files/data_iteration_' + str(iteration_ID)+'.pickle'
        with open(filename, 'rb') as handle:
            data = pickle.load(handle)
        master_data.append(data)
    filename='../../Production Runs/Production Run 2/actor_critic_losses/actor_critic_losses_'+str(policy+1)+'.pickle'
    with open(filename,'rb') as handle:
        data=pickle.load(handle)
    actor_losses.append(data['actor_losses'][-1])
    critic_losses.append(data['critic_losses'][-1])

def calculate_reward(rewards_data):    
    total_rewards = np.sum(rewards_data)
    avg_rewards = np.mean(rewards_data)

    return total_rewards, avg_rewards

def calculate_J_fluc(top_sens_data, mid_sens_data, bot_sens_data, free_stream_vel, CFD_timesteps_period,
                     CFD_timesteps_actions, CFD_timestep_spacing, num_actions, sampling_periods):
    
    sampling_timesteps = int(CFD_timesteps_period * sampling_periods / CFD_timestep_spacing)
    J_fluc = []
    
    for i in range(num_actions):
        timestep_end=np.sum(CFD_timesteps_actions[:(i+1)])
        sampling_end=int(timestep_end / CFD_timestep_spacing)
        
        top_sens_var=np.var(top_sens_data[(sampling_end-sampling_timesteps):sampling_end])
        mid_sens_var=np.var(mid_sens_data[(sampling_end-sampling_timesteps):sampling_end])
        bot_sens_var=np.var(bot_sens_data[(sampling_end-sampling_timesteps):sampling_end])
        
        J_fluc_temp=np.mean([top_sens_var,mid_sens_var,bot_sens_var])
        J_fluc_temp=J_fluc_temp/free_stream_vel**2
        
        J_fluc.append(J_fluc_temp)


    total_J_fluc = np.sum(J_fluc)
    avg_J_fluc= np.mean(J_fluc)

    return total_J_fluc,avg_J_fluc

def calculate_J_fluc_new(top_sens_data, mid_sens_data, bot_sens_data, free_stream_vel, CFD_timesteps_period,
                         CFD_timesteps_actions, CFD_timestep_spacing, num_actions, sampling_periods):
    
    sampling_timesteps = int(CFD_timesteps_period * sampling_periods / CFD_timestep_spacing)
    J_fluc = []
    
    for i in range(num_actions):
        timestep_end=np.sum(CFD_timesteps_actions[:(i+1)])
        sampling_end=int(timestep_end / CFD_timestep_spacing)
        
        top_sens_var=np.var(top_sens_data[(sampling_end-sampling_timesteps):sampling_end])
        mid_sens_var=np.var(mid_sens_data[(sampling_end-sampling_timesteps):sampling_end])
        bot_sens_var=np.var(bot_sens_data[(sampling_end-sampling_timesteps):sampling_end])
        
        J_fluc_temp=np.mean([top_sens_var,mid_sens_var,bot_sens_var])
        J_fluc_temp=J_fluc_temp/free_stream_vel**2
        J_fluc_temp=np.tanh(12.2*J_fluc_temp)
        
        J_fluc.append(J_fluc_temp)


    total_J_fluc = np.sum(J_fluc)
    avg_J_fluc= np.mean(J_fluc)

    return total_J_fluc,avg_J_fluc

def calculate_J_fluc_actions(top_sens_data, mid_sens_data, bot_sens_data, free_stream_vel, CFD_timesteps_period,
                             CFD_timesteps_actions, CFD_timestep_spacing, num_actions, sampling_periods):
    
    sampling_timesteps = int(CFD_timesteps_period * sampling_periods / CFD_timestep_spacing)
    J_fluc = []
    
    for i in range(num_actions):
        timestep_end=np.sum(CFD_timesteps_actions[:(i+1)])
        sampling_end=int(timestep_end / CFD_timestep_spacing)
        
        top_sens_var=np.var(top_sens_data[(sampling_end-sampling_timesteps):sampling_end])
        mid_sens_var=np.var(mid_sens_data[(sampling_end-sampling_timesteps):sampling_end])
        bot_sens_var=np.var(bot_sens_data[(sampling_end-sampling_timesteps):sampling_end])
        
        J_fluc_temp=np.mean([top_sens_var,mid_sens_var,bot_sens_var])
        J_fluc_temp=J_fluc_temp/free_stream_vel**2
        
        J_fluc.append(J_fluc_temp)

    return J_fluc

def calculate_J_fluc_new_actions(top_sens_data, mid_sens_data, bot_sens_data, free_stream_vel, CFD_timesteps_period,
                                 CFD_timesteps_actions, CFD_timestep_spacing, num_actions, sampling_periods):
    
    sampling_timesteps = int(CFD_timesteps_period * sampling_periods / CFD_timestep_spacing)
    J_fluc = []
    
    for i in range(num_actions):
        timestep_end=np.sum(CFD_timesteps_actions[:(i+1)])
        sampling_end=int(timestep_end / CFD_timestep_spacing)
        
        top_sens_var=np.var(top_sens_data[(sampling_end-sampling_timesteps):sampling_end])
        mid_sens_var=np.var(mid_sens_data[(sampling_end-sampling_timesteps):sampling_end])
        bot_sens_var=np.var(bot_sens_data[(sampling_end-sampling_timesteps):sampling_end])
        
        J_fluc_temp=np.mean([top_sens_var,mid_sens_var,bot_sens_var])
        J_fluc_temp=J_fluc_temp/free_stream_vel**2
        J_fluc_temp=np.tanh(12.2*J_fluc_temp)
        
        J_fluc.append(J_fluc_temp)

    return J_fluc

def calculate_J_act(front_cyl_data, top_cyl_data, bot_cyl_data, free_stream_vel, 
                    CFD_timesteps_actions, num_actions,CFD_timesteps_ramp):
    
    J_act = []
    
    for i in range(num_actions):
        if i==0:
            sampling_start=int(CFD_timesteps_ramp)
            sampling_end=int(CFD_timesteps_actions[0])
        else:
            sampling_start=int(np.sum(CFD_timesteps_actions[:i]))
            sampling_end=int(sampling_start+CFD_timesteps_actions[i])
            sampling_start=int(sampling_start+CFD_timesteps_ramp)
        
        sampling_timesteps=sampling_end-sampling_start
            
        front_cyl_act=0
        top_cyl_act=0
        bot_cyl_act=0
        
        for j in range(sampling_start,sampling_end):
            front_cyl_act += front_cyl_data[j-1]**2
            top_cyl_act += top_cyl_data[j-1]**2
            bot_cyl_act += bot_cyl_data[j-1]**2
        
        J_act_temp=front_cyl_act+top_cyl_act+bot_cyl_act
        J_act_temp=np.sqrt(J_act_temp/(3*sampling_timesteps))
        J_act_temp=J_act_temp*0.01/free_stream_vel
        
        J_act.append(J_act_temp)


    total_J_act = np.sum(J_act)
    avg_J_act= np.mean(J_act)

    return total_J_act,avg_J_act

def calculate_J_act_new(front_cyl_data, top_cyl_data, bot_cyl_data, free_stream_vel, 
                        CFD_timesteps_actions, num_actions,CFD_timesteps_ramp):
    
    J_act = []
    
    for i in range(num_actions):
        if i==0:
            sampling_start=int(CFD_timesteps_ramp)
            sampling_end=int(CFD_timesteps_actions[0])
        else:
            sampling_start=int(np.sum(CFD_timesteps_actions[:i]))
            sampling_end=int(sampling_start+CFD_timesteps_actions[i])
            sampling_start=int(sampling_start+CFD_timesteps_ramp)
            
        sampling_timesteps=sampling_end-sampling_start
            
        front_cyl_act=0
        top_cyl_act=0
        bot_cyl_act=0
        
        for j in range(sampling_start,sampling_end):
            front_cyl_act += front_cyl_data[j-1]**2
            top_cyl_act += top_cyl_data[j-1]**2
            bot_cyl_act += bot_cyl_data[j-1]**2
        
        J_act_temp=front_cyl_act+top_cyl_act+bot_cyl_act
        J_act_temp=np.sqrt(J_act_temp/(3*sampling_timesteps))
        J_act_temp=J_act_temp*0.01/free_stream_vel
        J_act_temp=np.tanh(0.7*J_act_temp)
        
        J_act.append(J_act_temp)


    total_J_act = np.sum(J_act)
    avg_J_act= np.mean(J_act)

    return total_J_act,avg_J_act

def calculate_J_act_actions(front_cyl_data, top_cyl_data, bot_cyl_data, free_stream_vel, 
                            CFD_timesteps_actions, num_actions,CFD_timesteps_ramp):
    
    J_act = []
    
    for i in range(num_actions):
        if i==0:
            sampling_start=int(CFD_timesteps_ramp)
            sampling_end=int(CFD_timesteps_actions[0])
        else:
            sampling_start=int(np.sum(CFD_timesteps_actions[:i]))
            sampling_end=int(sampling_start+CFD_timesteps_actions[i])
            sampling_start=int(sampling_start+CFD_timesteps_ramp)
            
        sampling_timesteps=sampling_end-sampling_start
            
        front_cyl_act=0
        top_cyl_act=0
        bot_cyl_act=0
        
        for j in range(sampling_start,sampling_end):
            front_cyl_act += front_cyl_data[j-1]**2
            top_cyl_act += top_cyl_data[j-1]**2
            bot_cyl_act += bot_cyl_data[j-1]**2
        
        J_act_temp=front_cyl_act+top_cyl_act+bot_cyl_act
        J_act_temp=np.sqrt(J_act_temp/(3*sampling_timesteps))
        J_act_temp=J_act_temp*0.01/free_stream_vel
        
        J_act.append(J_act_temp)

    return J_act

def calculate_J_act_new_actions(front_cyl_data, top_cyl_data, bot_cyl_data, free_stream_vel, 
                                CFD_timesteps_actions, num_actions,CFD_timesteps_ramp):
    
    J_act = []
    
    for i in range(num_actions):
        if i==0:
            sampling_start=int(CFD_timesteps_ramp)
            sampling_end=int(CFD_timesteps_actions[0])
        else:
            sampling_start=int(np.sum(CFD_timesteps_actions[:i]))
            sampling_end=int(sampling_start+CFD_timesteps_actions[i])
            sampling_start=int(sampling_start+CFD_timesteps_ramp)
            
        sampling_timesteps=sampling_end-sampling_start
        
        front_cyl_act=0
        top_cyl_act=0
        bot_cyl_act=0
        
        for j in range(sampling_start,sampling_end):
            front_cyl_act += front_cyl_data[j-1]**2
            top_cyl_act += top_cyl_data[j-1]**2
            bot_cyl_act += bot_cyl_data[j-1]**2
        
        J_act_temp=front_cyl_act+top_cyl_act+bot_cyl_act
        J_act_temp=np.sqrt(J_act_temp/(3*sampling_timesteps))
        J_act_temp=J_act_temp*0.01/free_stream_vel
        J_act_temp=np.tanh(0.7*J_act_temp)
        
        J_act.append(J_act_temp)

    return J_act

def calculate_episode_var_mean(sens_data, CFD_timesteps_period,CFD_timesteps_actions, 
                               CFD_timestep_spacing, num_actions, sampling_periods):
    
    sampling_timesteps = np.int64(CFD_timesteps_period * sampling_periods / CFD_timestep_spacing)
    episode_var = []
    episode_mean=[]
    
    for i in range(num_actions):
        timestep_end=np.sum(CFD_timesteps_actions[:(i+1)])
        sampling_end=int(timestep_end / CFD_timestep_spacing)
        
        if sampling_end-sampling_timesteps<0:
            sens_var=np.var(sens_data[0:(sampling_end)])
            sens_mean=np.mean(sens_data[0:(sampling_end)])
        else: 
            sens_var=np.var(sens_data[(sampling_end-sampling_timesteps):(sampling_end)])
            sens_mean=np.mean(sens_data[(sampling_end-sampling_timesteps):(sampling_end)])
        
        episode_var.append(sens_var)
        episode_mean.append(sens_mean)

    return episode_var,episode_mean

def calculate_episode_var_mean_two(sens_data, CFD_timesteps_period,CFD_timesteps_actions, 
                                   CFD_timestep_spacing, num_actions,sampling_periods,max_sampling_period):
    
    sampling_timesteps = np.int64(CFD_timesteps_period * sampling_periods / CFD_timestep_spacing)
    min_sampling_timestep=np.int64(CFD_timesteps_period* max_sampling_period / CFD_timestep_spacing)
    episode_var = []
    episode_mean=[]
    
    for i in range(num_actions):
        timestep_end=np.sum(CFD_timesteps_actions[:(i+1)])
        sampling_end=int(timestep_end / CFD_timestep_spacing)
        
        if sampling_end>=min_sampling_timestep:
            sens_var=np.var(sens_data[(sampling_end-sampling_timesteps):(sampling_end)])
            sens_mean=np.mean(sens_data[(sampling_end-sampling_timesteps):(sampling_end)])
            episode_var.append(sens_var)
            episode_mean.append(sens_mean)

    return episode_var,episode_mean

def calculate_policy_average_STD(y_data_temp,num_iterations):
    y_data_mean = np.zeros(len(y_data_temp) // num_iterations)
    y_data_std = np.zeros(len(y_data_mean))
    for i in range(len(y_data_mean)):
        policy_average = np.mean(y_data_temp[i * num_iterations:(i * num_iterations + num_iterations)])
        policy_std = np.std(y_data_temp[i * num_iterations:(i * num_iterations + num_iterations)])
        y_data_mean[i] = policy_average
        y_data_std[i] = policy_std
    
    return y_data_mean,y_data_std

def plot_regular(y_data,x_label,y_label):
    plt.figure(figsize=(15,7.5))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    x_data=np.zeros(len(y_data))
    plt.plot(y_data)
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    
def plot_regular_two(x_data,y_data,x_label,y_label):
    plt.figure(figsize=(15,7.5))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.plot(x_data,y_data)
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)

def plot_policy_regular(y_data,x_label,y_label,num_iterations):
    x_data=np.zeros(len(y_data))
    for i in range(len(x_data)):
        x_data[i]=i//num_iterations+1
    plt.figure(figsize=(15,7.5))
    plt.xticks(np.arange(min(x_data), max(x_data)+1, 1), fontsize=15)
    plt.yticks(fontsize=15)
    plt.scatter(x_data,y_data)
    plt.xlabel(x_label,fontsize=20)
    plt.ylabel(y_label,fontsize=20)

def plot_policy_average(y_data_temp, x_label, y_label, num_iterations):
    y_data = np.zeros(len(y_data_temp) // num_iterations)
    y_data_std = np.zeros(len(y_data))
    x_data = np.zeros(len(y_data))
    for i in range(len(y_data)):
        policy_average = np.mean(y_data_temp[i * num_iterations:(i * num_iterations + num_iterations)])
        policy_std = np.std(y_data_temp[i * num_iterations:(i * num_iterations + num_iterations)])
        y_data[i] = policy_average
        y_data_std[i] = policy_std
        x_data[i] = i + 1

    plt.figure(figsize=(15, 7.5))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.scatter(x_data, y_data)
    plt.errorbar(x_data, y_data, yerr=y_data_std, linestyle="None", capsize=10)
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    plt.xticks(np.arange(min(x_data), max(x_data) + 1, 1))

def plot_episode_sens_var(top_sens_var,mid_sens_var,bot_sens_var,CFD_timesteps_actions,):
    
    x_data=np.zeros(len(top_sens_var))
    for i in range(len(x_data)):
            x_data[i]=np.sum(CFD_timesteps_actions[:(i+1)])
    
    plt.figure(figsize=(15,7.5))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.step(x_data,top_sens_var,label='Top Sensor')
    plt.step(x_data,mid_sens_var,label='Mid Sensor')
    plt.step(x_data,bot_sens_var,label='Bottom Sensor')
    plt.xlabel('CFD Timesteps',fontsize=20)
    plt.ylabel('Sensor Variance',fontsize=20)
    plt.legend(loc='upper right')

def plot_episode_sens_mean(top_sens_mean,mid_sens_mean,bot_sens_mean, CFD_timesteps_action):
    
    
    x_data=np.zeros(len(top_sens_var))
    for i in range(len(x_data)):
            x_data[i]=np.sum(CFD_timesteps_actions[:(i+1)])
    
    plt.figure(figsize=(15,7.5))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.step(x_data,top_sens_mean,label='Top Sensor')
    plt.step(x_data,mid_sens_mean,label='Mid Sensor')
    plt.step(x_data,bot_sens_mean,label='Bottom Sensor')
    plt.xlabel('CFD Timesteps',fontsize=20)
    plt.ylabel('Sensor Mean (m/s)',fontsize=20)
    plt.legend(loc='upper right')
    
def plot_episode_sensor_sampling(sens_data, CFD_timesteps_period, CFD_timesteps_actions,
                                 CFD_timesteps_spacing,num_actions,sampling_periods,y_label,):
    
    x_data=np.zeros(len(sens_data))
    for i in range(len(x_data)):
        x_data[i]=(i+1)*CFD_timestep_spacing
        
    plt.figure(figsize=(15,7.5))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('CFD Timesteps',fontsize=20)
    plt.ylabel(y_label,fontsize=20)
    plt.plot(x_data,sens_data)
    
    sampling_timesteps = np.int64(CFD_timesteps_period * sampling_periods / CFD_timestep_spacing)
    
    for i in range(num_actions):
        timestep_end=np.sum(CFD_timesteps_actions[:(i+1)])
        sampling_end=int(timestep_end / CFD_timestep_spacing)
        
        x_data_temp=x_data[(sampling_end-sampling_timesteps):(sampling_end)]
        y_data_temp=sens_data[(sampling_end-sampling_timesteps):(sampling_end)]
        sens_var=np.var(y_data_temp)
        sens_mean=np.mean(y_data_temp)
        if i%4==0:
            y_data_temp=np.array(y_data_temp)-1
            plt.plot(x_data_temp,y_data_temp,'red')
        elif i%4==1:
            y_data_temp=np.array(y_data_temp)-2
            plt.plot(x_data_temp,y_data_temp,'black')
        elif i%4==2:
            y_data_temp=np.array(y_data_temp)-3
            plt.plot(x_data_temp,y_data_temp,'orange')
        else:
            y_data_temp=np.array(y_data_temp)-4
            plt.plot(x_data_temp,y_data_temp,'green')


total_rewards = []
avg_rewards = []

total_J_flucs = []
avg_J_flucs = []

total_J_flucs_new = []
avg_J_flucs_new = []

total_J_acts = []
avg_J_acts = []

total_J_acts_new = []
avg_J_acts_new = []

total_J_tots_new=[]
avg_J_tots_new=[]

avg_STD = []
avg_values=[]

avg_STD_first_states=[]

CFD_timesteps_actions=[]

for i in range(len(master_data)):
    rewards = master_data[i]['rewards']
    top_sens_data = master_data[i]['top_sens_values']
    mid_sens_data = master_data[i]['mid_sens_values']
    bot_sens_data = master_data[i]['bot_sens_values']
    front_cyl_data = master_data[i]['front_cyl_RPS_PI']
    top_cyl_data = master_data[i]['top_cyl_RPS_PI']
    bot_cyl_data = master_data[i]['bot_cyl_RPS_PI']
    STD=master_data[i]['stds']
    value=master_data[i]['values']
    CFD_timesteps_action=master_data[i]['CFD_timesteps_actions']

    total_reward, avg_reward = calculate_reward(rewards)

    total_J_fluc,avg_J_fluc = calculate_J_fluc(top_sens_data, mid_sens_data, bot_sens_data, free_stream_vel, 
                                               CFD_timesteps_period,CFD_timesteps_action, CFD_timestep_spacing, 
                                               num_actions, sampling_periods)
    
    total_J_fluc_new,avg_J_fluc_new = calculate_J_fluc_new(top_sens_data, mid_sens_data, bot_sens_data, 
                                                           free_stream_vel, CFD_timesteps_period,CFD_timesteps_action, 
                                                           CFD_timestep_spacing, num_actions, sampling_periods)
                                              
    total_J_act,avg_J_act = calculate_J_act(front_cyl_data, top_cyl_data, bot_cyl_data, free_stream_vel, 
                                            CFD_timesteps_action, num_actions,CFD_timesteps_ramp)
    
    total_J_act_new,avg_J_act_new = calculate_J_act_new(front_cyl_data, top_cyl_data, bot_cyl_data, free_stream_vel, 
                                                        CFD_timesteps_action, num_actions,CFD_timesteps_ramp)

    total_rewards.append(total_reward)
    avg_rewards.append(avg_reward)
    
    total_J_flucs.append(total_J_fluc)
    avg_J_flucs.append(avg_J_fluc)
    
    total_J_flucs_new.append(total_J_fluc_new)
    avg_J_flucs_new.append(avg_J_fluc_new)
    
    total_J_acts.append(total_J_act)
    avg_J_acts.append(avg_J_act)
    
    total_J_acts_new.append(total_J_act_new)
    avg_J_acts_new.append(avg_J_act_new)
    
    avg_STD.append(np.mean(STD))
    avg_values.append(np.mean(value))
    
    avg_STD_first_state=np.mean(STD[0])
    avg_STD_first_states.append(avg_STD_first_state)
    
    CFD_timesteps_actions.append(CFD_timesteps_action)

total_J_tots_new = np.array(total_J_flucs_new) + np.array(total_J_acts_new)
avg_J_tots_new=np.array(avg_J_flucs_new) + np.array(avg_J_acts_new)


# PERFORMANCE CHECKS

#plot_regular(total_rewards,'Episode Number','Total Episode Rewards')
plot_regular(avg_rewards,'Episode Number','Average Episode Rewards')

#plot_regular(total_J_flucs,'Episode Number','Total J_Flucs')
plot_regular(avg_J_flucs,'Episode Number','Average $\mathcal{J}_{Fluc}$')
# plot_regular(avg_J_flucs_new,'Episode Number','Average J_Flucs New')

#plot_regular(total_J_acts,'Episode Number','Total J_Acts')
plot_regular(avg_J_acts,'Epsiode Number','Average $\mathcal{J}_{Act}$')
# plot_regular(avg_J_acts_new,'Epsiode Number','Average J_Acts_New')

#plot_regular(total_J_tots,'Episode Number','Total J_Total')
# plot_regular(avg_J_tots_new,'Episode Number','Average J_Total New')

#plot_policy_regular(total_rewards,'Policy Number','Total Episode Rewards',num_iterations)
#plot_policy_regular(total_J_flucs,'Policy Number','Total J_Flucs',num_iterations)
#plot_policy_regular(total_J_acts,'Policy Number','Total J_Acts',num_iterations)
#plot_policy_regular(total_J_tots,'Policy Number','Total J_Total',num_iterations)

# plot_policy_regular(avg_rewards,'Policy Number','Average Episode Rewards',num_iterations)
# plot_policy_regular(avg_J_flucs,'Policy Number','Average J_Flucs',num_iterations)
# plot_policy_regular(avg_J_acts,'Policy Number','Average J_Acts',num_iterations)
# plot_policy_regular(avg_J_flucs_new,'Policy Number','Average J_Flucs_New',num_iterations)
# plot_policy_regular(avg_J_acts_new,'Policy Number','Average J_Acts_New',num_iterations)
# plot_policy_regular(avg_J_tots_new,'Policy Number','Average J_Total New',num_iterations)

#plot_policy_average(total_rewards,'Policy Number','Policy Total Rewards',num_iterations)
#plot_policy_average(total_J_flucs,'Policy Number','Policy Total J_Flucs',num_iterations)
#plot_policy_average(total_J_acts,'Policy Number','Policy Total J_Acts',num_iterations)
#plot_policy_average(total_J_tots,'Policy Number','Policy Total J_Tots',num_iterations)

plot_policy_average(avg_rewards,'Policy Number','Policy Average Rewards',num_iterations)
plot_policy_average(avg_J_flucs,'Policy Number','Policy Average $\mathcal{J}_{Fluc}$',num_iterations)
plot_policy_average(avg_J_acts,'Policy Number','Policy Average $\mathcal{J}_{Act}$',num_iterations)
plot_policy_average(avg_J_flucs_new,'Policy Number','Policy Average $\mathcal{J}_{Fluc}$ Filtered',num_iterations)
plot_policy_average(avg_J_acts_new,'Policy Number','Policy Average $\mathcal{J}_{Act}$ Filtered',num_iterations)
# plot_policy_average(avg_J_tots_new,'Policy Number','Policy Average J_Tots_New',num_iterations)

#RL CONVERGENCE CHECKS

plot_regular(avg_STD,'Episode Number','Average Episode Standard Deviation')
# plot_policy_regular(avg_STD,'Policy Number','Average Episode STD',num_iterations)

avg_STD_policy,_=calculate_policy_average_STD(avg_STD,num_iterations)
# plot_regular(avg_STD_policy,'Policy Number','Average Policy STD')

STD_convergence=np.zeros(len(avg_STD_policy))
STD_convergence_start=6
x_data=np.zeros(len(avg_STD_policy))
for i in range(STD_convergence_start,len(x_data)):
    temp_A=abs(avg_STD_policy[i]-avg_STD_policy[i-1])
    temp_B=abs(avg_STD_policy[i]-avg_STD_policy[i-2])
    temp_C=abs(avg_STD_policy[i]-avg_STD_policy[i-3])
    temp_D=abs(avg_STD_policy[i]-avg_STD_policy[i-4])
    STD_convergence[i]=np.mean([temp_A, temp_B, temp_C, temp_D])
    x_data[i]=i
    
# plot_regular_two(x_data[STD_convergence_start:], STD_convergence[STD_convergence_start:],
#                  'Policy Number','Convergence Number')

# plot_regular(avg_values,'Episode Number','Average Value')
# plot_regular(critic_losses[1:],'Policy Number','Critic Losses')
# plot_regular(actor_losses[1:],'Policy Number','Actor Losses')

# Statistics and Archicture Comparison Metrics 

print('STATISTICS:')

index = np.argmin(avg_J_tots_new)
value=avg_J_tots_new[index]
print('Lowest J Total New: ', value, ' Episode: ',index+1)

index = np.argmin(avg_J_acts)
value=avg_J_acts[index]
print('Lowest J Act: ', value, ' Episode: ',index+1)

index = np.argmin(avg_J_acts_new)
value=avg_J_acts_new[index]
print('Lowest J Act New: ', value, ' Episode: ',index+1)

index = np.argmin(avg_J_flucs)
value=avg_J_flucs[index]
print('Lowest J Fluc: ', value, ' Episode: ',index+1)
                  
index = np.argmin(avg_J_flucs_new)
value=avg_J_flucs_new[index]
print('Lowest J Fluc New: ', value, ' Episode: ',index+1)

J_fluc_avg_policies=[]
J_act_avg_policies=[]
J_fluc_avg_new_policies=[]
J_act_avg_new_policies=[]
J_total_avg_new_policies=[]

J_fluc_min_policies=[]
J_act_min_policies=[]
J_fluc_min_new_policies=[]
J_act_min_new_policies=[]
J_total_min_new_policies=[]

timesteps_policies=[]
timesteps_total=[]

for i in range(num_policies):
    start_index=int(i*num_iterations)
    end_index=int((i+1)*num_iterations)
    
    J_fluc_avg=np.mean(avg_J_flucs[start_index:end_index])
    J_act_avg=np.mean(avg_J_acts[start_index:end_index])
    
    J_fluc_avg_new=np.mean(avg_J_flucs_new[start_index:end_index])
    J_act_avg_new=np.mean(avg_J_acts_new[start_index:end_index])
    J_total_avg_new=np.mean(avg_J_tots_new[start_index:end_index])
    
    J_fluc_min=np.amin(avg_J_flucs[start_index:end_index])
    J_act_min=np.amin(avg_J_acts[start_index:end_index])
    
    J_fluc_min_new=np.amin(avg_J_flucs_new[start_index:end_index])
    J_act_min_new=np.amin(avg_J_acts_new[start_index:end_index])
    J_total_min_new=np.amin(avg_J_tots_new[start_index:end_index])
    
    J_fluc_avg_policies.append(J_fluc_avg)
    J_act_avg_policies.append(J_act_avg)
    J_fluc_avg_new_policies.append(J_fluc_avg_new)
    J_act_avg_new_policies.append(J_act_avg_new)
    J_total_avg_new_policies.append(J_total_avg_new)

    J_fluc_min_policies.append(J_fluc_min)
    J_act_min_policies.append(J_act_min)
    J_fluc_min_new_policies.append(J_fluc_min_new)
    J_act_min_new_policies.append(J_act_min_new)
    J_total_min_new_policies.append(J_total_min_new)
    
    timesteps_policy=0
    for j in range(start_index,end_index):
        timesteps_policy += np.sum(master_data[j]['CFD_timesteps_actions'])
        
    timesteps_policies.append(timesteps_policy)
    timesteps_total.append(np.sum(timesteps_policies[:]))


# print("COMPARING WITH J UNACTUATED")
J_unact=0.077

data_1=(np.array(J_total_avg_new_policies)-J_unact)/(J_unact*np.array(timesteps_total)*CFD_timestep)
data_2=(np.array(J_total_min_new_policies)-J_unact)/(J_unact*np.array(timesteps_total)*CFD_timestep)

# plot_regular(data_1,'Policy','Data Normalized Average J Total New')
# plot_regular(data_2,'Policy','Data Normalized Minimum J Total New')


# print("COMPARING WITH FIRST POLICY")
data_3=(np.array(J_fluc_avg_policies)-J_fluc_avg_policies[0])/(np.array(timesteps_total)*CFD_timestep)
data_4=(np.array(J_fluc_min_policies)-J_fluc_min_policies[0])/(np.array(timesteps_total)*CFD_timestep)
data_5=(np.array(J_fluc_avg_new_policies)-J_fluc_avg_new_policies[0])/(np.array(timesteps_total)*CFD_timestep)
data_6=(np.array(J_fluc_min_new_policies)-J_fluc_min_new_policies[0])/(np.array(timesteps_total)*CFD_timestep)

data_7=(np.array(J_act_avg_policies)-J_act_avg_policies[0])/(np.array(timesteps_total)*CFD_timestep)
data_8=(np.array(J_act_min_policies)-J_act_min_policies[0])/(np.array(timesteps_total)*CFD_timestep)
data_9=(np.array(J_act_avg_new_policies)-J_act_avg_new_policies[0])/(np.array(timesteps_total)*CFD_timestep)
data_10=(np.array(J_act_min_new_policies)-J_act_min_new_policies[0])/(np.array(timesteps_total)*CFD_timestep)

data_11=(np.array(J_total_avg_new_policies)-J_total_avg_new_policies[0])/(np.array(timesteps_total)*CFD_timestep)
data_12=(np.array(J_total_min_new_policies)-J_total_min_new_policies[0])/(np.array(timesteps_total)*CFD_timestep)
         
# plot_regular(data_3,'Policy','Data Normalized Average J Fluc')
# plot_regular(data_4,'Policy','Data Normalized Minimum J Fluc')
# plot_regular(data_5,'Policy','Data Normalized Average J Fluc New')
# plot_regular(data_6,'Policy','Data Normalized Minimum J Fluc New')
         
# plot_regular(data_7,'Policy','Data Normalized Average J Act')
# plot_regular(data_8,'Policy','Data Normalized Minimum J Act')
# plot_regular(data_9,'Policy','Data Normalized Average J Act New')
# plot_regular(data_10,'Policy','Data Normalized Minimum J Act New')
         
# plot_regular(data_11,'Policy','Data Normalized Average J Total new')
# plot_regular(data_12,'Policy','Data Normalized Minimum J Total New')

# EPISODE CHECKS

episode_start=119
episode_end=120

print('\nSTATISTICS: EPISODE ' + str(episode_start))

index = np.argmin(avg_J_tots_new)
value=avg_J_tots_new[episode_start - 1]
print('J Total New: ', value)

index = np.argmin(avg_J_acts)
value=avg_J_acts[episode_start - 1]
print('J Act: ', value)

index = np.argmin(avg_J_acts_new)
value=avg_J_acts_new[episode_start - 1]
print('J Act New: ', value)

index = np.argmin(avg_J_flucs)
value=avg_J_flucs[episode_start - 1]
print('J Fluc: ', value)
                  
index = np.argmin(avg_J_flucs_new)
value=avg_J_flucs_new[episode_start - 1]
print('J Fluc New: ', value)

for j in range(episode_start,episode_end):
    episode=j
    front_cyl_data=master_data[episode-1]['front_cyl_RPS_PI']
    top_cyl_data=master_data[episode-1]['top_cyl_RPS_PI']
    bot_cyl_data=master_data[episode-1]['bot_cyl_RPS_PI']

    top_sens_data=master_data[episode-1]['top_sens_values']
    mid_sens_data=master_data[episode-1]['mid_sens_values']
    bot_sens_data=master_data[episode-1]['bot_sens_values']
    CFD_timesteps_action=master_data[episode-1]['CFD_timesteps_actions']

    episode_stds_temp=master_data[episode-1]['stds']
    
    episode_stds=[]
    for i in range(len(episode_stds_temp)):
        episode_stds.append(np.mean(episode_stds_temp[i]))

    top_sens_var, top_sens_mean=calculate_episode_var_mean(top_sens_data,CFD_timesteps_period,CFD_timesteps_action,
                                                           CFD_timestep_spacing,num_actions,
                                                           sampling_periods)

    mid_sens_var, mid_sens_mean=calculate_episode_var_mean(mid_sens_data,CFD_timesteps_period,CFD_timesteps_action,
                                                            CFD_timestep_spacing,num_actions,
                                                           sampling_periods)

    bot_sens_var, bot_sens_mean=calculate_episode_var_mean(bot_sens_data,CFD_timesteps_period,CFD_timesteps_action,
                                                           CFD_timestep_spacing,num_actions,
                                                           sampling_periods)

    # fig, axs = plt.subplots(2,figsize=(20,10))
    # title='Episode ' + str(j)
    # fig.suptitle(title)
    x2=np.zeros(len(top_sens_data))
    for l in range(len(x2)):
        x2[l]=(l+1)*CFD_timestep_spacing
    
    #Plotting cylinder data 
    # axs[0].plot(front_cyl_data, label='Front Cylinder')
    # axs[0].plot(top_cyl_data, label='Top Cylinder')
    # axs[0].plot(bot_cyl_data, label='Bottom Cylinder')
    # axs[0].set_xlabel('CFD Timestep')
    # axs[0].set_ylabel('Motor Rotation Rate (rad/s)')
    # axs[0].legend(loc='upper right')
    
    #Plotting sensor data
    # axs[1].plot(x2, top_sens_data, label='Top Sensor')
    # axs[1].plot(x2, mid_sens_data, label='Mid Sensor')
    # axs[1].plot(x2, bot_sens_data, label='Bottom Sensor')
    # axs[1].set_xlabel('CFD Timestep')
    # axs[1].set_ylabel('Sensor Data (m/s)')
    # axs[1].legend(loc='upper right')

    # plot_episode_sensor_sampling(top_sens_data, CFD_timesteps_period, CFD_timesteps_action,
    #                              CFD_timestep_spacing,num_actions,
    #                              1.5,'Top Sensor Data')
    

    # plot_episode_sensor_sampling(mid_sens_data, CFD_timesteps_period, CFD_timesteps_action,
    #                              CFD_timestep_spacing,num_actions,
    #                              1.5,'Mid Sensor Data')

    # plot_episode_sensor_sampling(bot_sens_data, CFD_timesteps_period, CFD_timesteps_action,
    #                              CFD_timestep_spacing,num_actions,
    #                              1.5,'Bot Sensor Data')


    # plot_episode_sens_var(top_sens_var,mid_sens_var,bot_sens_var,CFD_timesteps_action)
    # plot_episode_sens_mean(top_sens_mean,mid_sens_mean,bot_sens_mean,CFD_timesteps_action)
    # plot_regular(episode_stds,'Action Number','STD')
    

actions_state_1=[]
actions_state_2=[]
actions_state_3=[]
actions_state_4=[]
actions_state_5=[]
actions_state_6=[]

actions_rewards=[]

for i in range(len(master_data)):
    for j in range(len(master_data[i]['states'])):
        actions_state_1.append(master_data[i]['states'][j][0])
        actions_state_2.append(master_data[i]['states'][j][1])
        actions_state_3.append(master_data[i]['states'][j][2])
        actions_state_4.append(master_data[i]['states'][j][3])
        actions_state_5.append(master_data[i]['states'][j][4])
        actions_state_6.append(master_data[i]['states'][j][5])

        
        actions_rewards.append(master_data[i]['rewards'][j])
        
# plot_regular(np.array(actions_state_1),'Actions','Top Sensor State')
# plot_regular(np.array(actions_state_2),'Actions','Mid Sensor State')
# plot_regular(np.array(actions_state_3),'Actions','Bot Sensor State')

# plot_regular((np.array(actions_state_1)+1.0)/5.0,'Actions','Top Sensor Variance')
# plot_regular((np.array(actions_state_2)+1.0)/5.0,'Actions','Mid Sensor Variance')
# plot_regular((np.array(actions_state_3)+1.0)/5.0,'Actions','Bot Sensor Variance')

# plot_regular(actions_state_4,'Actions','Front Cyl Offset')
# plot_regular(actions_state_5,'Actions','Top Cyl Offset')
# plot_regular(actions_state_6,'Actions','Bottom Cyl Offset')


# plot_regular(actions_rewards,'Actions','Rewards')

#Checking the state input and the variance calculations are the same/similar

actions_variance_top_sens=[]
actions_variance_mid_sens=[]
actions_variance_bot_sens=[]

sampling_periods=0.90

for i in range(len(master_data)):
    top_sens_data=master_data[i]['top_sens_values']
    mid_sens_data=master_data[i]['mid_sens_values']
    bot_sens_data=master_data[i]['bot_sens_values']
    CFD_timesteps_action=master_data[i]['CFD_timesteps_actions']

    top_sens_var, top_sens_mean=calculate_episode_var_mean(top_sens_data,CFD_timesteps_period,CFD_timesteps_action,
                                                           CFD_timestep_spacing,num_actions,
                                                           sampling_periods)
                                                        
    mid_sens_var, mid_sens_mean=calculate_episode_var_mean(mid_sens_data,CFD_timesteps_period,CFD_timesteps_action,
                                                           CFD_timestep_spacing,num_actions,
                                                           sampling_periods)
                                                     
    bot_sens_var, bot_sens_mean=calculate_episode_var_mean(bot_sens_data,CFD_timesteps_period,CFD_timesteps_action,
                                                           CFD_timestep_spacing,num_actions,
                                                           sampling_periods)
    actions_variance_top_sens.extend(top_sens_var)
    actions_variance_mid_sens.extend(mid_sens_var)
    actions_variance_bot_sens.extend(bot_sens_var)

top_sens_subtracted=(np.array(actions_state_1)+1.0)/5.0-np.array(actions_variance_top_sens)
mid_sens_subtracted=(np.array(actions_state_2)+1.0)/5.0-np.array(actions_variance_mid_sens)
bot_sens_subtracted=(np.array(actions_state_3)+1.0)/5.0-np.array(actions_variance_bot_sens)
# print('Top Sensor Difference', np.mean(top_sens_subtracted))
# print('Top Sensor Difference',np.mean(mid_sens_subtracted))
# print('Top Sensor Difference',np.mean(bot_sens_subtracted))

J_flucs_actions = []
J_acts_actions=[]

J_flucs_new_actions = []
J_acts_new_actions=[]
J_tots_new_actions = []

sampling_periods=1.5

for i in range(len(master_data)):
    top_sens_data = master_data[i]['top_sens_values']
    mid_sens_data = master_data[i]['mid_sens_values']
    bot_sens_data = master_data[i]['bot_sens_values']
    front_cyl_data = master_data[i]['front_cyl_RPS_PI']
    top_cyl_data = master_data[i]['top_cyl_RPS_PI']
    bot_cyl_data = master_data[i]['bot_cyl_RPS_PI']
    CFD_timesteps_action=master_data[i]['CFD_timesteps_actions']

    J_fluc_action= calculate_J_fluc_actions(top_sens_data, mid_sens_data, bot_sens_data, free_stream_vel, 
                                            CFD_timesteps_period,CFD_timesteps_action,
                                            CFD_timestep_spacing,num_actions,sampling_periods)
                                              
    J_act_action = calculate_J_act_actions(front_cyl_data, top_cyl_data, bot_cyl_data, free_stream_vel, 
                                           CFD_timesteps_action, num_actions,CFD_timesteps_ramp)

    
    J_fluc_new_action=calculate_J_fluc_new_actions(top_sens_data, mid_sens_data, bot_sens_data, free_stream_vel, 
                                                   CFD_timesteps_period,CFD_timesteps_action, 
                                                   CFD_timestep_spacing,num_actions,sampling_periods)
    
    J_act_new_action = calculate_J_act_new_actions(front_cyl_data, top_cyl_data, bot_cyl_data, free_stream_vel, 
                                                   CFD_timesteps_action, num_actions,CFD_timesteps_ramp)

    
    J_flucs_actions.extend(J_fluc_action)
    J_acts_actions.extend(J_act_action)
    
    J_flucs_new_actions.extend(J_fluc_new_action)
    J_acts_new_actions.extend(J_act_new_action)
    
    
J_tots_new_actions=np.array(J_flucs_new_actions)+np.array(J_acts_new_actions)

policy_start=1
policy_end=12
index_start=policy_start*num_iterations*num_actions
index_end=policy_end*num_iterations*num_actions


max_cost=np.amax(np.array(J_flucs_actions))
incret=0.005
x_data_J_flucs=np.zeros(int(max_cost/incret))
y_data_J_flucs=np.zeros(len(x_data_J_flucs))

for i in range(1,len(x_data_J_flucs)):
    x_data_J_flucs[i]=incret*i
    for j in range(index_start,index_end):
        if J_flucs_actions[j]>x_data_J_flucs[i-1] and J_flucs_actions[j]<=x_data_J_flucs[i]:
            y_data_J_flucs[i]+=1
            
            
max_cost=np.amax(np.array(J_flucs_new_actions))
incret=0.005
x_data_J_flucs_new=np.zeros(int(max_cost/incret))
y_data_J_flucs_new=np.zeros(len(x_data_J_flucs_new))

for i in range(1,len(x_data_J_flucs_new)):
    x_data_J_flucs_new[i]=incret*i
    for j in range(index_start,index_end):
        if J_flucs_new_actions[j]>x_data_J_flucs_new[i-1] and J_flucs_new_actions[j]<=x_data_J_flucs_new[i]:
            y_data_J_flucs_new[i]+=1
            
            
max_cost=np.amax(np.array(J_acts_actions))
incret=0.05
x_data_J_acts=np.zeros(int(max_cost/incret))
y_data_J_acts=np.zeros(len(x_data_J_acts))

for i in range(1,len(x_data_J_acts)):
    x_data_J_acts[i]=incret*i
    for j in range(index_start,index_end):
        if J_acts_actions[j]>x_data_J_acts[i-1] and J_acts_actions[j]<=x_data_J_acts[i]:
            y_data_J_acts[i]+=1
            
max_cost=np.amax(np.array(J_acts_new_actions))
incret=0.05
x_data_J_acts_new=np.zeros(int(max_cost/incret))
y_data_J_acts_new=np.zeros(len(x_data_J_acts_new))

for i in range(1,len(x_data_J_acts_new)):
    x_data_J_acts_new[i]=incret*i
    for j in range(index_start,index_end):
        if J_acts_new_actions[j]>x_data_J_acts_new[i-1] and J_acts_new_actions[j]<=x_data_J_acts_new[i]:
            y_data_J_acts_new[i]+=1

max_cost=np.amax(np.array(J_tots_new_actions))
incret=0.05
x_data_J_tots_new=np.zeros(int(max_cost/incret))
y_data_J_tots_new=np.zeros(len(x_data_J_tots_new))

for i in range(1,len(x_data_J_tots_new)):
    x_data_J_tots_new[i]=incret*i
    for j in range(index_start,index_end):
        if J_tots_new_actions[j]>x_data_J_tots_new[i-1] and J_tots_new_actions[j]<=x_data_J_tots_new[i]:
            y_data_J_tots_new[i]+=1
           
            

# plot_regular_two(x_data_J_flucs,y_data_J_flucs,'$\mathcal{J}_{Fluc}$ Cost','Frequency (Actions)')
# plot_regular_two(x_data_J_acts,y_data_J_acts,'J Acts Cost','Frequency (Actions)')

# plot_regular_two(x_data_J_flucs_new,y_data_J_flucs_new,'J Flucs Cost New','Frequency (Actions)')
# plot_regular_two(x_data_J_acts_new,y_data_J_acts_new,'J Acts Cost New','Frequency (Actions)')

# plot_regular_two(x_data_J_tots_new,y_data_J_tots_new,'J Total Cost New','Frequency (Actions)')

if Json_files==True:
    for i in range(len(master_data)):
        json_dict = {'iteration_ID': 0, 'motor_data': {'front': [], 'top': [], 'bot': []},
                    'sensor_data': {'top': [], 'mid': [], 'bot': []},
                    'costs': {'J_fluc': 0, 'J_act': 0, 'J_tot': 0},
                    'total_rewards': 0}
        json_dict['iteration_ID'] = master_data[i]['iteration_ID']
        json_dict['motor_data']['front'].extend(master_data[i]['front_cyl_RPS_PI'])
        json_dict['motor_data']['top'].extend(master_data[i]['top_cyl_RPS_PI'])
        json_dict['motor_data']['bot'].extend(master_data[i]['bot_cyl_RPS_PI'])
        json_dict['sensor_data']['top'].extend(master_data[i]['top_sens_values'])
        json_dict['sensor_data']['mid'].extend(master_data[i]['mid_sens_values'])
        json_dict['sensor_data']['bot'].extend(master_data[i]['bot_sens_values'])
        json_dict['costs']['J_fluc'] = avg_J_flucs_new[i]
        json_dict['costs']['J_act'] = avg_J_acts_new[i]
        json_dict['costs']['J_tot'] = avg_J_tots_new[i]
        json_dict['average_reward'] = avg_rewards[i]

        iteration = master_data[i]['iteration_ID']
        filename = '../../Production Runs/Production Run 5/json_files/data_iteration_' + str(iteration) + '.json'
        with open(filename, 'w') as outfile:
            json.dump(json_dict, outfile)

plt.show()