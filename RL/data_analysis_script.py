import pickle
import numpy as np
import matplotlib.pyplot as plt
import json

num_iterations=10
num_policies=14
CFD_timestep=5e-4
CFD_timestep_spacing=5
num_actions=15
dur_actions=1.004198
dur_action_one=1.50838
shedding_freq=8.42
free_stream_vel=1.5
sampling_periods=0.9
gamma_act=0.009
Json_files = False

shedding_period=1/shedding_freq
CFD_timesteps_period=shedding_period/CFD_timestep
CFD_timesteps_action=CFD_timesteps_period*dur_actions
CFD_timesteps_action_one=CFD_timesteps_period*dur_action_one

master_data=[]
actor_losses=[]
critic_losses=[]


for policy in range(num_policies):
    for iteration in range(num_iterations):
        iteration_ID=policy*num_iterations+iteration+1
        filename='../../Production Runs/Production Run 3/pickle_files/data_iteration_' + str(iteration_ID)+'.pickle'
        with open(filename, 'rb') as handle:
            data = pickle.load(handle)
        master_data.append(data)
    filename='../../Production Runs/Production Run 3/actor_critic_losses/actor_critic_losses_'+str(policy+1)+'.pickle'
    with open(filename,'rb') as handle:
        data=pickle.load(handle)
    actor_losses.append(data['actor_losses'][-1])
    critic_losses.append(data['critic_losses'][-1])


def calculate_reward(rewards_data):
    total_rewards = np.sum(rewards_data)
    avg_rewards = np.mean(rewards_data)

    return total_rewards, avg_rewards


def calculate_J_fluc(top_sens_data, mid_sens_data, bot_sens_data, free_stream_vel, CFD_timesteps_period,
                     CFD_timesteps_action, CFD_timesteps_action_one, CFD_timestep_spacing, num_actions,
                     sampling_periods, ):
    sampling_timesteps = int(CFD_timesteps_period * sampling_periods / CFD_timestep_spacing)
    J_fluc = []

    for i in range(num_actions):
        if i == 0:
            sampling_end = int(CFD_timesteps_action_one / CFD_timestep_spacing)
        else:
            sampling_end = int((CFD_timesteps_action_one + i * CFD_timesteps_action) / CFD_timestep_spacing)

        top_sens_var = np.var(top_sens_data[(sampling_end - sampling_timesteps):sampling_end])
        mid_sens_var = np.var(mid_sens_data[(sampling_end - sampling_timesteps):sampling_end])
        bot_sens_var = np.var(bot_sens_data[(sampling_end - sampling_timesteps):sampling_end])

        J_fluc_temp = np.mean([top_sens_var, mid_sens_var, bot_sens_var])
        J_fluc_temp = J_fluc_temp / free_stream_vel ** 2

        J_fluc.append(J_fluc_temp)

    total_J_fluc = np.sum(J_fluc)
    avg_J_fluc = np.mean(J_fluc)

    return total_J_fluc, avg_J_fluc


def calculate_J_act(front_cyl_data, top_cyl_data, bot_cyl_data, free_stream_vel, CFD_timesteps_period,
                    CFD_timesteps_action, CFD_timesteps_action_one, CFD_timesteps_spacing, num_actions,
                    sampling_periods):
    sampling_timesteps = int(CFD_timesteps_period * sampling_periods)
    J_act = []

    for i in range(num_actions):
        if i == 0:
            sampling_end = int(CFD_timesteps_action_one)
        else:
            sampling_end = int(CFD_timesteps_action_one + i * CFD_timesteps_action)

        front_cyl_act = 0
        top_cyl_act = 0
        bot_cyl_act = 0

        for j in range(sampling_end - sampling_timesteps, sampling_end):
            front_cyl_act += front_cyl_data[j - 1] ** 2
            top_cyl_act += top_cyl_data[j - 1] ** 2
            bot_cyl_act += bot_cyl_data[j - 1] ** 2

        J_act_temp = front_cyl_act + top_cyl_act + bot_cyl_act
        J_act_temp = np.sqrt(J_act_temp / (3 * sampling_timesteps))
        J_act_temp = J_act_temp * 0.01 / free_stream_vel

        J_act.append(J_act_temp)

    total_J_act = np.sum(J_act)
    avg_J_act = np.mean(J_act)

    return total_J_act, avg_J_act


def calculate_episode_var_mean(sens_data, CFD_timesteps_period, CFD_timesteps_action, CFD_timesteps_action_one,
                               CFD_timestep_spacing, num_actions, sampling_periods, ):
    sampling_timesteps = np.int64(CFD_timesteps_period * sampling_periods / CFD_timestep_spacing)
    episode_var = []
    episode_mean = []

    for i in range(num_actions):
        if i == 0:
            sampling_end = np.int64(np.ceil(CFD_timesteps_action_one / CFD_timestep_spacing))
        else:
            sampling_end = np.int64(
                np.ceil((CFD_timesteps_action_one + i * CFD_timesteps_action) / CFD_timestep_spacing))

        if sampling_end - sampling_timesteps < 0:
            sens_var = np.var(sens_data[0:(sampling_end)])
            sens_mean = np.mean(sens_data[0:(sampling_end)])
        else:
            sens_var = np.var(sens_data[(sampling_end - sampling_timesteps):(sampling_end)])
            sens_mean = np.mean(sens_data[(sampling_end - sampling_timesteps):(sampling_end)])

        episode_var.append(sens_var)
        episode_mean.append(sens_mean)

    return episode_var, episode_mean


def calculate_episode_var_mean_two(sens_data, CFD_timesteps_period, CFD_timesteps_action, CFD_timesteps_action_one,
                                   CFD_timestep_spacing, num_actions, sampling_periods, max_sampling_period):
    sampling_timesteps = np.int64(CFD_timesteps_period * sampling_periods / CFD_timestep_spacing)
    min_sampling_timestep = np.int64(CFD_timesteps_period * max_sampling_period / CFD_timestep_spacing)
    episode_var = []
    episode_mean = []

    for i in range(num_actions):
        if i == 0:
            sampling_end = np.int64(np.ceil(CFD_timesteps_action_one / CFD_timestep_spacing))
        else:
            sampling_end = np.int64(
                np.ceil((CFD_timesteps_action_one + i * CFD_timesteps_action) / CFD_timestep_spacing))

        if sampling_end >= min_sampling_timestep:
            sens_var = np.var(sens_data[(sampling_end - sampling_timesteps):(sampling_end)])
            sens_mean = np.mean(sens_data[(sampling_end - sampling_timesteps):(sampling_end)])
            episode_var.append(sens_var)
            episode_mean.append(sens_mean)

    return episode_var, episode_mean


def plot_regular(y_data, x_label, y_label):
    plt.figure(figsize=(15, 7.5))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    x_data = np.zeros(len(y_data))
    plt.plot(y_data)
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)


def plot_regular_two(x_data, y_data, x_label, y_label):
    plt.figure(figsize=(15, 7.5))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.plot(x_data, y_data)
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)


def plot_policy_regular(y_data, x_label, y_label, num_iterations):
    x_data = np.zeros(len(y_data))
    for i in range(len(x_data)):
        x_data[i] = i // num_iterations + 1
    plt.figure(figsize=(15, 7.5))
    plt.xticks(np.arange(min(x_data), max(x_data) + 1, 1), fontsize=15)
    plt.yticks(fontsize=15)
    plt.scatter(x_data, y_data)
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)


def plot_moving_average_regular(y_data_temp, x_label, y_label, avg_sample):
    x_data = np.zeros(len(y_data_temp) - avg_sample + 1)
    y_data = np.zeros(len(x_data))
    y_data_upper = np.zeros(len(x_data))
    y_data_lower = np.zeros(len(x_data))
    for i in range(len(x_data)):
        x_data[i] = avg_sample + i
        y_data[i] = np.mean(y_data_temp[i:avg_sample + i])
        std = np.std(y_data_temp[i:avg_sample + i])
        y_data_upper[i] = y_data[i] + std
        y_data_lower[i] = y_data[i] - std

    plt.figure(figsize=(15, 7.5))
    plt.xticks(np.arange(min(x_data), max(x_data) + 1), fontsize=15)
    plt.yticks(fontsize=20)
    plt.plot(x_data, y_data, label='Moving Average')
    plt.plot(x_data, y_data_upper, label='Moving Average + STD')
    plt.plot(x_data, y_data_lower, label='Moving Average - STD')
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    plt.legend(loc='upper right')


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


def plot_episode_cyl_data(front_cyl_data, top_cyl_data, bot_cyl_data):
    plt.figure(figsize=(15, 7.5))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.plot(front_cyl_data, label='Front Cylinder')
    plt.plot(top_cyl_data, label='Top Cylinder')
    plt.plot(bot_cyl_data, label='Bottom Cylinder')
    plt.xlabel('CFD Timesteps', fontsize=20)
    plt.ylabel('Cylinder Rotation Rate (rad/s)', fontsize=20)
    plt.legend(loc='upper right')


def plot_episode_sens_data(top_sens_data, mid_sens_data, bot_sens_data, CFD_timestep_spacing):
    x_data = np.zeros(len(top_sens_data))
    for i in range(len(x_data)):
        x_data[i] = (i + 1) * CFD_timestep_spacing

    plt.figure(figsize=(15, 7.5))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.plot(x_data, top_sens_data, label='Top Sensor')
    plt.plot(x_data, mid_sens_data, label='Mid Sensor')
    plt.plot(x_data, bot_sens_data, label='Bottom Sensor')
    plt.xlabel('CFD Timesteps', fontsize=20)
    plt.ylabel('Sensor Speed (m/s)', fontsize=20)
    plt.legend(loc='upper right')


def plot_episode_sens_var(top_sens_var, mid_sens_var, bot_sens_var, CFD_timesteps_action_one,
                          CFD_timesteps_action):
    x_data = np.zeros(len(top_sens_var))
    for i in range(len(x_data)):
        if i == 0:
            x_data[i] = CFD_timesteps_action_one
        else:
            x_data[i] = CFD_timesteps_action_one + i * CFD_timesteps_action

    plt.figure(figsize=(15, 7.5))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.step(x_data, top_sens_var, label='Top Sensor')
    plt.step(x_data, mid_sens_var, label='Mid Sensor')
    plt.step(x_data, bot_sens_var, label='Bottom Sensor')
    plt.xlabel('CFD Timesteps', fontsize=20)
    plt.ylabel('Sensor Variance', fontsize=20)
    plt.legend(loc='upper right')


def plot_episode_sens_mean(top_sens_mean, mid_sens_mean, bot_sens_mean, CFD_timesteps_action_one,
                           CFD_timesteps_action):
    x_data = np.zeros(len(top_sens_var))
    for i in range(len(x_data)):
        if i == 0:
            x_data[i] = CFD_timesteps_action_one
        else:
            x_data[i] = CFD_timesteps_action_one + i * CFD_timesteps_action

    plt.figure(figsize=(15, 7.5))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.step(x_data, top_sens_mean, label='Top Sensor')
    plt.step(x_data, mid_sens_mean, label='Mid Sensor')
    plt.step(x_data, bot_sens_mean, label='Bottom Sensor')
    plt.xlabel('CFD Timesteps', fontsize=20)
    plt.ylabel('Sensor Mean (m/s)', fontsize=20)
    plt.legend(loc='upper right')


def plot_episode_sensor_sampling(sens_data, CFD_timesteps_period, CFD_timesteps_action,
                                 CFD_timesteps_action_one, CFD_timesteps_spacing, num_actions,
                                 sampling_periods, y_label, ):
    x_data = np.zeros(len(sens_data))
    for i in range(len(x_data)):
        x_data[i] = (i + 1) * CFD_timestep_spacing

    plt.figure(figsize=(15, 7.5))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('CFD Timesteps', fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    plt.plot(x_data, sens_data)

    sampling_timesteps = np.int64(CFD_timesteps_period * sampling_periods / CFD_timestep_spacing)

    for i in range(num_actions):
        if i == 0:
            sampling_end = np.int64(np.ceil(CFD_timesteps_action_one / CFD_timestep_spacing))
        else:
            sampling_end = np.int64(
                np.ceil((CFD_timesteps_action_one + i * CFD_timesteps_action) / CFD_timestep_spacing))

        x_data_temp = x_data[(sampling_end - sampling_timesteps):(sampling_end)]
        y_data_temp = sens_data[(sampling_end - sampling_timesteps):(sampling_end)]
        sens_var = np.var(y_data_temp)
        sens_mean = np.mean(y_data_temp)
        if i % 4 == 0:
            y_data_temp = np.array(y_data_temp) - 1
            plt.plot(x_data_temp, y_data_temp, 'red')
        elif i % 4 == 1:
            y_data_temp = np.array(y_data_temp) - 2
            plt.plot(x_data_temp, y_data_temp, 'black')
        elif i % 4 == 2:
            y_data_temp = np.array(y_data_temp) - 3
            plt.plot(x_data_temp, y_data_temp, 'orange')
        else:
            y_data_temp = np.array(y_data_temp) - 4
            plt.plot(x_data_temp, y_data_temp, 'green')


def plot_std_reward_values(std_data, reward_data, values_data, std_scale, reward_scale, values_scale):
    plt.figure(figsize=(15, 7.5))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    x_data = np.zeros(len(std_data))
    for i in range(len(std_data)):
        x_data[i] = i + 1
    std_data = np.array(std_data) / std_scale
    reward_data = np.array(reward_data) / std_scale
    values_data = np.array(values_data) / values_scale
    plt.plot(x_data, std_data, label='Average STD', color='red')
    plt.xlabel('Episode', fontsize=20)
    plt.plot(x_data, reward_data, label='Average Reward')
    plt.plot(x_data, values_data, label='Average Values')
    plt.legend(prop={'size': 15})


total_rewards = []
avg_rewards = []

total_J_flucs = []
avg_J_flucs = []

total_J_acts = []
avg_J_acts = []

total_J_tots = []
avg_J_tots = []

avg_STD = []
avg_values = []

avg_STD_first_states = []

for i in range(len(master_data)):
    rewards = master_data[i]['rewards']
    top_sens_data = master_data[i]['top_sens_values']
    mid_sens_data = master_data[i]['mid_sens_values']
    bot_sens_data = master_data[i]['bot_sens_values']
    front_cyl_data = master_data[i]['front_cyl_RPS_PI']
    top_cyl_data = master_data[i]['top_cyl_RPS_PI']
    bot_cyl_data = master_data[i]['bot_cyl_RPS_PI']
    STD = master_data[i]['stds']
    value = master_data[i]['values']

    total_reward, avg_reward = calculate_reward(rewards)

    total_J_fluc, avg_J_fluc = calculate_J_fluc(top_sens_data, mid_sens_data, bot_sens_data, free_stream_vel,
                                                CFD_timesteps_period, CFD_timesteps_action, CFD_timesteps_action_one,
                                                CFD_timestep_spacing, num_actions, sampling_periods)

    total_J_act, avg_J_act = calculate_J_act(front_cyl_data, top_cyl_data, top_cyl_data, free_stream_vel,
                                             CFD_timesteps_period, CFD_timesteps_action, CFD_timesteps_action_one,
                                             CFD_timestep_spacing, num_actions, sampling_periods)

    total_rewards.append(total_reward)
    avg_rewards.append(avg_reward)

    total_J_flucs.append(total_J_fluc)
    avg_J_flucs.append(avg_J_fluc)

    total_J_acts.append(total_J_act)
    avg_J_acts.append(avg_J_act)

    avg_STD.append(np.mean(STD))
    avg_values.append(np.mean(value))

    avg_STD_first_state = np.mean(STD[0])
    avg_STD_first_states.append(avg_STD_first_state)

total_J_tots = np.array(total_J_flucs) + gamma_act * np.array(total_J_acts)
avg_J_tots = np.array(avg_J_flucs) + gamma_act * np.array(avg_J_acts)

# PERFORMANCE CHECKS

# plot_regular(total_rewards,'Episode Number','Total Episode Rewards')
plot_regular(avg_rewards, 'Episode Number', 'Average Episode Rewards')

# plot_regular(total_J_flucs,'Episode Number','Total J_Flucs')
plot_regular(avg_J_flucs, 'Episode Number', 'Average J_Flucs')

# plot_regular(total_J_acts,'Episode Number','Total J_Acts')
plot_regular(avg_J_acts, 'Epsiode Number', 'Average J_Acts')

# plot_regular(total_J_tots,'Episode Number','Total J_Total')
# plot_regular(avg_J_tots, 'Episode Number', 'Average J_Total')

# plot_policy_regular(total_rewards,'Policy Number','Total Episode Rewards',num_iterations)
# plot_policy_regular(total_J_flucs,'Policy Number','Total J_Flucs',num_iterations)
# plot_policy_regular(total_J_acts,'Policy Number','Total J_Acts',num_iterations)
# plot_policy_regular(total_J_tots,'Policy Number','Total J_Total',num_iterations)

# plot_policy_regular(avg_rewards, 'Policy Number', 'Average Episode Rewards', num_iterations)
# plot_policy_regular(avg_J_flucs, 'Policy Number', 'Average J_Flucs', num_iterations)
# plot_policy_regular(avg_J_acts, 'Policy Number', 'Average J_Acts', num_iterations)
# plot_policy_regular(avg_J_tots, 'Policy Number', 'Average J_Total', num_iterations)

# plot_policy_average(total_rewards,'Policy Number','Policy Total Rewards',num_iterations)
# plot_policy_average(total_J_flucs,'Policy Number','Policy Total J_Flucs',num_iterations)
# plot_policy_average(total_J_acts,'Policy Number','Policy Total J_Acts',num_iterations)
# plot_policy_average(total_J_tots,'Policy Number','Policy Total J_Tots',num_iterations)

plot_policy_average(avg_rewards, 'Policy Number', 'Policy Average Rewards', num_iterations)
plot_policy_average(avg_J_flucs, 'Policy Number', 'Policy Average J_Flucs', num_iterations)
plot_policy_average(avg_J_acts, 'Policy Number', 'Policy Average J_Acts', num_iterations)
# plot_policy_average(avg_J_tots, 'Policy Number', 'Policy Average J_Tots', num_iterations)

# RL CONVERGENCE CHECKS

# plot_regular(avg_STD, 'Episode Number', 'Average STD')
# plot_regular(avg_values, 'Episode Number', 'Average Value')
# plot_regular(critic_losses[1:], 'Policy Number', 'Critic Losses')
# plot_regular(actor_losses[1:], 'Policy Number', 'Actor Losses')

# RL CONVERGENCE CHECKS 2

# plot_std_reward_values(avg_STD, avg_rewards, avg_values, 1, 1, 8)

print('STATS')

index = np.argmin(avg_J_tots)
value = avg_J_tots[index]
print('Lowest J Total: ', value, ' Episode: ', index + 1)

index = np.argmin(avg_J_acts)
value = avg_J_acts[index]
print('Lowest J Act: ', value, ' Episode: ', index + 1)

index = np.argmin(avg_J_flucs)
value = avg_J_flucs[index]
print('Lowest J Fluc: ', value, ' Episode: ', index + 1)

# EPISODE CHECKS

episode = 136

print('\nSTATS')

print('Episode: ', episode)
value = avg_J_tots[episode - 1]
print('J Total: ', value)
value = avg_J_acts[episode - 1]
print('J Act: ', value)
value = avg_J_flucs[episode - 1]
print('J Fluc: ', value, '\n')

front_cyl_data = master_data[episode - 1]['front_cyl_RPS_PI']
top_cyl_data = master_data[episode - 1]['top_cyl_RPS_PI']
bot_cyl_data = master_data[episode - 1]['bot_cyl_RPS_PI']

top_sens_data = master_data[episode - 1]['top_sens_values']
mid_sens_data = master_data[episode - 1]['mid_sens_values']
bot_sens_data = master_data[episode - 1]['bot_sens_values']

episode_stds_temp = master_data[episode - 1]['stds']
episode_stds = []
for i in range(len(episode_stds_temp)):
    episode_stds.append(np.mean(episode_stds_temp[i]))

top_sens_var, top_sens_mean = calculate_episode_var_mean(top_sens_data, CFD_timesteps_period, CFD_timesteps_action,
                                                         CFD_timesteps_action_one, CFD_timestep_spacing, num_actions,
                                                         sampling_periods)

mid_sens_var, mid_sens_mean = calculate_episode_var_mean(mid_sens_data, CFD_timesteps_period, CFD_timesteps_action,
                                                         CFD_timesteps_action_one, CFD_timestep_spacing, num_actions,
                                                         sampling_periods)

bot_sens_var, bot_sens_mean = calculate_episode_var_mean(bot_sens_data, CFD_timesteps_period, CFD_timesteps_action,
                                                         CFD_timesteps_action_one, CFD_timestep_spacing, num_actions,
                                                         sampling_periods)

plot_episode_cyl_data(front_cyl_data,top_cyl_data,bot_cyl_data)
plot_episode_sens_data(top_sens_data,mid_sens_data,bot_sens_data,CFD_timestep_spacing)

# plot_episode_sensor_sampling(top_sens_data, CFD_timesteps_period, CFD_timesteps_action,
#                              CFD_timesteps_action_one, CFD_timestep_spacing, num_actions,
#                              1.5, 'Top Sensor Data')
#
# plot_episode_sensor_sampling(mid_sens_data, CFD_timesteps_period, CFD_timesteps_action,
#                              CFD_timesteps_action_one, CFD_timestep_spacing, num_actions,
#                              1.5, 'Mid Sensor Data')
#
# plot_episode_sensor_sampling(bot_sens_data, CFD_timesteps_period, CFD_timesteps_action,
#                              CFD_timesteps_action_one, CFD_timestep_spacing, num_actions,
#                              1.5, 'Bot Sensor Data')

# plot_episode_sens_var(top_sens_var,mid_sens_var,bot_sens_var,CFD_timesteps_action_one,CFD_timesteps_action)
# plot_episode_sens_mean(top_sens_mean,mid_sens_mean,bot_sens_mean,CFD_timesteps_action_one,CFD_timesteps_action)
# plot_regular(episode_stds, 'Action Number', 'STD')

total_variance_top = []
total_variance_mid = []
total_variance_bot = []
sampling_time = []
dt = 0.05
min_sampling_period = 0.1
max_sampling_period = 3.0

for j in range(int((max_sampling_period - min_sampling_period) / dt)):
    variance_top_temp = []
    variance_mid_temp = []
    variance_bot_temp = []
    for i in range(130, len(master_data)):
        top_sens_data = master_data[i]['top_sens_values']
        mid_sens_data = master_data[i]['mid_sens_values']
        bot_sens_data = master_data[i]['bot_sens_values']

        top_sens_var, top_sens_mean = calculate_episode_var_mean_two(top_sens_data, CFD_timesteps_period,
                                                                     CFD_timesteps_action,
                                                                     CFD_timesteps_action_one, CFD_timestep_spacing,
                                                                     num_actions, min_sampling_period + dt * j,
                                                                     max_sampling_period)
        mid_sens_var, mid_sens_mean = calculate_episode_var_mean_two(mid_sens_data, CFD_timesteps_period,
                                                                     CFD_timesteps_action,
                                                                     CFD_timesteps_action_one, CFD_timestep_spacing,
                                                                     num_actions, min_sampling_period + dt * j,
                                                                     max_sampling_period)
        bot_sens_var, bot_sens_mean = calculate_episode_var_mean_two(bot_sens_data, CFD_timesteps_period,
                                                                     CFD_timesteps_action,
                                                                     CFD_timesteps_action_one, CFD_timestep_spacing,
                                                                     num_actions, min_sampling_period + dt * j,
                                                                     max_sampling_period)

        variance_top_temp.extend(top_sens_var)
        variance_mid_temp.extend(mid_sens_var)
        variance_bot_temp.extend(bot_sens_var)

    total_variance_top.append(variance_top_temp)
    total_variance_mid.append(variance_mid_temp)
    total_variance_bot.append(variance_bot_temp)

    sampling_time.append(min_sampling_period + dt * j)

top_dv = []
mid_dv = []
bot_dv = []

baseline = 0.1

for i in range(len(sampling_time)):
    if sampling_time[i] >= baseline:
        index = i
        break

for i in range(0, len(total_variance_top)):
    top_dv_temp = np.sqrt((np.array(total_variance_top[i]) ** 2))  # -np.array(total_variance_top[index]))**2)
    top_dv_temp = np.mean(top_dv_temp)
    top_dv.append(top_dv_temp)

    mid_dv_temp = np.sqrt((np.array(total_variance_mid[i]) ** 2))  # -np.array(total_variance_mid[index]))**2)
    mid_dv_temp = np.mean(mid_dv_temp)
    mid_dv.append(mid_dv_temp)

    bot_dv_temp = np.sqrt((np.array(total_variance_bot[i]) ** 2))  # -np.array(total_variance_bot[index]))**2)
    bot_dv_temp = np.mean(bot_dv_temp)
    bot_dv.append(bot_dv_temp)

# plot_regular_two(sampling_time[:],top_dv,'Sampling Time (Shedding Periods)','Top Sensor Variance')
# plot_regular_two(sampling_time[:],mid_dv,'Sampling Time (Shedding Periods)','Mid Sensor Variance')
# plot_regular_two(sampling_time[:],bot_dv,'Sampling Time (Shedding Periods)','Bot Sensor Variance')

# plt.figure(figsize=(15, 7.5))
# plt.plot(sampling_time[:], top_dv, label='Top Sensor Variance')
# plt.plot(sampling_time[:], mid_dv, label='Mid Sensor Variance')
# plt.plot(sampling_time[:], bot_dv, label='Bot Sensor Variance')
# plt.xlabel('Sampling Time (Shedding Periods)', size=15)
# plt.ylabel('Variance', size=15)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.suptitle('Averaged Last 1 Policies', fontsize=16)
# plt.legend()

actions_state_1 = []
actions_state_2 = []
actions_state_3 = []
actions_state_4 = []
actions_state_5 = []
actions_state_6 = []
actions_state_7 = []
actions_state_8 = []
actions_state_9 = []
actions_state_10 = []
actions_state_11 = []
actions_state_12 = []
actions_state_13 = []
actions_state_14 = []
actions_state_15 = []
actions_rewards = []

for i in range(len(master_data)):
    for j in range(len(master_data[i]['states'])):
        actions_state_1.append(master_data[i]['states'][j][0])
        actions_state_2.append(master_data[i]['states'][j][1])
        actions_state_3.append(master_data[i]['states'][j][2])
        actions_state_4.append(master_data[i]['states'][j][3])
        actions_state_5.append(master_data[i]['states'][j][4])
        actions_state_6.append(master_data[i]['states'][j][5])
        actions_state_7.append(master_data[i]['states'][j][6])
        actions_state_8.append(master_data[i]['states'][j][7])
        actions_state_9.append(master_data[i]['states'][j][8])
        actions_state_10.append(master_data[i]['states'][j][9])
        actions_state_11.append(master_data[i]['states'][j][10])
        actions_state_12.append(master_data[i]['states'][j][11])
        actions_state_13.append(master_data[i]['states'][j][12])
        actions_state_14.append(master_data[i]['states'][j][13])
        actions_state_15.append(master_data[i]['states'][j][14])

        actions_rewards.append(master_data[i]['rewards'][j])

# a = np.mean((np.array(actions_state_1) + 1.0) / 5.0)
# b = np.mean((np.array(actions_state_2) + 1.0) / 5.0)
# c = np.mean((np.array(actions_state_3) + 1.0) / 5.0)
#
# d = np.std((np.array(actions_state_1) + 1.0) / 5.0)
# e = np.std((np.array(actions_state_2) + 1.0) / 5.0)
# f = np.std((np.array(actions_state_3) + 1.0) / 5.0)
#
# g = np.mean(np.array(actions_rewards))
# h = np.std(np.array(actions_rewards))

# print('Mean Variance', np.mean([a, b, c]))
# print('STD Variance', np.mean([d, e, f]))
# print('Mean Reward', g)
# print('STD Reward', f)
# print(' ')

# recent_episodes = 10
# recent_actions = recent_episodes * num_actions
# a = np.mean((np.array(actions_state_1[0:750]) + 1.0) / 5.0)
# b = np.mean((np.array(actions_state_2[0:750]) + 1.0) / 5.0)
# c = np.mean((np.array(actions_state_3[0:750]) + 1.0) / 5.0)
#
# d = np.std((np.array(actions_state_1[0:750]) + 1.0) / 5.0)
# e = np.std((np.array(actions_state_2[0:750]) + 1.0) / 5.0)
# f = np.std((np.array(actions_state_3[0:750]) + 1.0) / 5.0)
#
# g = np.mean(np.array(actions_rewards[0:750]))
# h = np.std(np.array(actions_rewards[0:750]))

# print('Recent Mean Variance', np.mean([a, b, c]))
# print('Recent STD Variance', np.mean([d, e, f]))
# print('Recent Mean Reward', g)
# print('Recent STD Reward', f)

# plot_regular(np.array(actions_state_1), 'Actions', 'Top Sensor State')
# plot_regular(np.array(actions_state_2), 'Actions', 'Mid Sensor State')
# plot_regular(np.array(actions_state_3), 'Actions', 'Bot Sensor State')
#
# plot_regular((np.array(actions_state_1) + 1.0) / 5.0, 'Actions', 'Top Sensor Variance')
# plot_regular((np.array(actions_state_2) + 1.0) / 5.0, 'Actions', 'Mid Sensor Variance')
# plot_regular((np.array(actions_state_3) + 1.0) / 5.0, 'Actions', 'Bot Sensor Variance')
#
# plot_regular(actions_state_4, 'Actions', 'Front Cyl State Offset')
# plot_regular(actions_state_5, 'Actions', 'Front Cyl State Amp')
# plot_regular(actions_state_6, 'Actions', 'Front Cyl State Phase')
# plot_regular(actions_state_7, 'Actions', 'Front Cyl State Freq')
#
# plot_regular(actions_state_8, 'Actions', 'Top Cyl State Offset')
# plot_regular(actions_state_9, 'Actions', 'Top Cyl State Amp')
# plot_regular(actions_state_10, 'Actions', 'Top Cyl State Phase')
# plot_regular(actions_state_11, 'Actions', 'Top Cyl State Freq')
#
# plot_regular(actions_state_12, 'Actions', 'Bot Cyl State Offset')
# plot_regular(actions_state_13, 'Actions', 'Bot Cyl State Amp')
# plot_regular(actions_state_14, 'Actions', 'Bot Cyl State Phase')
# plot_regular(actions_state_14, 'Actions', 'Bot Cyl State Freq')
#
# plot_regular(actions_rewards, 'Actions', 'Rewards')

actions_variance_top_sens = []
actions_variance_mid_sens = []
actions_variance_bot_sens = []

for i in range(len(master_data)):
    top_sens_data = master_data[i]['top_sens_values']
    mid_sens_data = master_data[i]['mid_sens_values']
    bot_sens_data = master_data[i]['bot_sens_values']

    top_sens_var, top_sens_mean = calculate_episode_var_mean(top_sens_data, CFD_timesteps_period, CFD_timesteps_action,
                                                             CFD_timesteps_action_one, CFD_timestep_spacing,
                                                             num_actions,
                                                             sampling_periods)

    mid_sens_var, mid_sens_mean = calculate_episode_var_mean(mid_sens_data, CFD_timesteps_period, CFD_timesteps_action,
                                                             CFD_timesteps_action_one, CFD_timestep_spacing,
                                                             num_actions,
                                                             sampling_periods)

    bot_sens_var, bot_sens_mean = calculate_episode_var_mean(bot_sens_data, CFD_timesteps_period, CFD_timesteps_action,
                                                             CFD_timesteps_action_one, CFD_timestep_spacing,
                                                             num_actions,
                                                             sampling_periods)
    actions_variance_top_sens.extend(top_sens_var)
    actions_variance_mid_sens.extend(mid_sens_var)
    actions_variance_bot_sens.extend(bot_sens_var)

top_sens_subtracted = (np.array(actions_state_1) + 1.0) / 5.0 - np.array(actions_variance_top_sens)
mid_sens_subtracted = (np.array(actions_state_2) + 1.0) / 5.0 - np.array(actions_variance_mid_sens)
bot_sens_subtracted = (np.array(actions_state_3) + 1.0) / 5.0 - np.array(actions_variance_bot_sens)
# print('Top Sensor Difference', np.mean(top_sens_subtracted))
# print('Top Sensor Difference', np.mean(mid_sens_subtracted))
# print('Top Sensor Difference', np.mean(bot_sens_subtracted))

if Json_files==True:
    for i in range(len(master_data)):
        json_dict = {'iteration_ID': 0, 'motor_data': {'front': [], 'top': [], 'bot': []},
                     'sensor_data': {'top': [], 'mid': [], 'bot': []},
                     'costs': {'J_fluc': 0, 'gamma': gamma_act, 'J_act': 0, 'J_tot': 0},
                     'total_rewards': 0}
        json_dict['iteration_ID'] = master_data[i]['iteration_ID']
        json_dict['motor_data']['front'].extend(master_data[i]['front_cyl_RPS_PI'])
        json_dict['motor_data']['top'].extend(master_data[i]['top_cyl_RPS_PI'])
        json_dict['motor_data']['bot'].extend(master_data[i]['bot_cyl_RPS_PI'])
        json_dict['sensor_data']['top'].extend(master_data[i]['top_sens_values'])
        json_dict['sensor_data']['mid'].extend(master_data[i]['mid_sens_values'])
        json_dict['sensor_data']['bot'].extend(master_data[i]['bot_sens_values'])
        json_dict['costs']['J_fluc'] = avg_J_flucs[i]
        json_dict['costs']['gamma'] = gamma_act
        json_dict['costs']['J_act'] = avg_J_acts[i]
        json_dict['costs']['J_tot'] = avg_J_tots[i]
        json_dict['average_reward'] = avg_rewards[i]

        iteration = master_data[i]['iteration_ID']
        filename = '../../Production Runs/Production Run 3/json_files/data_iteration_' + str(iteration) + '.json'
        with open(filename, 'w') as outfile:
            json.dump(json_dict, outfile)

plt.show()