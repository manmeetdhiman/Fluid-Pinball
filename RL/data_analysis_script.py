import pickle
import numpy as np
import matplotlib.pyplot as plt
import json

num_iterations=10
num_policies=4
CFD_timestep=5e-4
CFD_timestep_spacing=5
num_actions=15
dur_actions=1.004198
dur_action_one=1.50838
shedding_freq=8.42
free_stream_vel=1.5
sampling_periods=0.9
gamma_act=0.02

Json_files=True

shedding_period=1/shedding_freq
CFD_timesteps_period=shedding_period/CFD_timestep
CFD_timesteps_action=CFD_timesteps_period*dur_actions
CFD_timesteps_action_one=CFD_timesteps_period*dur_action_one

master_data=[]

for policy in range(num_policies):
    for iteration in range(num_iterations):
        iteration_ID=policy*num_iterations+iteration+1
        filename='pickle_files/data_iteration_' + str(iteration_ID)+'.pickle'
        with open(filename, 'rb') as handle:
            data = pickle.load(handle)
        master_data.append(data)

# filename='actor_critic_losses.'+str(num_policies)+'.pickle'
# with open(filename,'rb') as handle:
#     data=pickle.load(handle)
# actor_losses=data['actor_losses']
# critic_losses=data['critic_losses']


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
    sampling_timesteps = int(CFD_timesteps_period * sampling_periods / CFD_timestep_spacing)
    episode_var = []
    episode_mean = []

    for i in range(num_actions):
        if i == 0:
            sampling_end = np.int(np.ceil(CFD_timesteps_action_one / CFD_timestep_spacing))
        else:
            sampling_end = np.int(np.ceil((CFD_timesteps_action_one + i * CFD_timesteps_action) / CFD_timestep_spacing))

        sens_var = np.var(sens_data[(sampling_end - sampling_timesteps):(sampling_end)])
        sens_mean = np.mean(sens_data[(sampling_end - sampling_timesteps):(sampling_end)])

        episode_var.append(sens_var)
        episode_mean.append(sens_mean)

    return episode_var, episode_mean

def plot_regular(y_data,x_label,y_label):
    x_data = np.arange(1,41,1)
    plt.figure(figsize=(15,7.5))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # m, b = np.polyfit(x_data, y_data, 1)
    plt.plot(x_data, y_data)
    # plt.plot(x_data, m * x_data + b)
    plt.xlabel(x_label, fontsize=15)
    plt.ylabel(y_label, fontsize=15)
    # print(m,b)


def plot_policy_regular(y_data,x_label,y_label,num_iterations):
    x_data=np.zeros(len(y_data))
    for i in range(len(x_data)):
        x_data[i]=i//num_iterations+1
    plt.figure(figsize=(15,7.5))
    plt.xticks(np.arange(min(x_data), max(x_data)+1, 1), fontsize=15)
    plt.yticks(fontsize=15)
    plt.scatter(x_data,y_data)
    plt.xlabel(x_label,fontsize=15)
    plt.ylabel(y_label,fontsize=15)


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
    plt.yticks(fontsize=15)
    plt.plot(x_data, y_data, label='Moving Average')
    plt.plot(x_data, y_data_upper, label='Moving Average + STD')
    plt.plot(x_data, y_data_lower, label='Moving Average - STD')
    plt.xlabel(x_label, fontsize=15)
    plt.ylabel(y_label, fontsize=15)
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
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.scatter(x_data, y_data)
    plt.errorbar(x_data, y_data, yerr=y_data_std, linestyle="None", capsize=10)
    plt.xlabel(x_label, fontsize=15)
    plt.ylabel(y_label, fontsize=15)
    plt.xticks(np.arange(min(x_data), max(x_data) + 1, 1))


def plot_episode_cyl_data(front_cyl_data, top_cyl_data, bot_cyl_data):
    plt.figure(figsize=(15, 7.5))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.plot(front_cyl_data, label='Front Cylinder')
    plt.plot(top_cyl_data, label='Top Cylinder')
    plt.plot(bot_cyl_data, label='Bottom Cylinder')
    plt.xlabel('CFD Timesteps', fontsize=15)
    plt.ylabel('Cylinder Rotation Rate (rad/s)', fontsize=15)
    plt.legend(loc='upper right')


def plot_episode_sens_data(top_sens_data, mid_sens_data, bot_sens_data, CFD_timestep_spacing):
    x_data = np.zeros(len(top_sens_data))
    for i in range(len(x_data)):
        x_data[i] = (i + 1) * CFD_timestep_spacing

    plt.figure(figsize=(15, 7.5))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.plot(x_data, top_sens_data, label='Top Sensor')
    plt.plot(x_data, mid_sens_data, label='Mid Sensor')
    plt.plot(x_data, bot_sens_data, label='Bottom Sensor')
    plt.xlabel('CFD Timesteps', fontsize=15)
    plt.ylabel('Sensor Speed (m/s)', fontsize=15)
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
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.step(x_data, top_sens_var, label='Top Sensor')
    plt.step(x_data, mid_sens_var, label='Mid Sensor')
    plt.step(x_data, bot_sens_var, label='Bottom Sensor')
    plt.xlabel('CFD Timesteps', fontsize=15)
    plt.ylabel('Sensor Variance', fontsize=15)
    plt.legend(loc='upper right')


# def plot_episode_sens_mean(top_sens_mean, mid_sens_mean, bot_sens_mean, CFD_timesteps_action_one,
#                            CFD_timesteps_action):
#     x_data = np.zeros(len(top_sens_var))
#     for i in range(len(x_data)):
#         if i == 0:
#             x_data[i] = CFD_timesteps_action_one
#         else:
#             x_data[i] = CFD_timesteps_action_one + i * CFD_timesteps_action
#
#     plt.figure(figsize=(15, 7.5))
#     plt.xticks(fontsize=15)
#     plt.yticks(fontsize=15)
#     plt.step(x_data, top_sens_mean, label='Top Sensor')
#     plt.step(x_data, mid_sens_mean, label='Mid Sensor')
#     plt.step(x_data, bot_sens_mean, label='Bottom Sensor')
#     plt.xlabel('CFD Timesteps', fontsize=15)
#     plt.ylabel('Sensor Mean (m/s)', fontsize=15)
#     plt.legend(loc='upper right')


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

total_J_tots = np.array(total_J_flucs) + gamma_act * np.array(total_J_acts)
avg_J_tots = np.array(avg_J_flucs) + gamma_act * np.array(avg_J_acts)

# PERFORMANCE CHECKS


#plot_regular(total_rewards,'Episode Number','Total Episode Rewards')
plot_regular(avg_rewards,'Episode Number','Average Episode Rewards')

#plot_regular(total_J_flucs,'Episode Number','Total J_Flucs')
plot_regular(avg_J_flucs,'Episode Number','Average J_Flucs')

#plot_regular(total_J_acts,'Episode Number','Total J_Acts')
plot_regular(avg_J_acts,'Epsiode Number','Average J_Acts')

#plot_regular(total_J_tots,'Episode Number','Total J_Total')
plot_regular(avg_J_tots,'Episode Number','Average J_Total')

#plot_policy_regular(total_rewards,'Policy Number','Total Episode Rewards',num_iterations)
#plot_policy_regular(total_J_flucs,'Policy Number','Total J_Flucs',num_iterations)
#plot_policy_regular(total_J_acts,'Policy Number','Total J_Acts',num_iterations)
#plot_policy_regular(total_J_tots,'Policy Number','Total J_Total',num_iterations)

plot_policy_regular(avg_rewards,'Policy Number','Average Episode Rewards',num_iterations)
plot_policy_regular(avg_J_flucs,'Policy Number','Average J_Flucs',num_iterations)
plot_policy_regular(avg_J_acts,'Policy Number','Average J_Acts',num_iterations)
plot_policy_regular(avg_J_tots,'Policy Number','Average J_Total',num_iterations)

#plot_policy_average(total_rewards,'Policy Number','Policy Total Rewards',num_iterations)
#plot_policy_average(total_J_flucs,'Policy Number','Policy Total J_Flucs',num_iterations)
#plot_policy_average(total_J_acts,'Policy Number','Policy Total J_Acts',num_iterations)
#plot_policy_average(total_J_tots,'Policy Number','Policy Total J_Tots',num_iterations)

plot_policy_average(avg_rewards,'Policy Number','Policy Average Rewards',num_iterations)
plot_policy_average(avg_J_flucs,'Policy Number','Policy Average J_Flucs',num_iterations)
plot_policy_average(avg_J_acts,'Policy Number','Policy Average J_Acts',num_iterations)
plot_policy_average(avg_J_tots,'Policy Number','Policy Average J_Tots',num_iterations)

#RL CONVERGENCE CHECKS

# plot_regular(avg_STD,'Episode Number','Average STD')
# plot_regular(avg_values,'Episode Number','Average Value')
# plot_regular(critic_losses,'Policy Number','Critic Losses')
# plot_regular(actor_losses,'Policy Number','Actor Losses')

# EPISODE CHECKS

# episode = 3
#
# front_cyl_data = master_data[episode - 1]['front_cyl_RPS_PI']
# top_cyl_data = master_data[episode - 1]['top_cyl_RPS_PI']
# bot_cyl_data = master_data[episode - 1]['bot_cyl_RPS_PI']
#
# top_sens_data = master_data[episode - 1]['top_sens_values']
# mid_sens_data = master_data[episode - 1]['mid_sens_values']
# bot_sens_data = master_data[episode - 1]['bot_sens_values']
#
# top_sens_var, top_sens_mean = calculate_episode_var_mean(top_sens_data, CFD_timesteps_period, CFD_timesteps_action,
#                                                          CFD_timesteps_action_one, CFD_timestep_spacing, num_actions,
#                                                          sampling_periods)
#
# mid_sens_var, mid_sens_mean = calculate_episode_var_mean(mid_sens_data, CFD_timesteps_period, CFD_timesteps_action,
#                                                          CFD_timesteps_action_one, CFD_timestep_spacing, num_actions,
#                                                          sampling_periods)
#
# bot_sens_var, bot_sens_mean = calculate_episode_var_mean(bot_sens_data, CFD_timesteps_period, CFD_timesteps_action,
#                                                          CFD_timesteps_action_one, CFD_timestep_spacing, num_actions,
#                                                          sampling_periods)
#
# plot_episode_cyl_data(front_cyl_data, top_cyl_data, bot_cyl_data)
# plot_episode_sens_data(top_sens_data, mid_sens_data, bot_sens_data, CFD_timestep_spacing)
# plot_episode_sens_var(top_sens_var, mid_sens_var, bot_sens_var, CFD_timesteps_action_one, CFD_timesteps_action)
# plot_episode_sens_mean(top_sens_mean, mid_sens_mean, bot_sens_mean, CFD_timesteps_action_one, CFD_timesteps_action)
#
# actions_state_1 = []
# actions_state_2 = []
# actions_state_3 = []
# actions_state_4 = []
# actions_state_5 = []
# actions_state_6 = []
#
# actions_rewards = []
#
# for i in range(len(master_data)):
#     for j in range(3, len(master_data[i]['states'])):
#         actions_state_1.append(master_data[i]['states'][j][0])
#         actions_state_2.append(master_data[i]['states'][j][1])
#         actions_state_3.append(master_data[i]['states'][j][2])
#         actions_state_4.append(master_data[i]['states'][j][3])
#         actions_state_5.append(master_data[i]['states'][j][4])
#         actions_state_6.append(master_data[i]['states'][j][5])
#         actions_rewards.append(master_data[i]['rewards'][j])
#
# plot_regular(np.array(actions_state_1), 'Actions', 'Top Sensor State')
# plot_regular(np.array(actions_state_2), 'Actions', 'Mid Sensor State')
# plot_regular(np.array(actions_state_3), 'Actions', 'Bot Sensor State')
#
# plot_regular((np.array(actions_state_1) + 1.13) / 4.26, 'Actions', 'Top Sensor Variance')
# plot_regular((np.array(actions_state_2) + 1.13) / 4.26, 'Actions', 'Mid Sensor Variance')
# plot_regular((np.array(actions_state_3) + 1.13) / 4.26, 'Actions', 'Bot Sensor Variance')
#
# plot_regular(actions_state_4, 'Actions', 'Front Cyl State')
# plot_regular(actions_state_5, 'Actions', 'Top Cyl State')
# plot_regular(actions_state_6, 'Actions', 'Bot Cyl State')
#
# plot_regular(actions_rewards, 'Actions', 'Rewards')
#
# a = np.mean((np.array(actions_state_1)))
# b = np.mean((np.array(actions_state_2)))
# c = np.mean((np.array(actions_state_2)))

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
        filename = 'json_files/data_iteration_' + str(iteration) + '.json'
        with open(filename, 'w') as outfile:
            json.dump(json_dict, outfile)

plt.show()