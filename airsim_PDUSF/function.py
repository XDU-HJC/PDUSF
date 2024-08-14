
import numpy as np
import torch
import airsim
import time
import os
from matplotlib import pyplot as plt
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



def connect_drone(ip_address='127.0.0.0', client=[]):
    if client != []:
        client.reset()
    client = airsim.MultirotorClient(ip=ip_address, timeout_value=10)
    client.confirmConnection()
    time.sleep(1)

    name_agent = "drone0"
    client.enableApiControl(True, name_agent)
    client.armDisarm(True, name_agent)

    client.takeoffAsync(vehicle_name=name_agent)
    time.sleep(1)
    old_posit = client.simGetVehiclePose(vehicle_name=name_agent)

    initZ = old_posit.position.z_val

    return client, old_posit, initZ
def translate_action(action,num_actions):
    # action_word = ['Forward', 'Right', 'Left', 'Sharp Right', 'Sharp Left']
    sqrt_num_actions = np.sqrt(num_actions)
    # ind = np.arange(sqrt_num_actions)
    if sqrt_num_actions % 2 == 0:
        v_string = list('U' * int((sqrt_num_actions - 1) / 2) + 'D' * int((sqrt_num_actions - 1) / 2))
        h_string = list('L' * int((sqrt_num_actions - 1) / 2) + 'R' * int((sqrt_num_actions - 1) / 2))
    else:
        v_string = list('U' * int(sqrt_num_actions / 2) + 'F' + 'D' * int(sqrt_num_actions / 2))
        h_string = list('L' * int(sqrt_num_actions / 2) + 'F' + 'R' * int(sqrt_num_actions / 2))

    v_ind = int(action / sqrt_num_actions)
    h_ind = int(action % sqrt_num_actions)
    action_word = v_string[v_ind] + str(int(np.ceil(abs((sqrt_num_actions - 1) / 2 - v_ind)))) + '-' + h_string[
        h_ind] + str(int(np.ceil(abs((sqrt_num_actions - 1) / 2 - h_ind))))

    return action_word

def train_PPO(data_tuple,agent,epochs,input_size,lmbda,gamma,eps):

    num_batches = 1

    for i in range(num_batches):
         episode_len = len(data_tuple)

         curr_states = torch.zeros(episode_len, input_size, input_size, 3).to(device)
         next_states = torch.zeros(episode_len, input_size, input_size, 3).to(device)
         action_prob = torch.zeros(episode_len, 1).to(device)
         actions = torch.zeros(episode_len, 1, dtype=int).to(device)
         crashes = torch.zeros(episode_len, 1).to(device)
         rewards = torch.zeros(episode_len, 1).to(device)

         for ii, m in enumerate(data_tuple):
            curr_state_m, action_m, next_state_m, reward_m, action_prob_m, crash_m = m
            curr_states[ii, :, :, :] = curr_state_m[...]
            next_states[ii, :, :, :] = next_state_m[...]
            actions[ii] = action_m
            rewards[ii] =torch.tensor(reward_m).to(device)
            action_prob[ii] = action_prob_m
            crashes[ii] = ~crash_m

         for i in range(epochs):
            # 目标，下一个状态的state_value  [batchsize,1]
            next_q_target = agent.network.get_state_value(next_states)
            # 目标，当前状态的state_value  [batchsize,1]
            td_target = rewards + gamma * next_q_target * crashes
            # 预测，当前状态的state_value  [batchsize,1]
            td_value = agent.network.get_state_value(curr_states)
            # 目标值和预测值state_value之差  [batchsize,1]
            td_delta = td_target - td_value

            # 时序差分值
            td_delta = td_delta.cpu().detach().numpy()
            advantage_list = []
            advantage = 0

            # 计算优势函数
            for delta in td_delta[::-1]:  # 逆序时序差分值 axis=1轴上倒着取 [], [], []
                # 优势函数GAE的公式
                advantage = gamma * lmbda * advantage + delta
                advantage_list.append(advantage)
            # 正序
            advantage_list.reverse()

            # 减小训练方差
            mean_advantage = np.mean(advantage_list)
            std_advantage = np.std(advantage_list)
            # 对advantage_list中的每个值进行标准化
            normalized_advantage_list = [(adv - mean_advantage) / (std_advantage + 1e-8) for adv in advantage_list]
            # 将标准化后的advantage_list重新赋值给advantage_list
            advantage_list = normalized_advantage_list
            advantage = torch.tensor(advantage_list, dtype=torch.float).to(device)

            agent.network.train_policy(curr_states, next_states, actions, td_target, action_prob, advantage, eps)

def initialize_test(env_cfg, env_folder):
    if not os.path.exists('results'):
        os.makedirs('results')

    # Mapping floor to 0 height
    f_z = env_cfg.floor_z / 100
    c_z = (env_cfg.ceiling_z - env_cfg.floor_z) / 100
    p_z = (env_cfg.player_start_z - env_cfg.floor_z) / 100

    plt.ion()
    fig_z = plt.figure()
    ax_z = fig_z.add_subplot(111)
    line_z, = ax_z.plot(0, 0)
    ax_z.set_ylim(0, c_z)
    plt.title("Altitude variation")

    fig_nav = plt.figure()
    ax_nav = fig_nav.add_subplot(111)
    img = plt.imread(env_folder + env_cfg.floorplan)
    ax_nav.imshow(img)
    plt.axis('off')
    plt.title("Navigational map")
    plt.plot(env_cfg.o_x, env_cfg.o_y, 'b*', linewidth=20)
    nav, = ax_nav.plot(env_cfg.o_x, env_cfg.o_y)

    return p_z, f_z, fig_z, ax_z, line_z, fig_nav, ax_nav, nav



