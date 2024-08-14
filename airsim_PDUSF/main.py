
# 中山大学电与通信工程学院  CPNT 胡蛟城
# 2023年10月

import os, subprocess,sys
import pygame

import time
import csv
import airsim
import numpy as np
import torch
from Agent import Agent
from function import translate_action,train_PPO,connect_drone,initialize_test
from read_cfg import read_cfg
from transformations import euler_from_quaternion
# GPU
device= torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# 打开Airsim仿真环境
env_name="indoor_long"
env_folder = os.path.dirname(os.path.abspath(__file__)) + "/unreal_envs/" + env_name + "/"
path = env_folder + env_name + ".exe"
env_process = subprocess.Popen(path)
time.sleep(2)
print("Successfully loaded environment: " + env_name)

client = []
name_agent = 'drone0'
client, old_posit, initZ = connect_drone(ip_address='127.0.0.5',client=client)

# 获取无人机初始位置
old_posit = client.simGetVehiclePose(vehicle_name=name_agent)
z = old_posit.position.z_val
pp = airsim.Pose(airsim.Vector3r(0, 0, z), airsim.to_quaternion(0, 0, np.pi))


reset_array = []  # 储存无人机三维位置信息、欧拉角转换后的四元素信息
reset_array.append(pp)


# 参数设置
input_size = 103
num_actions = 25
distance = 0
data_tuple = []
ret = 0
epi_num = 0
iter = 0
last_crash = 0
epi_max = 200000
epochs = 10
lr = 1e-4
lmbda = 0.95
gamma = 0.99
eps = 0.2
crash = False
Train = True
Test = False

# 实例化模型
agent = Agent(client, input_size, num_actions, name_agent, lr)
# 获取无人机当前状态

current_state = agent.get_state()
current_state = torch.tensor(current_state).to(device)




# 模型训练
network_path = 'models/trained/Indoor/model'
if not os.path.exists(network_path):
    os.makedirs(network_path)



if Test:
    env_folder = 'unreal_envs/indoor_long/'
    env_cfg = read_cfg(config_filename=env_folder+'config.cfg')
    altitude = []
    nav_x = []
    nav_y = []
    p_z, f_z, fig_z, ax_z, line_z, fig_nav, ax_nav, nav = initialize_test(env_cfg,env_folder)
    nav_text = ax_nav.text(0, 0, '')
    agent.reset_to_initial(reset_array)
    old_posit = client.simGetVehiclePose(vehicle_name=name_agent)

while Train:
   try:
        # 根据动作空间概率分布选择动作a
        action, action_prob = agent.network.select_action(current_state)
        action_word = translate_action(action.item(),num_actions)
        # step a,获得下一时刻状态s
        agent.take_action(action.item())
        new_state = agent.get_state()
        new_state = torch.tensor(new_state).to(device)
        # 计算深度
        new_depth, thresh = agent.get_CustomDepth()
        # 计算奖励
        reward_depth, crash = agent.get_reward(new_depth,action,thresh)

        # 获取无人机位置
        posit = client.simGetVehiclePose(vehicle_name=name_agent)
        position = posit.position
        old_p = np.array(
            [old_posit.position.x_val, old_posit.position.y_val])
        new_p = np.array([position.x_val, position.y_val])

        # 计算无人机运动距离
        distance = distance + np.linalg.norm(new_p - old_p)
        old_posit = posit

        reward = reward_depth

        # 判断无人机是否发生碰撞
        if client.simGetCollisionInfo(vehicle_name=name_agent).has_collided or distance < 0.01:
            if client.simGetCollisionInfo(vehicle_name=name_agent).has_collided:
                print('Crash: Collision detected from environment')
            else:
                print('Crash: Collision detected from distance')
            crash = True
            reward = -1

        ret = ret + reward
        # 将(s,a,s',r)储存在经验池中
        data_tuple.append([current_state, action, new_state, reward, action_prob, crash])

        if crash:
            if distance < 0.01:
                client, old_posit, initZ = connect_drone(ip_address='127.0.0.5', client=client)
                time.sleep(2)
                agent.client = client
                old_posit = client.simGetVehiclePose(vehicle_name=name_agent)

            else:
                # on-policy
                train_PPO(data_tuple, agent, epochs, input_size, lmbda, gamma, eps)
                epi_num = epi_num + 1
                if epi_num % 100 == 0:
                    agent.network.save_network(network_path,epi_num)


                with open('log_file0.csv', mode='a', newline='') as log_file:
                        writer = csv.writer(log_file)
                        writer.writerow([ret, distance, len(data_tuple)])
                ret = 0
                distance = 0
                data_tuple = []
                last_crash = 0
                # 无人机位置初始化
                agent.reset_to_initial(reset_array)
                # 获取无人机当前状态
                current_state = agent.get_state()
                current_state = torch.tensor(current_state).to(device)
                # 获取无人机坐标
                old_posit = client.simGetVehiclePose(vehicle_name=name_agent)

        else:
            current_state = new_state

            s_log = '{:<6s} - Iter: {:>6d}/{:<5d} {:<8s}  Last Crash = {:<5d}  SF = {:<5.4f}  Reward: {:<+1.4f}  '.format(
                name_agent,
                iter,
                epi_num,
                action_word,
                last_crash,
                distance,
                reward)
            print_interval = 1
            if iter % print_interval == 0:
                print(s_log)
            iter = iter + 1
            last_crash = last_crash + 1


   except Exception as e:

            if str(e) == 'cannot reshape array of size 1 into shape (0,0,3)':
                print('Recovering from AirSim error')

                client, old_posit, initZ = connect_drone(ip_address='127.0.0.3', client=client)
                time.sleep(2)
                agent.client = client

            else:
                print('------------- Error -------------')
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print(exc_obj)
                automate = False
                print('Hit r and then backspace to start from this point')





while Test:
    try:
            agent_state = agent.get_state()
            if client.simGetCollisionInfo(vehicle_name=name_agent).has_collided:
                with open('distance_file.csv', mode='a', newline='') as log_file:
                        writer = csv.writer(log_file)
                        writer.writerow([distance])
                print('Drone collided')
                print("Total distance traveled: ", np.round(distance, 2))
                client.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=1, vehicle_name=name_agent).join()

                if nav_x:  # Nav_x is empty if the drone collides in first iteration
                    ax_nav.plot(nav_x.pop(), nav_y.pop(), 'r*', linewidth=20)
                file_path = 'results/'
                fig_z.savefig(file_path + 'altitude_variation.png', dpi=500)
                fig_nav.savefig(file_path + 'navigation.png', dpi=500)
                print('Figures saved')
                break
            else:
                posit = client.simGetVehiclePose(vehicle_name=name_agent)
                distance = distance + np.linalg.norm(np.array(
                    [old_posit.position.x_val - posit.position.x_val,
                     old_posit.position.y_val - posit.position.y_val]))

                altitude.append(-posit.position.z_val - f_z)

                quat = (posit.orientation.w_val, posit.orientation.x_val,
                        posit.orientation.y_val, posit.orientation.z_val)
                yaw = euler_from_quaternion(quat)[2]

                x_val = posit.position.x_val
                y_val = posit.position.y_val
                z_val = posit.position.z_val

                nav_x.append(env_cfg.alpha * x_val + env_cfg.o_x)
                nav_y.append(env_cfg.alpha * y_val + env_cfg.o_y)
                nav.set_data(nav_x, nav_y)
                nav_text.remove()
                nav_text = ax_nav.text(25, 55, 'Distance: ' + str(np.round(distance, 2)),
                                       style='italic',
                                       bbox={'facecolor': 'white', 'alpha': 0.5})

                line_z.set_data(np.arange(len(altitude)), altitude)
                ax_z.set_xlim(0, len(altitude))
                fig_z.canvas.draw()
                fig_z.canvas.flush_events()

                current_state = agent.get_state()
                current_state = torch.tensor(current_state).to(device)
                action, action_prob = agent.network.select_action(current_state)
                action_word = translate_action(action.item(), num_actions)
                agent.take_action(action.item())
                old_posit = posit

                s_log = 'Position = ({:<3.2f},{:<3.2f}, {:<3.2f}) Orientation={:<1.3f} Predicted Action: {:<8s}  '.format(
                    x_val, y_val, z_val, yaw, action_word
                )
                print(s_log)

    except Exception as e:

            if str(e) == 'cannot reshape array of size 1 into shape (0,0,3)':
                print('Recovering from AirSim error')

                client, old_posit, initZ = connect_drone(ip_address='127.0.0.5', client=client)
                time.sleep(2)
                agent.client = client

            else:
                print('------------- Error -------------')
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print(exc_obj)
                automate = False
                print('Hit r and then backspace to start from this point')





