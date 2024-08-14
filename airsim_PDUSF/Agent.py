import airsim
import numpy as np
from PIL import Image
import cv2
import random
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from transformations import euler_from_quaternion
import time
from torch.distributions import Categorical
from model import initialize_network_DeepPPO
import importlib
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Agent():
    #def __init__(self,client,input_size,num_actions,name_agent,actor_lr,critic_lr):
    def __init__(self, client, input_size, num_actions, name_agent, learning_rate):
         self.client = client
         self.input_size = input_size
         self.num_actions = num_actions
         self.name_agent = name_agent
         self.network = initialize_network_DeepPPO(num_actions,learning_rate)
    def get_state(self):
        responses1 = self.client.simGetImages([
            airsim.ImageRequest('front_center', airsim.ImageType.Scene, False,
                                False)], vehicle_name=self.name_agent)  # scene vision image in uncompressed RGBA array
        response = responses1[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)  # get numpy array
        img_rgba = img1d.reshape(response.height, response.width, 3)
        img = Image.fromarray(img_rgba)
        img_rgb = img.convert('RGB')
        camera_image_rgb = np.asarray(img_rgb)
        camera_image = camera_image_rgb
        state = cv2.resize(camera_image, (self.input_size, self.input_size), cv2.INTER_LINEAR)
        state = cv2.normalize(state, state, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        state_rgb = []
        state_rgb.append(state[:, :, 0:3])
        state_rgb = np.array(state_rgb)
        current_state = state_rgb.astype('float32')

        return current_state

    def take_action(self,action):

        fov_v = (45 * np.pi / 180) / 1.5
        fov_h = (80 * np.pi / 180) / 1.5
        r = 0.5

        ignore_collision = False
        sqrt_num_actions = np.sqrt(self.num_actions)

        posit = self.client.simGetVehiclePose(vehicle_name=self.name_agent)
        pos = posit.position
        orientation = posit.orientation

        quat = (orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val)
        eulers = euler_from_quaternion(quat)
        alpha = eulers[2]

        theta_ind = int(action / sqrt_num_actions)
        psi_ind = action % sqrt_num_actions

        theta = fov_v / sqrt_num_actions * (theta_ind - (sqrt_num_actions - 1) / 2)
        psi = fov_h / sqrt_num_actions * (psi_ind - (sqrt_num_actions - 1) / 2)

        # r_infer = 2
        # vx = r_infer * np.cos(alpha + psi)
        # vy = r_infer * np.sin(alpha + psi)
        # vz = r_infer * np.sin(theta)
        # # TODO
        # # Take average of previous velocities and current to smoothen out drone movement.
        # self.client.moveByVelocityAsync(vx=vx, vy=vy, vz=vz, duration=1,
        #                                 drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
        #                                 yaw_mode=airsim.YawMode(is_rate=False,
        #                                                         yaw_or_rate=180 * (alpha + psi) / np.pi),)
        # time.sleep(0.07)
        # self.client.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=1)

        noise_theta = (fov_v / sqrt_num_actions) / 6
        noise_psi = (fov_h / sqrt_num_actions) / 6

        psi = psi + random.uniform(-1, 1) * noise_psi
        theta = theta + random.uniform(-1, 1) * noise_theta

        x = pos.x_val + r * np.cos(alpha + psi)
        y = pos.y_val + r * np.sin(alpha + psi)
        z = pos.z_val + r * np.sin(theta)  # -ve because Unreal has -ve z direction going upwards

        self.client.simSetVehiclePose(
            airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(0, 0, alpha + psi)),
            ignore_collison=ignore_collision, vehicle_name=self.name_agent)

    def get_CustomDepth(self):
        camera_name = 2
        # responses = self.client.simGetImages([airsim.ImageRequest(1, airsim.ImageType.DepthPlanner, True)])
        # depth = airsim.list_to_2d_float_array(responses[0].image_data_float, responses[0].width,
        #                                       responses[0].height)
        # thresh = 50
        # super_threshold_indices = depth > thresh
        # depth[super_threshold_indices] = thresh
        # depth = depth / thresh
        #
        # return depth, thresh
        max_tries = 5
        tries = 0
        correct = False
        while not correct and tries < max_tries:
            tries = tries + 1
            responses = self.client.simGetImages(
                [airsim.ImageRequest(camera_name, airsim.ImageType.DepthVis, False, False)],
                vehicle_name=self.name_agent)
            img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)
            # AirSim bug: Sometimes it returns invalid depth map with a few 255 and all 0s
            if np.max(img1d) == 255 and np.mean(img1d) < 0.05:
                correct = False
            else:
                correct = True
        if img1d.size > 1:
            depth = img1d.reshape(responses[0].height, responses[0].width, 3)[:, :, 0]
        thresh = 50
        super_threshold_indices = depth > thresh
        depth[super_threshold_indices] = thresh
        depth = depth / thresh

        return depth, thresh

    def get_reward(self,new_depth,action,thresh):
        L_new, C_new, R_new = self.avg_depth(new_depth, thresh)
        # For now, lets keep the reward a simple one

        if C_new < 0.07:
            done = True
            reward = -1
        else:
            done = False
            if action == 0:
                reward = C_new
            else:
                reward = C_new

        return reward, done
    def avg_depth(self,new_depth, thresh):
        depth_map = new_depth
        global_depth = np.mean(depth_map)
        n = max(global_depth * thresh / 3, 1)
        H = np.size(depth_map, 0)
        W = np.size(depth_map, 1)
        grid_size = (np.array([H, W]) / n)

        # scale by 0.9 to select the window towards top from the mid line
        h = max(int(0.9 * H * (n - 1) / (2 * n)), 0)
        w = max(int(W * (n - 1) / (2 * n)), 0)
        grid_location = [h, w]

        x_start = int(round(grid_location[0]))
        y_start_center = int(round(grid_location[1]))
        x_end = int(round(grid_location[0] + grid_size[0]))
        y_start_right = min(int(round(grid_location[1] + grid_size[1])), W)
        y_start_left = max(int(round(grid_location[1] - grid_size[1])), 0)
        y_end_right = min(int(round(grid_location[1] + 2 * grid_size[1])), W)

        fract_min = 0.05

        L_map = depth_map[x_start:x_end, y_start_left:y_start_center]
        C_map = depth_map[x_start:x_end, y_start_center:y_start_right]
        R_map = depth_map[x_start:x_end, y_start_right:y_end_right]

        if not L_map.any():
            L1 = 0
        else:
            L_sort = np.sort(L_map.flatten())
            end_ind = int(np.round(fract_min * len(L_sort)))
            L1 = np.mean(L_sort[0:end_ind])

        if not R_map.any():
            R1 = 0
        else:
            R_sort = np.sort(R_map.flatten())
            end_ind = int(np.round(fract_min * len(R_sort)))
            R1 = np.mean(R_sort[0:end_ind])

        if not C_map.any():
            C1 = 0
        else:
            C_sort = np.sort(C_map.flatten())
            end_ind = int(np.round(fract_min * len(C_sort)))
            C1 = np.mean(C_sort[0:end_ind])

        return L1, C1, R1

    def reset_to_initial(self,reset_array):
        reset_pos = reset_array[0]
        self.client.simSetVehiclePose(reset_pos, ignore_collison=True, vehicle_name=self.name_agent)
        time.sleep(0.1)


































