import torch
import torch.nn as nn
import torch.optim as optim
import scipy
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from network import C3F2_ActorCriticShared
import numpy as np
import torchvision.transforms as transforms
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
custom_load_path = 'models/trained/Indoor/model_900.pth'
custom_load = False
class initialize_network_DeepPPO(nn.Module):
    def __init__(self,num_actions,learning_rate):
        super(initialize_network_DeepPPO, self).__init__()

        self.iter_policy = 0
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.model = C3F2_ActorCriticShared(self.num_actions).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.99))
        if custom_load:
            print('Loading weights from: ', custom_load_path)
            self.load_network(custom_load_path)


    def train_policy(self,curr_states,next_states,actions, td_target, action_prob_old, advantage, eps):
        self.iter_policy += 1
        self.curr_states = curr_states
        self.next_states = next_states
        self.action_prob_old = action_prob_old
        self.actions = actions
        self.advantage = advantage
        self.td_target = td_target

        self.action_prob,self.state_value = self.model(self.curr_states)

        self.ind = torch.nn.functional.one_hot(self.actions.squeeze(), num_classes=self.num_actions).float()
        self.action_prob_new = torch.sum(self.action_prob * self.ind, dim=1, keepdim=True)
        self.ratio = torch.exp(torch.log(self.action_prob_new + 1e-10) - torch.log(self.action_prob_old + 1e-10))

        p1 = self.ratio * self.advantage
        p2 = torch.clamp(self.ratio, 1 - eps, 1 + eps) * self.advantage

        self.loss_actor_op = -torch.mean(torch.minimum(p1, p2))

        self.loss_entropy = torch.mean((torch.log(self.action_prob + 1e-8) * self.action_prob))

        self.loss_critic_op = 0.5 * F.mse_loss(self.state_value, self.td_target)
        self.loss_op = self.loss_critic_op + self.loss_actor_op + 0.01 * self.loss_entropy

        self.optimizer.zero_grad()
        self.loss_op.backward()
        self.optimizer.step()


    def select_action(self,state):

        self.model.eval()
        with torch.no_grad():
            action_probs,_= self.model(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_action_prob = dist.log_prob(action)
        action_prob = log_action_prob.exp()
        self.model.train()

        return action, action_prob


    def get_state_value(self, state):


        self.model.eval()
        with torch.no_grad():
            _, value = self.model(state)
        self.model.train()
        return value


    def log_to_tensorboard(self, tag, group, value, index):
         tag = group + '/' + tag
         self.stat_writer = SummaryWriter(log_dir='logs')
         self.stat_writer.add_scalar(tag, value, index)


    def save_network(self,save_path,episode=''):
        save_path = f"{save_path}_{episode}.pth"
        torch.save(self.state_dict(),save_path)
    

    def load_network(self,load_path):
        self.load_state_dict(torch.load(load_path))
