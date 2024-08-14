import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

class ActorCritic(torch.nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()


        # FC and embedding
        self.fc1 = nn.Linear(8192, 512)
        self.embedding_layer = nn.Linear(1024, 512)


        self.fc_omega = nn.Linear(512, 512)

        self.fc_usr1 = nn.Linear(512, 512)
        self.fc_usr2 = nn.Linear(512, 512)

        self.fc_policy = nn.Linear(512, 512)
        self.actor_linear = nn.Linear(512, 4)

        self.fc_phi_a = nn.Linear(512 + 4, 4)
        self.fc_a = nn.Linear(4, 4)




        self.apply(weights_init)
        self.fc_omega.weight.data = normalized_columns_initializer(self.fc_omega.weight.data, 0.1)
        self.fc_omega.bias.data.fill_(0)
        self.apply(weights_init)
        self.fc_usr2.weight.data = normalized_columns_initializer(self.fc_usr2.weight.data, 0.5)
        self.fc_usr2.bias.data.fill_(0)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)

    def forward(self, state_input, target_input):

        # 这里可以使用RNN之类的循环神经网络，但这个在AI2thor数据集上几乎没啥作用，在Habitat模拟器的数据集可以使用RNN而不是连续多帧
        state_input = state_input.reshape(state_input.size(0), -1)
        target_input = target_input.reshape(target_input.size(0), -1)

        #后继特征
        phi_s = F.relu(self.fc1(state_input))
        phi_t = F.relu(self.fc1(target_input))


        #奖励预测向量
        omega = F.relu(self.fc_omega(phi_t))

        #通用后继特征
        input_embedding = torch.cat((phi_s, phi_t), -1)
        output_embedding = F.relu(self.embedding_layer(input_embedding))

        usr = F.relu(self.fc_usr1(output_embedding))
        usr = F.relu(self.fc_usr2(usr))

        value = torch.mean(usr * omega, dim=1, keepdim=True)

        #策略输出
        action_prob = F.relu(self.fc_policy(output_embedding))
        logit = self.actor_linear(action_prob)







        return value, logit
