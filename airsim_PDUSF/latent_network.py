import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class C3F2_ActorCriticShared(nn.Module):
    def __init__(self, num_actions):
        super(C3F2_ActorCriticShared, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=4, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 64, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # self.gru_hidden = None
        # self.gru =  nn.GRU(1024, 1024, batch_first=True)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 96, kernel_size=5, stride=3, padding=1)
        self.deconv3 = nn.ConvTranspose2d(96, 3, kernel_size=6, stride=9, padding=1)

        self.bottle_neck = nn.Linear(1024,2048)

        self.fc1_values = nn.Linear(1024, 1024)
        self.fc2_values = nn.Linear(1024, 1)

        self.fc1_actions = nn.Linear(1024, 1024)
        self.fc2_actions = nn.Linear(1024, num_actions)

        # self.fc_vector1 =  nn.Linear(1024, 1024)
        # self.fc_vector2 = nn.Linear(1024, 1024)

        self.fc_h1 = nn.Linear(1024 + num_actions, num_actions)
        self.fc_h2 = nn.Linear(num_actions, num_actions)

        self.fc =  nn.Linear(1024 + num_actions, 1024)


    def forward(self,x):

        NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        x = x.permute(0, 3, 1, 2).float().to(device)
        x = NORMALIZE(x)

        x = self.encoder(x)
        x = x.reshape(x.size(0), -1)  # 展平

        x = self.bottle_neck(x)
        params_s = x.shape[-1] // 2
        mu = x[:, :params_s]
        sigmas = x[:,params_s:]

        #mu, self.gru_hidden = self.gru(mu, self.gru_hidden)

        # # 使用重参数化技巧采样潜在变量
        eps = torch.randn_like(sigmas)  # 从标准正态分布中采样噪声
        z = mu + eps * sigmas  # 采样潜在变量
        # Decoder
        z = z.view(z.size(0), 64, 4, 4)
        decoded_x3 = F.relu(self.deconv1(z))
        decoded_x2 = F.relu(self.deconv2(decoded_x3))
        decoded_image  = F.relu(self.deconv3(decoded_x2))
        decoded_image = decoded_image.permute(0, 2, 3, 1).float().to(device)

        value = F.relu(self.fc1_values(mu))
        value = self.fc2_values(value)

        fc1_actions = F.relu(self.fc1_actions(mu))
        action_probs = F.softmax(self.fc2_actions(fc1_actions), dim=1)

        # reward_vector =F.relu(self.fc_vector1(z))
        # reward_vector = self.fc_vector2(reward_vector)

        # predict_r = torch.sum(mu * reward_vector, dim=1, keepdim=True)

        # z + a ---> h
        h_za = F.sigmoid(self.fc_h1(torch.cat((mu,action_probs),dim=1)))

        # a----> h
        h_a = F.sigmoid(self.fc_h2(action_probs))


        kl_h = F.binary_cross_entropy(h_a+1e-8, h_za+1e-8, reduction='mean')
        #kl_h = F.binary_cross_entropy_with_logits(self.fc_h2(action_probs),
         #                                         self.fc_h1(torch.cat((mu, action_probs), dim=1)), reduction='mean')

        mu_ = self.fc(torch.cat((mu,action_probs),dim=1))

        return action_probs,value,mu,sigmas,mu_,kl_h,decoded_image











