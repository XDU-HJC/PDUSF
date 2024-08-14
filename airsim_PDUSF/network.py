
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class C3F2_ActorCriticShared(nn.Module):
    def __init__(self, num_actions):
        super(C3F2_ActorCriticShared, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, kernel_size=7, stride=4, padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(96, 64, kernel_size=5, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)


        self.fc1_values = nn.Linear(1024, 1024)
        self.fc2_values = nn.Linear(1024, 1)

        self.fc1_actions = nn.Linear(1024, 1024)
        self.fc2_actions = nn.Linear(1024, num_actions)


    def forward(self, x):

        x = x.permute(0, 3, 1, 2).float().to(device)
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)

        x = F.relu(self.conv2(x))

        x = self.maxpool2(x)

        x = F.relu(self.conv3(x))

        x = x.reshape(x.size(0), -1)  # 展平

        value = F.relu(self.fc1_values(x))
        value = self.fc2_values(value)

        fc1_actions = F.relu(self.fc1_actions(x))
        action_probs = F.softmax(self.fc2_actions(fc1_actions), dim=1)


        return action_probs, value



