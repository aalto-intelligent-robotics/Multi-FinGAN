import torch.nn as nn
from .networks import NetworkBase
import torch
import torch.nn.functional as F


class Network(NetworkBase):
    def __init__(self, input_dim):
        super(Network, self).__init__()
        self.name = 'grasp_generator'
        self.fc0 = nn.Linear(2048, 128)
        self.fc1 = nn.Linear(128 + input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 312)
        self.fc4 = nn.Linear(312, 256)
        self.fcHR_residual = nn.Linear(256, 128)
        self.fcHR_residual_2 = nn.Linear(
            128, 1)
        self.fcR = nn.Linear(256, 64)
        self.fcR_2 = nn.Linear(64, 3)
        self.fcT = nn.Linear(256, 64)
        self.fcT_2 = nn.Linear(64, 3)

    def forward(self, image_representation, hand_representations, Ro, Ts):
        x_image = self.fc0(image_representation)
        x = self.fc1(torch.cat((x_image, hand_representations, Ro, Ts), -1))

        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)

        x_hr = self.fcHR_residual(x)
        x_hr = F.relu(x_hr)
        HR = torch.zeros((Ro.shape[0], 7)).cuda()
        # Only output the spread between the fingers and not the actual finger rotations as those are refined
        # in the refinement layer
        HR[:, :1] = self.fcHR_residual_2(
            x_hr)
        HR += hand_representations
        x_r = self.fcR(x)
        R = Ro + self.fcR_2(x_r)
        x_t = self.fcT(x)
        x_t = F.relu(x_t)
        T = Ts + self.fcT_2(x_t)
        return HR, R, T
