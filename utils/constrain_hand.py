import torch.nn as nn
import torch
from utils import util
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


class ConstrainHand(nn.Module):
    def __init__(self, constrain_method, batch_size, device='cpu'):
        super(ConstrainHand, self).__init__()
        self.constain_method = constrain_method
        if constrain_method == "hard":
            self.constraint = nn.Hardtanh(0, util.deg2rad(180))
        if constrain_method == "soft":
            self.constraint = nn.Sigmoid()
        if constrain_method == "cvx":
            self.set_cvx_layer(batch_size, device)

    def forward(self, HR):
        HR_unconstrained = HR.clone()
        if self.constain_method == "cvx":
            solution, = self.cvxpylayer(self.theta_max_torch, self.theta_min_torch, HR)
            less_than_min = HR < self.theta_min_torch
            HR[less_than_min] = HR[less_than_min] + solution[less_than_min]
            more_than_max = HR > self.theta_max_torch
            HR[more_than_max] = HR[more_than_max] - solution[more_than_max]
        elif self.constain_method == "hard":
            HR[:, 0] = self.constraint(1*HR[:, 0])
        elif self.constain_method == "soft":
            HR[:, 0] = self.constraint(HR[:, 0])*util.deg2rad(180)
        return HR, HR_unconstrained

    def set_cvx_layer(self, batch_size, device):
        x = cp.Variable((batch_size, 7))
        theta_max = cp.Parameter((batch_size, 7))
        theta_min = cp.Parameter((batch_size, 7))
        theta = cp.Parameter((batch_size, 7))
        constraints = [theta-x <= theta_max, theta+x >= theta_min]
        objective = cp.Minimize(cp.pnorm(x))
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()
        self.cvxpylayer = CvxpyLayer(problem, parameters=[
            theta_max, theta_min, theta], variables=[x])
        eps = 1e-10
        self.theta_max_torch = util.deg2rad(torch.tensor(
            [180., 140., 140., 140., 48., 48., 48.], requires_grad=True)).to(device)-eps
        self.theta_max_torch = self.theta_max_torch.unsqueeze(
            0).repeat(batch_size, 1)+eps
        self.theta_min_torch = torch.zeros(
            (batch_size, 7), requires_grad=True).to(device)
