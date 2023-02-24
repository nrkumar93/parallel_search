# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical


class MDN(torch.nn.Module):
    """ Mixture Density Networks helper - predicts multivariate gaussian for
    a single variable, in our case one of the dimensions of the xyz movement.
    """
    def __init__(self, num_features, num_outputs, num_clusters, eps=1e-4):
        super(MDN, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.pi = nn.Linear(num_features, num_clusters)
        self.mu = nn.Linear(num_features, num_clusters * num_outputs)
        self.sigma = nn.Linear(num_features, num_clusters * num_outputs)
        self.eps = eps
        self.num_outputs = num_outputs
        self.num_clusters = num_clusters

    def to_device(self, value):
        """ Helper to move things onto our device """
        return value.to(self.device)

    def forward(self, x):
        """
        Predict weight on each cluster, plus center and scale.

        Returns:
        --------
        mu: can be any real number
        pi: logits, can be any real number
        sigma: variance, must be positive
        """
        pi, mu, sigma = self.pi(x), self.mu(x), self.sigma(x)
        pi = F.softmax(pi, dim=-1)
        sigma = F.relu(torch.exp(sigma)) + self.eps
        mu = mu.view(-1, self.num_outputs, self.num_clusters)
        sigma = sigma.view(-1, self.num_outputs, self.num_clusters)
        return pi, mu, sigma

    def sample_from(self, pi, mu, sigma):
        """
        Sample from this set of pi/mu/sigmas.
        """
        #pis = pi[:,None].repeat(1, self.num_outputs, 1)
        mode = Categorical(pi).sample()[:, None]
        mode_1hot = self.to_device(torch.zeros(pi.shape[0], pi.shape[1]))
        mode_1hot.scatter_(1, mode, 1)
        mode_1hot = mode_1hot[:, None].repeat(1, self.num_outputs, 1)
        normal = Normal(mu, sigma)
        x = torch.sum((normal.sample() * mode_1hot), dim=-1)
        return x

    def loss(self, target, pi, mu, sigma):
        """ Mixture Density Networks (MDN) loss """
        m = torch.distributions.Normal(loc=mu, scale=sigma)
        lp = torch.sum(m.log_prob(target), dim=1)
        neg_log_loss = -torch.logsumexp(lp + torch.log(pi), dim=1)
        loss2 = F.relu(neg_log_loss)

        # Check for infinite values
        # If need be -- we can add a small eps somewhere to prevent these
        res = torch.mean(loss2)
        if np.isinf(float(res)) or np.isnan(float(res)):
            print("ERROR: was infinite:", loss2)
            import pdb; pdb.set_trace()
            res = torch.zeros(1).to(self.device)
        return res
