import torch
import torch.nn as nn
import numpy as np

class Sine_layer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, omega_0=30., is_first=False):
        super().__init__()
        
        self.in_features = in_features
        self.omega_0 = omega_0
        self.is_first = is_first

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, np.sqrt(6 / self.in_features) / self.omega_0)
    
    def forward(self, x):
        y = torch.sin(self.omega_0 * self.linear(x))
        return y

class Siren(nn.Module):
    def __init__(self, n_hidden_layers=2, hidden_features=256, first_omega_0=30., hidden_omega_0=30., outermost_linear=True):
        '''Our classic SIREN network for this project'''
        super().__init__()

        # Constants
        in_features = 2
        out_features = 1

        self.net = []

        # First layer
        self.net.append(Sine_layer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))

        # Hidden layers
        for i in range(n_hidden_layers):
            self.net.append(Sine_layer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))

        # Last layer
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, np.sqrt(6 / hidden_features) / hidden_omega_0)
            self.net.append(final_linear)
        else:
            self.net.append(Sine_layer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        x = x.clone().detach().requires_grad_(True)
        y = self.net(x)
        return y, x
