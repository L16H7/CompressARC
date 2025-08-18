import torch

def encode(self, x):
    x = self.encoder(x)
    mean, logvar = self.mean_layer(x), self.logvar_layer(x)
    return mean, logvar


def reparameterization(self, mean, var):
    epsilon = torch.randn_like(var)
    z = mean + var * epsilon
    return z


def decode(self, x):
    return self.decoder(x)


def forward(self, x):
    mean, log_var = self.encode(x)
    z = self.reparameterization(mean, log_var)
    x_hat = self.decode(z)
    return x_hat, mean, log_var
