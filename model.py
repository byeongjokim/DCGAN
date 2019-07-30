import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.args.length_z, 1024, )
        )
