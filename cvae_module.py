
import torch
from torch import nn
import torch.nn.functional as F

#===========================================================================
# Conditional VAE
#===========================================================================

#===========================================================================
#  Generator, alias Decoder of Conditional VAE
#  Variance of the Normal output is assumed to be known
#  This is the architecture of 
# "Auto-encoding GPS data to reveal individual and collective behaviour"
#===========================================================================
class cCNNGenerator(nn.Module):
    def __init__(self, nz, nc):
        """  Generator, alias Decoder of Conditional VAE with CNN layers.
     Variance of the Normal output is assumed to be known
    This is the architecture of 
    "Auto-encoding GPS data to reveal individual and collective behaviour"
    
    Return a trajectory tensor of shape (-1, 2, 24)

        Args:
            nz (int): The dimension of the latent representation
            nc (_type_): The dimension of the covariates. nc=2 when the day of
            the year is projected on a circle (cos/sin).
        """        
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels = nz + nc,
                      out_channels= (nz+nc)*4,
                      kernel_size=2,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm1d((nz+nc)*4),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose1d(in_channels=(nz+nc)*4,
                               out_channels=(nz+nc)*2,
                               kernel_size=5,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm1d((nz+nc)*2),
            nn.ReLU(True),
            #nn.LeakyReLU(0.2),

            nn.ConvTranspose1d(in_channels=(nz+nc)*2,
                               out_channels=nz+nc,
                               kernel_size=4,
                               stride=2,
                               padding=0,
                               bias=False),
            nn.BatchNorm1d(nz+nc),
            nn.ReLU(True),
            #nn.LeakyReLU(0.2),

            nn.ConvTranspose1d(in_channels=nz+nc,
                               out_channels=2,
                               kernel_size=2,
                               stride=2,
                               padding=0,
                               bias=False)#,
#            nn.Tanh()
        )

    def forward(self, z, c):
        input = torch.concat([z, c], dim = 1)
      #  print(input.shape)
        out = self.cnn(input)
        return out


#===========================================================================
#  Encoder Depth
#  In this architecture, the covariates are added to the channels of the 
#  input image.
#  This is the encoder architecture of 
#  "Auto-encoding GPS data to reveal individual and
#  collective behaviour"
#===========================================================================
class cdepthCNNEncoder(nn.Module):
    def __init__(self, nz, nc):
        """ Decoder of a Conditional VAE with CNN layers.
        In this architecture, the covariates are added to the channels of the 
      input image.
      This is the encoder architecture of 
     "Auto-encoding GPS data to reveal individual and
      collective behaviour"

    Return a list of Z, mu, logvar (value and parameters of the latent distribution) of
    shape [(-1, nz, 1), (-1, nz), (-1, nz)]

        Args:
            nz (int): The dimension of the latent representation
            nc (_type_): The dimension of the covariates. nc=2 when the day of
            the year is projected on a circle (cos/sin).
        """        
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=2+nc,
                      out_channels=8,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=False),


            nn.Conv1d(in_channels=8,
                      out_channels=32,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(in_channels=32,
                      out_channels=128,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(in_channels=128,
                      out_channels=nz,
                      kernel_size=2,
                      stride=1,
                      padding=0,
                      bias=False)
        )

    def reparametrize(self,z):
        mu, log_sigma = z[:, :, 0], z[:, :, 1]
        std = torch.exp(log_sigma/2)
        eps = torch.randn_like(std)
        return mu + eps * std
 
    def forward(self, x, c):
        y = torch.concat([x, c.repeat(1,1,24)], dim = 1)
        z = self.cnn(y)
        out = self.reparametrize(z)
        return out.reshape(out.shape[0], out.shape[1], 1), z[:, :, 0], z[:, :, 1]


#==========================================================================
# Variational auto-encoder
#==========================================================================
#===========================================================================
#  Generator, alias decoder
# 
#===========================================================================
class CNNGenerator(nn.Module):
    def __init__(self, nz):
        """ Decoder of a VAE with CNN layers.
      This is the encoder architecture of 
     "Auto-encoding GPS data to reveal individual and
      collective behaviour"
      
    Return a trajectory tensor of shape (-1, 2, 24)

        Args:
            nz (int): The dimension of the latent representation
        """       
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels = nz,
                      out_channels= nz*4,
                      kernel_size=2,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm1d(nz*4),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose1d(in_channels=nz*4,
                               out_channels=nz*2,
                               kernel_size=5,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm1d(nz*2),
            nn.ReLU(True),

            nn.ConvTranspose1d(in_channels=nz*2,
                               out_channels=nz,
                               kernel_size=4,
                               stride=2,
                               padding=0,
                               bias=False),
            nn.BatchNorm1d(nz),
            nn.ReLU(True),

            nn.ConvTranspose1d(in_channels=nz,
                               out_channels=2,
                               kernel_size=2,
                               stride=2,
                               padding=0,
                               bias=False)#,
        )

    def forward(self, z):
        out = self.cnn(z)
        return out

#===========================================================================
#  Encoder
#===========================================================================
class CNNEncoder(nn.Module):
    def __init__(self, nz):
        """ Encoder of a VAE with CNN layers.
      This is the encoder architecture of 
     "Auto-encoding GPS data to reveal individual and
      collective behaviour"
      
     Return a list of Z, mu, logvar (value and parameters of the latent distribution) of
    shape [(-1, nz, 1), (-1, nz), (-1, nz)]

        Args:
            nz (int): The dimension of the latent representation
        """       
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=2,
                      out_channels=8,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=False),


            nn.Conv1d(in_channels=8,
                      out_channels=32,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(in_channels=32,
                      out_channels=128,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(in_channels=128,
                      out_channels=nz,
                      kernel_size=2,
                      stride=1,
                      padding=0,
                      bias=False)
        )
    
    def reparametrize(self,z):
        mu, log_sigma = z[:, :, 0], z[:, :, 1]
        std = torch.exp(log_sigma/2)
        eps = torch.randn_like(std)
        return mu + eps * std
 
    def forward(self, x):
        z = self.cnn(x)
        out = self.reparametrize(z)
        return out.reshape(out.shape[0], out.shape[1], 1), z[:, :, 0], z[:, :, 1]


#==========================================================================
# Simplified models for visualization purpose
#==========================================================================
class graphcCNNGenerator(nn.Module):
    def __init__(self, nz, nc):
        super().__init__()

        self.conv_norm_act1 = nn.Conv1d(in_channels = nz + nc,
                      out_channels= (nz+nc)*4,
                      kernel_size=2,
                      stride=1,
                      padding=1,
                      bias=False)

        self.convT_norm_act2 = nn.ConvTranspose1d(in_channels=(nz+nc)*4,
                               out_channels=(nz+nc)*2,
                               kernel_size=5,
                               stride=2,
                               padding=1,
                               bias=False)
        
        self.convT_norm_act3 = nn.ConvTranspose1d(in_channels=(nz+nc)*2,
                               out_channels=nz+nc,
                               kernel_size=4,
                               stride=2,
                               padding=0,
                               bias=False)
        
        self.convT4 = nn.ConvTranspose1d(in_channels=nz+nc,
                               out_channels=2,
                               kernel_size=2,
                               stride=2,
                               padding=0,
                               bias=False)

    def forward(self, z, c):
        input = torch.concat([z, c], dim = 1)
        x = self.conv_norm_act1(input)
        x = self.convT_norm_act2(x)
        x = self.convT_norm_act3(x)
        out = self.convT4(x)
        return out
    
    
    
class graphcCNNEncoder(nn.Module):
    def __init__(self, nz, nc):
        super().__init__()

        self.conv_norm_act1 = nn.Conv1d(in_channels=2+nc,
                      out_channels=8,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False)

        self.conv_norm_act2 = nn.Conv1d(in_channels=8,
                      out_channels=32,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False)
        
        self.conv_norm_act3 = nn.Conv1d(in_channels=32,
                      out_channels=128,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=False)

        self.conv4 = nn.Conv1d(in_channels=128,
                      out_channels=nz,
                      kernel_size=2,
                      stride=1,
                      padding=0,
                      bias=False)
        
 
    def forward(self, x, c):
        y = torch.concat([x, c.repeat(1,1,24)], dim = 1)
        y = self.conv_norm_act1(y)
        y = self.conv_norm_act2(y)
        y = self.conv_norm_act3(y)
        out = self.conv4(y)
        return out
