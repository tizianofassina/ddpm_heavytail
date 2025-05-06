import torch 
import torch.nn as nn
import torch.nn.functional as F

"""
We implement a classical UNet model as sketched in the paper 
U-Net: Convolutional Networks for BiomedicalImage Segmentation
arXiv:1505.04597v1
"""



def get_timestep_embedding(timesteps, embedding_dim):
    """
    Create sinusoidal timestep embeddings.
    Args:
        timesteps: Tensor of shape [batch_size] with integer time indices.
        embedding_dim: Dimension of the embedding vector.
    
    Returns:
        Tensor of shape [batch_size, embedding_dim]
    """
    

    half_dim = embedding_dim // 2
    exponent = -math.log(10000) / (half_dim - 1)
    exponents = torch.exp(torch.arange(half_dim, device=timesteps.device) * exponent)
    emb = timesteps[:, None].float() * exponents[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    
    if embedding_dim % 2 == 1:  # if embedding_dim is odd
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


        

class UNet(nn.Module):
    def __init__(self, chan_input = 1, chan_output = 2):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(chan_input, 64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True))
        self.pool_1 = nn.MaxPool2d(2)

        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=0),
            nn.ReLU(inplace=True))
        self.pool_2 = nn.MaxPool2d(2)

        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=0),
            nn.ReLU(inplace=True))
        self.pool_3 = nn.MaxPool2d(2)

        self.conv_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=0),
            nn.ReLU(inplace=True))
        self.pool_4 = nn.MaxPool2d(2)

        self.conv_5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=0),
            nn.ReLU(inplace=True))

        self.upconv_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_6 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=0),
            nn.ReLU(inplace=True))

        self.upconv_2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_7 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=0),
            nn.ReLU(inplace=True))

        self.upconv_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_8 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=0),
            nn.ReLU(inplace=True))

        self.upconv_4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_9 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True))

        self.conv_10 = nn.Conv2d(64, chan_output, kernel_size=1, padding=0)
    
    
    def center_crop(self, layer, target_size):
        _, _, h, w = layer.size()
        diff_y = (h - target_size[0]) // 2
        diff_x = (w - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x):
        x1 = self.conv_1(x)
        p1 = self.pool_1(x1)

        x2 = self.conv_2(p1)
        p2 = self.pool_2(x2)

        x3 = self.conv_3(p2)
        p3 = self.pool_3(x3)

        x4 = self.conv_4(p3)
        p4 = self.pool_4(x4)

        x5 = self.conv_5(p4)

        u1 = self.upconv_1(x5)
        x4_cropped = self.center_crop(x4, u1.shape[2:])
        u1 = torch.cat([x4_cropped, u1], dim=1)
        x6 = self.conv_6(u1)

        u2 = self.upconv_2(x6)
        x3_cropped = self.center_crop(x3, u2.shape[2:])
        u2 = torch.cat([x3_cropped, u2], dim=1)
        x7 = self.conv_7(u2)

        u3 = self.upconv_3(x7)
        x2_cropped = self.center_crop(x2, u3.shape[2:])
        u3 = torch.cat([x2_cropped, u3], dim=1)
        x8 = self.conv_8(u3)

        u4 = self.upconv_4(x8)
        x1_cropped = self.center_crop(x1, u4.shape[2:])
        u4 = torch.cat([x1_cropped, u4], dim=1)
        x9 = self.conv_9(u4)

        out = self.conv_10(x9)
        
        
        return out


class UNet_mnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(chan_input, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)) #  28 x 28
        
        
        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)) #  28 x 28
        self.pool_1 = nn.MaxPool2d(2) # 14 x 14
        
        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)) # 14 x 14
        self.pool_2 = nn.MaxPool2d(2) # 7 x 7
        
        self.upconv_1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 14 x 14
        self.conv_4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)) # 14 x 14
        
        self.upconv_2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)   # 28 x 28
        self.conv_5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, chan_input, kernel_size=1)) # 28 x 28
        
        
    def forward(self, x):
        x1 = self.conv_1(x) #  28 x 28
        

        x2 = self.conv_2(x1) # 28 x 28
        p2 = self.pool_1(x2) # 14 x 14 

        x3 = self.conv_3(p2) # 14 x 14
        p3 = self.pool_2(x3) # 7 x 7
        u3 = self.upconv_1(p3) # 14 x 14
        
        cat_1 = torch.cat([x3, u3], dim=1)
        u2 = self.conv_4(cat_1) # 14 x 14
        
        u2 = self.upconv_2(u2)
        
        cat_2 = torch.cat([x2, u2], dim=1) # 28 x 28
        
        
        return self.conv_5(cat_2) # 28 x 28



class UNet_mnist_time(nn.Module):
    def __init__(self):
        super().__init__(time_emb_dim = 64)
        
        self.linear_1 = nn.Linear(time_emb_dim, 128)
        self.linear_2 = nn.Linear(time_emb_dim, 256)
        self.linear_3 = nn.Linear(time_emb_dim, 128)
        self.linear_4 = nn.Linear(time_emb_dim, 64)
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(chan_input, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)) #  28 x 28
        
        
        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)) #  28 x 28
        self.pool_1 = nn.MaxPool2d(2) # 14 x 14
        
        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)) # 14 x 14
        self.pool_2 = nn.MaxPool2d(2) # 7 x 7
        
        self.upconv_1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 14 x 14
        self.conv_4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)) # 14 x 14
        
        self.upconv_2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)   # 28 x 28
        self.conv_5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, chan_input, kernel_size=1)) # 28 x 28
        
        
    def forward(self, x, t):
        x1 = self.conv_1(x) #  28 x 28
        t_emb = get_timestep_embedding(t, self.time_emb_dim)
        
        t_emb_1 = self.linear_1(t_emb).view(batch_size, 64, 1, 1) # batch x 64 x 1  x 1

        x2 = self.conv_2(x1) + t_emb_1 # 1 x 64 x 28 x 28
        p2 = self.pool_1(x2) # 1 x 64 x 14 x 14 

        x3 = self.conv_3(p2) # 1 x 128 x 14 x 14 
        
        t_emb_2 = self.linear_2(t_emb).view(batch_size, 128, 1, 1) # batch x 128 x 1  x 1
        x3 = x3 + t_emb_2
        
        p3 = self.pool_2(x3) # 7 x 7
        
        u3 = self.upconv_1(p3) # 1 x 256 x 14 x 14
        
        cat_1 = torch.cat([x3, u3], dim=1)
        u2 = self.conv_4(cat_1) # 14 x 14
        
        u2 = self.upconv_2(u2)
        
        cat_2 = torch.cat([x2, u2], dim=1) # 28 x 28
        
        
        return self.conv_5(cat_2) # 28 x 28

        