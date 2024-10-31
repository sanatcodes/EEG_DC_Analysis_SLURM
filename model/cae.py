import torch
import torch.nn as nn

class CAE(nn.Module):
    def __init__(self, latent_dim=6, dropout_conv=0.25, dropout_fc=0.5, leaky_relu_slope=0.2):
        super(CAE, self).__init__()
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(leaky_relu_slope),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(leaky_relu_slope),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_conv),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(leaky_relu_slope),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(leaky_relu_slope),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_conv)
        )
        
        self.fc_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 32 * 32, 1024),
            nn.LeakyReLU(leaky_relu_slope),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout_fc),
            nn.Linear(1024, self.latent_dim)
        )
        
        self.fc_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.LeakyReLU(leaky_relu_slope),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout_fc),
            nn.Linear(1024, 512 * 32 * 32),
            nn.LeakyReLU(leaky_relu_slope),
            nn.BatchNorm1d(512 * 32 * 32)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(leaky_relu_slope),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(leaky_relu_slope),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(leaky_relu_slope),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(leaky_relu_slope),
            nn.BatchNorm2d(32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.ConvTranspose2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        encoded = self.encoder(x)
        latent = self.fc_encoder(encoded)
        decoded = self.fc_decoder(latent)
        decoded = decoded.view(-1, 512, 32, 32)
        reconstructed = self.decoder(decoded)
        return reconstructed, latent
