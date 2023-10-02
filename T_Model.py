import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CNNNet(nn.Module):
    def __init__(self, num_classes, output_len, output_size):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2, 1), stride=(2, 1), padding=(0, 0)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16))  # 2000*6 -> 1000*6  output_len/2
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 1), stride=(5, 1), padding=(0, 0)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32))  # 200*6 output_len/10
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(5, 1), stride=(5, 1), padding=(0, 0)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),  # 40*6  output_len/50
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))  # 20*3  output_len/100

        self.fc1 = nn.Linear(16 * int(output_len/100) * output_size, 64)  # 16:chanel
        self.fc1_bn = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x[:, None, :, :]  
        x = self.conv1(x)  
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, 1)  
        x = self.fc1_bn(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        # x = self.tanh(x)
        # x = F.relu(x)
        x = F.softplus(x)
        # x = self.maxout(x)
        return x



class LSTM_Segmenter(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(LSTM_Segmenter, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out)
        return torch.sigmoid(out)
    

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )
        
        self.encoder = nn.Sequential(
            conv_block(in_channels, 64),
            nn.MaxPool2d(2, 2),
            conv_block(64, 128),
            nn.MaxPool2d(2, 2),
            conv_block(128, 256),
            nn.MaxPool2d(2, 2),
            conv_block(256, 512)
        )
        
        self.upconv_512_256 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.upblock_256 = conv_block(512, 256)
        self.upconv_256_128 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.upblock_128 = conv_block(256, 128)
        self.upconv_128_64 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.upblock_64 = conv_block(128, 64)
        
        self.out = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        x1 = self.encoder[0:2](x)
        x2 = self.encoder[2:5](x1)
        x3 = self.encoder[5:8](x2)
        x4 = self.encoder[8:](x3)

        # Decoder path
        up3 = self.upblock_256(torch.cat([x3, self.upconv_512_256(x4)], 1))
        up2 = self.upblock_128(torch.cat([x2, self.upconv_256_128(up3)], 1))
        up1 = self.upblock_64(torch.cat([x1, self.upconv_128_64(up2)], 1))

        return self.out(up1)
    


class FCN8s(nn.Module):
    def __init__(self, num_classes):
        super(FCN8s, self).__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=(3,1), padding=(1,0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3,1), padding=(1,0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3,1), padding=(1,0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3,1), padding=(1,0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3,1), padding=(1,0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3,1), padding=(1,0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3,1), padding=(1,0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        )
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3,1), padding=(1,0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3,1), padding=(1,0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3,1), padding=(1,0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        )
        
        self.enc5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3,1), padding=(1,0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3,1), padding=(1,0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3,1), padding=(1,0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        )
        
        # Decoder
        self.dec1 = nn.ConvTranspose2d(512, 512, kernel_size=(3,1), stride=(2,1), padding=(1,0), output_padding=(1,0))
        self.dec2 = nn.ConvTranspose2d(512, 256, kernel_size=(3,1), stride=(2,1), padding=(1,0), output_padding=(1,0))
        self.dec3 = nn.ConvTranspose2d(256, 128, kernel_size=(3,1), stride=(2,1), padding=(1,0), output_padding=(1,0))
        self.dec4 = nn.ConvTranspose2d(128, 64, kernel_size=(3,1), stride=(2,1), padding=(1,0), output_padding=(1,0))
        self.dec5 = nn.ConvTranspose2d(64, 32, kernel_size=(3,1), stride=(2,1), padding=(1,0), output_padding=(1,0))
        
        self.classifier =  nn.Conv2d(32, 1, kernel_size=1)
        
    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(-1)
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        
        dec1 = self.dec1(x5)
        dec2 = self.dec2(dec1 + x4)
        dec3 = self.dec3(dec2 + x3)
        dec4 = self.dec4(dec3 + x2)
        dec5 = self.dec5(dec4 + x1)
        
        return self.classifier(dec5)
