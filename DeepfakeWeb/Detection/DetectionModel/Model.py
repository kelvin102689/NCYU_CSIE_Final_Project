from torchvision.models import resnet50, efficientnet_b4
from torch import nn
from torchvision import models
import torch
#from torchvision.models import resnet50, ResNet50_Weights, efficientnet
from torch import nn
from torchvision import models
#from resnest.torch import resnest50
#from pytorch_model_summary import summary
import torch
import ssl
import torch
#from efficientnet_pytorch import EfficientNet

# b5 : 2048, b4 : 1792 b6 : 2304 b3:1536

'''
model =  models.efficientnet_b2(pretrained=True)
Efficientnet = nn.Sequential(*list(model.children())[:])
print(model)
'''

class Model(nn.Module):
    def __init__(self, classes , latent_dim= 2048, lstm_layers=1 , hidden_dim = 2048, bidirectional = True):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)  # Residual Network CNN
        self.model = nn.Sequential(*list(model.children())[:-2])
        #Resnet50 = models.resnet50(pretrained=True).eval().cuda()
        #self.Efficientnet = EfficientNet.from_pretrained('efficientnet-b4').eval().cuda()
        #self.model = nn.Sequential(*list(Resnet50.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.mish =nn.Mish()###
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    def forward(self , x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        #fmap = self.Efficientnet.extract_features(x)
        #fmap = self.Resnet50(x)
        #print(fmap.shape)                 #(torch.Size([seq_length, 2048, 5, 5]))
        x = self.avgpool(fmap)
        #print(x.shape)                    #(torch.Size([seq_length, 2048, 1, 1]))
        x = x.view(batch_size, seq_length, 2048)##
        #x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        #print(x_lstm.shape)                #torch.Size([batch_size, seq_length, 2048])
        output = self.dp(self.linear1(torch.mean(x_lstm, dim=1)))
        #output = self.mish(output)###
        output = self.dp(self.linear2(output))
        output = self.mish(output)
        output = self.dp(self.linear3(output))

        return fmap, output