import torch
import torch.nn as nn


class CALMNet(nn.Module):

    def __init__(self,num_classes=5):

        super(CALMNet,self).__init__()


        self.cnn = nn.Sequential(

            nn.Conv2d(4,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )


        self.lstm = nn.LSTM(
            input_size=16384,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )


        self.fc = nn.Sequential(

            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128,num_classes)
        )


    def forward(self,x):

        b,c,h,t = x.shape

        frames=[]

        win=64
        step=8


        for i in range(0,t-win,step):

            f=x[:,:,:,i:i+win]

            f=self.cnn(f)

            f=f.view(b,-1)

            frames.append(f)


        x=torch.stack(frames,dim=1)

        x,_=self.lstm(x)

        x=x[:,-1,:]

        x=self.fc(x)

        return x