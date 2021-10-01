# pytorch model speed testing in CPU and GPU

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from tqdm import tqdm
from time import time


class Dummy_model(nn.Module):
    def __init__(self, channel_input, num_output,batch_size):
        super(Dummy_model, self).__init__()
        self.channel_input = channel_input # image channel number, 1 for black and white, 3 for color image
        self.num_output = num_output       # output nodes of the layer for iput of the next layer
        self.batch_size = batch_size
        self.conv1 = torch.nn.Conv2d(in_channels=self.channel_input, kernel_size=3, padding=1, stride=1, out_channels=16) # keep kernel size 3 padding 1 and stride 1 for simplicity sake
        self.conv2 = torch.nn.Conv2d(in_channels=16, kernel_size= 3, padding=1 , out_channels=32)
        self.maxpooling1 = torch.nn.MaxPool2d(kernel_size=2)   
        self.maxpooling2 = torch.nn.MaxPool2d(kernel_size=2)  # maxpolling the conv layer
        self.linear1 = torch.nn.Linear(32 * 57 * 57,100)  # linear layer of the model. original image was 228 , 228. Maxpolling 2 time will get the size 57, 57.
        self.linear2 = torch.nn.Linear(100,num_output)   #logits layer of the network


    def forward(self, image_batch):
        output = F.relu(self.conv1(image_batch))
        output = self.maxpooling1(output)
        output = F.relu(self.conv2(output))
        output = self.maxpooling2(output)
        output = output.view(output.shape[0],-1)
        output = self.linear1(output)
        output = self.linear2(output)
 

        return output


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_funcrion = torch.nn.CrossEntropyLoss()
model = Dummy_model(1,10,10).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=.001)

start_time = time()

for i in tqdm(range(100)):
    image = torch.rand(64,1,228,228).to(device)
    label = torch.randint(low=0,high=10,size=(64,)).to(device)
    output = model(image)
    loss = loss_funcrion(output,label)
    loss.backward()
    optimizer.step()
stop_time = time()
print('time required for 100 loop ', stop_time-start_time)
   







