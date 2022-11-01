import torch
import torch.nn as nn
import torch.nn.functional as F

class MyEnsembleModel(nn.Module):
    
    def __init__(self, shape, num_models):
        super(MyEnsembleModel, self).__init__()
        self.models = [MyModel(shape) for _ in range(num_models)]
        
    def forward(self, x):
        os = [mymodel(x) for mymodel in self.models]
        # print(f"Line 16, os: {os}")
        o = torch.mean(torch.stack(os))
        return o


class MyModel(nn.Module):
    
    def __init__(self, shape):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(shape, 64)
        self.layer2 = nn.Linear(64, 1)
        
        self.lrelu1 = nn.LeakyReLU()
        self.lrelu2 = nn.LeakyReLU()
        
    def forward(self, x):
        z = self.lrelu1(self.layer1(x))
        o = self.lrelu2(self.layer2(z))
        return o