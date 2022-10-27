from turtle import forward
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import prod

class MyEnsembleModel(nn.Module):
    
    def __init__(self, shape, num_models):
        super(MyEnsembleModel, self).__init__()
        self.models = [MyModel(shape) for _ in range(num_models)]
        
    def forward(self, x):
        os = [mymodel(x) for mymodel in self.models]
        print(f"Line 16, os: {os}")
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
        

# def reward_pred(obs):
    """_summary_
    """    
    
    


env = gym.make("CartPole-v1")
observation, info = env.reset(seed=42)

mode_input_dim = prod(env.observation_space.shape)

mymodel = MyModel(mode_input_dim)
myensemble = MyEnsembleModel(mode_input_dim, 10)



for _ in range(1000):
    action = env.action_space.sample()
    print(f"Line 27, action: {action}")
    print(f"Line 27, env.action_space:{env.action_space}")
    observation, reward, terminated, truncated, info = env.step(action)
    
    myreward = myensemble(torch.from_numpy(observation)).detach().numpy()
    print(f"Line 46, myreward: {myreward}")
    print(f"Line 9, observation: {observation}")
    print(reward)

    if terminated or truncated:
        observation, info = env.reset()
env.close()