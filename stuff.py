from turtle import forward
import gym
# from gym.wrappers import TimeLimit
from gym.wrappers.record_video import RecordVideo
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import prod
import wandb
from VideoWrapper import VideoWrapper
from models import MyEnsembleModel, MyModel

env = gym.make("CartPole-v1", render_mode="rgb_array")
env = VideoWrapper(env)
observation, info = env.reset(seed=42)

mode_input_dim = prod(env.observation_space.shape)
mymodel = MyModel(mode_input_dim)
myensemble = MyEnsembleModel(mode_input_dim, 10)

wandb.init(project="RLPref", entity="alihkw", monitor_gym=True, save_code=True)
wandb.config = {
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 128
}
wandb.gym.monitor()

for _ in range(2000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    myreward = myensemble(torch.from_numpy(observation)).detach().numpy()

    wandb.log({"myreward": myreward})
    wandb.log({"reward": reward})
    wandb.log({"action": action})
    wandb.log({"observation": observation})

    if terminated or truncated:
        observation, info = env.reset()
env.close()

# def ml_training_loop():
