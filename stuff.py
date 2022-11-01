from turtle import forward
import gym
# from gym.wrappers import TimeLimit
from gym.wrappers.record_video import RecordVideo
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from math import prod


import wandb

class VideoWrapper(gym.Wrapper):
    """Gathers up the frames from an episode and allows to upload them to Weights & Biases
    Thanks to @cyrilibrahim for this snippet
    """

    def __init__(self, env, update_freq=25):
        super(VideoWrapper, self).__init__(env)
        self.episode_images = []
        # we need to store the last episode's frames because by the time we
        # wanna upload them, reset() has juuust been called, so the self.episode_rewards buffer would be empty
        self.last_frames = None

        # we also only render every 20th episode to save framerate
        self.episode_no = 0
        self.render_every_n_episodes = update_freq  # can be modified
        
    def send_wandb_video(self):
        if self.last_frames is None or len(self.last_frames) == 0:
            
            print("Not enough images for GIF. continuing...")
            return
        lf = np.array(self.last_frames)
        print(lf.shape)
        frames = np.swapaxes(lf, 1, 3)
        frames = np.swapaxes(frames, 2, 3)
        wandb.log({"video": wandb.Video(frames, fps=10, format="gif")})
        print("=== Logged GIF")

    def get_wandb_video(self):
        if self.last_frames is None or len(self.last_frames) == 0:
            print("Not enough images for GIF. continuing...")
            return None
        lf = np.array(self.last_frames)
        print(lf.shape)
        frames = np.swapaxes(lf, 1, 3)
        frames = np.swapaxes(frames, 2, 3)
        return frames

    def reset(self, **kwargs):

        self.episode_no += 1
        if self.episode_no == self.render_every_n_episodes:
            print("This is second if")
            
            self.episode_no = 0
            self.last_frames = self.episode_images[:]
            self.episode_images.clear()
            self.send_wandb_video()
            
        if self.episode_no == self.render_every_n_episodes:
            print("This is second if")
            if self.last_frames:
                print("frames are there ")
                print(f"Line 32, len(self.last_frames): {len(self.last_frames)}")
            else:
                print("no framws")
            

        state = self.env.reset()
        

        return state

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)
        # state, reward, done, info = self.env.step(action)

        if self.episode_no + 1 == self.render_every_n_episodes:
            print(f"Line 77, self.episode_no: {self.episode_no}")
            print(f"Line 77, self.episode_no + 1: {self.episode_no + 1}")
            # frame = np.copy(self.env.render("rgb_array"))
            print("saving frames")
            frame = np.copy(self.env.render())
            self.episode_images.append(frame)

        return state, reward, done, truncated, info



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
        

# def reward_pred(obs):
    """_summary_
    """    
    
    


# env = gym.make("CartPole-v1", )
env = gym.make("CartPole-v1", render_mode="rgb_array")
# env = Monitor(env, f'videos/{experiment_name}')
# env = Monitor(env)
# env = RecordVideo(
#     gym.make("CartPole-v1", render_mode="rgb_array"),
#     "video"
# )
env = VideoWrapper(env)
observation, info = env.reset(seed=42)

mode_input_dim = prod(env.observation_space.shape)

mymodel = MyModel(mode_input_dim)
myensemble = MyEnsembleModel(mode_input_dim, 10)

# wandb.init(project="RLPref", entity="alihkw")
wandb.init(project="RLPref", entity="alihkw", monitor_gym = True, save_code=True)
wandb.config = {
  "learning_rate": 0.001,
  "epochs": 100,
  "batch_size": 128
}
wandb.gym.monitor()

for _ in range(10000):
    action = env.action_space.sample()
    # print(f"Line 27, action: {action}")
    # print(f"Line 27, env.action_space:{env.action_space}")
    observation, reward, terminated, truncated, info = env.step(action)
    wandb.log({"action": action})
    wandb.log({"observation": observation})
    
    myreward = myensemble(torch.from_numpy(observation)).detach().numpy()
    # print(f"Line 46, myreward: {myreward}")
    # print(f"Line 9, observation: {observation}")
    # print(reward)
    wandb.log({"reward": reward})
    

    if terminated or truncated:
        observation, info = env.reset()
        # print("HIIIIIII")
env.close()