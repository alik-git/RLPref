import wandb
import numpy as np
import gym

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import datetime 
import os
import shutil

#  make string with timestamp for labeling
def make_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def convert_numpy_array_to_video(frames):
    # size = 720*16//9, 720
    size = frames.shape[3], frames.shape[2]
    duration = 2
    fps = 25
    video = cv2.VideoWriter(f'{make_timestamp()}output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (size[1], size[0]), False)
    
    for i in range(frames.shape[0]):
        video.write(np.moveaxis(frames[i], [0,1,2], [2,1,0]))
    video.release()
    print("Video saved to video.mp4")
        
    
    # for _ in range(fps * duration):
    #     data = np.random.randint(0, 256, size, dtype='uint8')
    #     out.write(data)
    # out.release()

    # # Create a video writer
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video = cv2.VideoWriter(f'{make_timestamp()}_video.mp4', fourcc, 30, (frames.shape[3], frames.shape[2]))

    # # Write the frames to the video
    # for i in range(frames.shape[0]):
    #     video.write(frames[i])

    # # Close the video writer
    # video.release()
    # print("Video saved to video.mp4")

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
        self.reset_counter = 0
        self.render_every_n_episodes = update_freq  # can be modified
        
    def send_wandb_video(self):
        if self.last_frames is None or len(self.last_frames) == 0:
            
            print("Not enough images for GIF. continuing...")
            return
        lf = np.array(self.last_frames)
        print(lf.shape)
        frames = np.swapaxes(lf, 1, 3)
        frames = np.swapaxes(frames, 2, 3)
        
        # video = convert_numpy_array_to_video(frames)
        
        temp = wandb.Video(frames, fps=10, format="mp4", caption="batman")
        
        shutil.move(temp._path, f"saved_videos/{make_timestamp()}_{wandb.run.name}_rc:{self.reset_counter}_video.mp4")
        # temp.save(f"{make_timestamp()}_video.mp4")

        
        
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
        self.reset_counter += 1

        self.episode_no += 1
        if self.episode_no == self.render_every_n_episodes:
            print("This is second if")
            
            self.episode_no = 0
            self.last_frames = self.episode_images[:]
            self.episode_images.clear()
            self.send_wandb_video()
            

        state = self.env.reset()
        

        return state

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)
        # state, reward, done, info = self.env.step(action)

        if self.episode_no + 1 == self.render_every_n_episodes:
            # frame = np.copy(self.env.render("rgb_array"))
            frame = np.copy(self.env.render())
            self.episode_images.append(frame)

        return state, reward, done, truncated, info