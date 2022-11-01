import wandb
import numpy as np
import gym

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